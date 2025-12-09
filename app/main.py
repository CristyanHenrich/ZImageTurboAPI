import base64
import io
import logging
import os
import time
from typing import List, Literal, Optional

import requests
import torch
from diffusers import Flux2Pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import get_token
from PIL import Image
from pydantic import BaseModel

from app.services.storage import storage

# -------------------------------------------------------------------
# ENV & LOGGING
# -------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux2-server")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
FLUX2_REPO = os.getenv("FLUX2_REPO", "black-forest-labs/FLUX.2-dev")

DEVICE = os.getenv("DEVICE") or (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
TORCH_DTYPE = (
    torch.float16
    if DEVICE.startswith("cuda")
    else torch.float16
    if DEVICE == "mps"
    else torch.float32
)


def _resolve_token() -> str:
    token = HF_TOKEN or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN/HUGGINGFACE_TOKEN não definido.")
    return token


# -------------------------------------------------------------------
# REMOTE TEXT ENCODER (recomendado pela Black Forest Labs)
# -------------------------------------------------------------------
def remote_text_encoder(prompts: List[str]) -> torch.Tensor:
    """
    Usa o endpoint remoto oficial do FLUX.2 para gerar prompt_embeds
    a partir de textos. Retorna um tensor no device configurado.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    headers = {
        "Authorization": f"Bearer {_resolve_token()}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
            json={"prompt": prompts},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.error("Erro no remote_text_encoder: %s", exc)
        raise HTTPException(status_code=502, detail="Falha no encoder remoto.")

    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds.to(DEVICE)


# -------------------------------------------------------------------
# CARREGANDO O FLUX.2 DEV
# -------------------------------------------------------------------
logger.info("Carregando pipeline FLUX.2 a partir de %s em %s...", FLUX2_REPO, DEVICE)
pipe: Flux2Pipeline = Flux2Pipeline.from_pretrained(
    FLUX2_REPO,
    text_encoder=None,  # usamos encoder remoto
    torch_dtype=TORCH_DTYPE,
    token=_resolve_token(),
).to(DEVICE)

pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
logger.info("Pipeline FLUX.2 carregado.")

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(title="FLUX2 Image Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# MODELOS
# -------------------------------------------------------------------
class ImageGenerationRequest(BaseModel):
    prompt: str
    n: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 4.0
    seed: Optional[int] = None
    response_format: Literal["url", "b64_json"] = "url"


class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]


def pil_to_bytes(pil_image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
def generate_image(body: ImageGenerationRequest):
    """
    Geração de imagem a partir de texto com FLUX.2-dev quantizado.
    Estilo OpenAI Images API.
    """
    logger.info("[txt2img] prompt='%s' n=%s", body.prompt, body.n)

    if body.n < 1 or body.n > 8:
        raise HTTPException(status_code=400, detail="n deve estar entre 1 e 8")

    prompt_embeds = remote_text_encoder([body.prompt] * body.n)

    generator = None
    if body.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(body.seed)

    with torch.inference_mode():
        out = pipe(
            prompt_embeds=prompt_embeds,
            generator=generator,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            height=body.height,
            width=body.width,
        )
        images = out.images

    data: List[ImageObject] = []
    for img in images:
        buf = pil_to_bytes(img)
        if body.response_format == "url":
            url = storage.upload_image(buf, content_type="image/png")
            data.append(ImageObject(url=url))
        else:
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data.append(ImageObject(b64_json=b64))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=data,
    )


@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(0.8),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    seed: Optional[int] = Form(None),
    response_format: Literal["url", "b64_json"] = Form("url"),
):
    """
    Edição de imagem (image-to-image) com FLUX.2-dev.
    """
    logger.info("[img2img] prompt='%s' strength=%s file=%s", prompt, strength, image.filename)

    try:
        contents = await image.read()
        init_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")

    prompt_embeds = remote_text_encoder([prompt])

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.inference_mode():
        out = pipe(
            prompt_embeds=prompt_embeds,
            image=init_image,
            strength=strength,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        result_image = out.images[0]

    buf = pil_to_bytes(result_image)

    data: List[ImageObject] = []
    if response_format == "url":
        url = storage.upload_image(buf, content_type="image/png")
        data.append(ImageObject(url=url))
    else:
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data.append(ImageObject(b64_json=b64))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=data,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": FLUX2_REPO, "device": DEVICE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
