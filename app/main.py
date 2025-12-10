import base64
import io
import logging
import os
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from app.services.generator import qwen_generator, zimage_generator
from app.services.storage import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zimage-service")

app = FastAPI(title="ZImage + Qwen API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    prompt: str
    data: List[ImageObject]


class TextGenerationRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    seed: Optional[int] = None
    n: int = 1
    response_format: Literal["url", "b64_json"] = "url"


def _build_response(prompt: str, response_format: Literal["url", "b64_json"], images: List[bytes]) -> ImageGenerationResponse:
    data: List[ImageObject] = []
    for image_bytes in images:
        if response_format == "url":
            url = storage.upload_image(io.BytesIO(image_bytes), content_type="image/png")
            data.append(ImageObject(url=url))
        else:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data.append(ImageObject(b64_json=encoded))
    return ImageGenerationResponse(
        created=int(time.time()),
        prompt=prompt,
        data=data,
    )


@app.post("/generate", response_model=ImageGenerationResponse)
def generate_image(body: TextGenerationRequest):
    if body.n < 1 or body.n > 4:
        raise HTTPException(status_code=400, detail="n deve estar entre 1 e 4")

    try:
        images = zimage_generator.generate(
            prompt=body.prompt,
            height=body.height,
            width=body.width,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            seed=body.seed,
            n=body.n,
        )
    except Exception as exc:
        logger.error("Falha no Z-Image: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao gerar imagem.")

    return _build_response(body.prompt, body.response_format, images)


@app.post("/edit", response_model=ImageGenerationResponse)
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    true_cfg_scale: float = Form(4.0),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    strength: float = Form(0.8),
    n: int = Form(1),
    seed: Optional[int] = Form(None),
    response_format: Literal["url", "b64_json"] = Form("url"),
):
    if n < 1 or n > 4:
        raise HTTPException(status_code=400, detail="n precisa estar entre 1 e 4")

    try:
        contents = await image.read()
        init_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.error("Falha ao decodificar imagem: %s", exc)
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    try:
        images = qwen_generator.generate(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            n=n,
        )
    except Exception as exc:
        logger.error("Erro no Qwen edit: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao editar imagem.")

    return _build_response(prompt, response_format, images)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "zimage": os.getenv("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo"),
            "qwen": os.getenv("QWEN_MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit"),
        },
        "device": {
            "zimage": os.getenv("ZIMAGE_DEVICE"),
            "qwen": os.getenv("QWEN_DEVICE"),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
