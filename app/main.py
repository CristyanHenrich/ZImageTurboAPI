import base64
import io
import logging
import os
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.generator import generator
from app.services.storage import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zimage-gguf")

app = FastAPI(title="ZZ-GGUF Flux2", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
def generate_image(body: ImageGenerationRequest):
    logger.info("[gguf txt2img] prompt='%s' n=%s height=%s width=%s", body.prompt, body.n, body.height, body.width)

    if body.n < 1 or body.n > 8:
        raise HTTPException(status_code=400, detail="n deve estar entre 1 e 8")

    try:
        images = generator.generate(
            prompt=body.prompt,
            n=body.n,
            height=body.height,
            width=body.width,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            seed=body.seed,
        )
    except Exception as exc:
        logger.error("Falha ao gerar imagem: %s", exc)
        raise HTTPException(status_code=500, detail="Erro no runtime GGUF")

    data: List[ImageObject] = []
    for image_bytes in images:
        if body.response_format == "url":
            url = storage.upload_image(io.BytesIO(image_bytes), content_type="image/png")
            data.append(ImageObject(url=url))
        else:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data.append(ImageObject(b64_json=encoded))

    return ImageGenerationResponse(created=int(time.time()), data=data)


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
    logger.info("[gguf img2img] prompt='%s' strength=%s file=%s", prompt, strength, image.filename)

    try:
        contents = await image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")

    try:
        generated = generator.generate(
            prompt=prompt,
            n=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image_bytes=contents,
            strength=strength,
        )
    except Exception as exc:
        logger.error("Falha no runtime GGUF para edições: %s", exc)
        raise HTTPException(status_code=500, detail="Erro no runtime GGUF")

    image_bytes = generated[0]

    if response_format == "url":
        url = storage.upload_image(io.BytesIO(image_bytes), content_type="image/png")
        data = [ImageObject(url=url)]
    else:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data = [ImageObject(b64_json=encoded)]

    return ImageGenerationResponse(created=int(time.time()), data=data)


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "gguf", "device": os.getenv("FLUX_GGUF_DEVICE")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
