import base64
import io
import logging
import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from app.services.generator import generator
from app.services.storage import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-service")

app = FastAPI(title="Qwen Edit API", version="1.0.0")

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
    data: list[ImageObject]


@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    true_cfg_scale: float = Form(4.0),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
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
        logger.error("Falha ao abrir imagem enviada: %s", exc)
        raise HTTPException(status_code=400, detail="Não foi possível processar a imagem.")

    try:
        generated_images = generator.generate(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            n=n,
        )
    except Exception as exc:
        logger.error("Erro no runtime do modelo: %s", exc)
        raise HTTPException(status_code=500, detail="Falha ao gerar imagem.")

    data = []
    for image_bytes in generated_images:
        if response_format == "url":
            url = storage.upload_image(io.BytesIO(image_bytes), content_type="image/png")
            data.append(ImageObject(url=url))
        else:
            data.append(ImageObject(b64_json=base64.b64encode(image_bytes).decode("utf-8")))

    return ImageGenerationResponse(
        created=int(time.time()),
        prompt=prompt,
        data=data,
    )


@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok",
            "model": os.getenv("QWEN_MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit"),
            "device": os.getenv("QWEN_DEVICE"),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
