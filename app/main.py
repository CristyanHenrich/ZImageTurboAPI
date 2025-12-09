from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.generator import ModelName, generator
from app.services.storage import storage

app = FastAPI(title="ZImage API", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str
    model: ModelName = ModelName.ZIMAGE
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    backend: Optional[str] = None
    compile_model: Optional[bool] = None

    @classmethod
    def as_form(
        cls,
        prompt: str = Form(...),
        model: ModelName = Form(ModelName.ZIMAGE),
        height: Optional[int] = Form(None),
        width: Optional[int] = Form(None),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        backend: Optional[str] = Form(None),
        compile_model: Optional[bool] = Form(None),
    ) -> "GenerateRequest":
        return cls(
            prompt=prompt,
            model=model,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            backend=backend,
            compile_model=compile_model,
        )


@app.get("/")
def read_root():
    return {"status": "ok", "service": "ZImage API"}


@app.post("/generate")
async def generate_image_endpoint(
    request: GenerateRequest = Depends(GenerateRequest.as_form),
    image: Optional[UploadFile] = File(None),
):
    try:
        image_bytes = await image.read() if image else None
        options = request.dict(
            exclude={"prompt"}, exclude_none=True
        )
        model = options.pop("model", ModelName.ZIMAGE)

        print(
            f"[generate] model={model} prompt_length={len(request.prompt)} image_bytes={bool(image_bytes)} options={options}"
        )
        generated_bytes = generator.generate(
            model=model,
            prompt=request.prompt,
            image_bytes=image_bytes,
            **options,
        )

        image_url = storage.upload_image(generated_bytes)

        return {
            "prompt": request.prompt,
            "model": model.value,
            "image_url": image_url,
            "local_image": str(storage.local_image_dir),
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
