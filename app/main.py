from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.generator import ModelName, generator
from app.services.storage import storage

app = FastAPI(title="ZImage API", version="1.0.0")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "ok", "service": "ZImage API"}

class GenerateRequest(BaseModel):
    prompt: str
    model: ModelName = ModelName.ZIMAGE
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    backend: Optional[str] = None
    compile_model: Optional[bool] = None


@app.post("/generate")
def generate_image_endpoint(request: GenerateRequest):
    try:
        options = request.dict(exclude={"prompt"}, exclude_none=True)
        model = options.pop("model", ModelName.ZIMAGE)

        print(f"[generate] model={model} prompt_length={len(request.prompt)} options={options}")
        image_bytes = generator.generate(
            model=model,
            prompt=request.prompt,
            **options,
        )

        image_url = storage.upload_image(image_bytes)

        return {
            "prompt": request.prompt,
            "model": model.value,
            "image_url": image_url,
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
