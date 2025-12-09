from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.generator import generator
from app.services.storage import storage

app = FastAPI(title="ZImage API", version="1.0.0")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "ok", "service": "ZImage API"}

@app.post("/generate")
def generate_image_endpoint(request: PromptRequest):
    try:
        # 1. Generate Image
        print(f"Generating image for prompt: {request.prompt}")
        image_bytes = generator.generate(request.prompt)
        
        # 2. Upload to MinIO
        print("Uploading to MinIO...")
        image_url = storage.upload_image(image_bytes)
        
        return {
            "prompt": request.prompt,
            "image_url": image_url
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
