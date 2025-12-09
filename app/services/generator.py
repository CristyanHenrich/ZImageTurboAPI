import io

import torch
from diffusers import ZImagePipeline

from app.services.storage import storage


class ImageGenerator:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Loading model on {self.device}...")
        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.device == "cuda":
            self.pipe.to("cuda")
        elif self.device == "mps":
            self.pipe.to("mps")
        else:
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str) -> io.BytesIO:
        image = self.pipe(
            prompt=prompt,
            height=736,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator(self.device).manual_seed(42),
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def generate_and_upload(self, prompt: str) -> str:
        img_bytes = self.generate(prompt)
        return storage.upload_image(img_bytes, content_type="image/png")


# Instância reutilizável
generator = ImageGenerator()
