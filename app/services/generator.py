import torch
import io
from diffusers import ZImagePipeline

class ImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # If running on macOS for testing, it might fallback to mps or cpu
        if torch.backends.mps.is_available():
            self.device = "mps"
        
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
             # CPU offload is good for lower VRAM or CPU usage
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str) -> io.BytesIO:
        # Default parameters from the user's original script
        image = self.pipe(
            prompt=prompt,
            height=736,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator(self.device).manual_seed(42), 
        ).images[0]
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

generator = ImageGenerator()
