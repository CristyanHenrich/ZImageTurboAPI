import torch
import io
from diffusers import ZImagePipeline
from storage import MinioStorage


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
            # otimização para CPU
            self.pipe.enable_model_cpu_offload()

    # -------------------------------------------------------------------------
    # Gera a imagem somente em memória (BytesIO)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Gera a imagem e faz upload no MinIO usando sua nova classe
    # -------------------------------------------------------------------------
    def generate_and_upload(self, prompt: str) -> str:
        """
        Retorna a URL pública da imagem gerada.
        """
        img_bytes = self.generate(prompt)

        url = MinioStorage.upload_image(
            img_bytes,
            content_type="image/png"
        )

        return url


# Instância reutilizável
generator = ImageGenerator()
