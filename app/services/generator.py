import io
import logging
import os
from enum import Enum
from typing import Optional

import requests
import torch
from diffusers import Flux2Pipeline, ZImagePipeline
from huggingface_hub import get_token

from app.services.storage import storage

logger = logging.getLogger(__name__)


class ModelName(str, Enum):
    ZIMAGE = "zimage"
    FLUX = "flux"


class ZImageGenerator:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        # !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`.
        # This would give you sub-second generation speed on Hopper GPUs (H100/H200/H800) after warm-up.
        self.backend = "_flash_3"
        self.compile_model = True

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.compile_model:
            pipe_kwargs["torch_compile"] = True

        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            **pipe_kwargs,
        )

        if self.device == "cuda":
            self.pipe.to("cuda")
        elif self.device == "mps":
            self.pipe.to("mps")
        else:
            self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        backend: Optional[str] = None,
        compile_model: Optional[bool] = None,
    ) -> io.BytesIO:
        height = height or 736
        width = width or 1024
        num_inference_steps = num_inference_steps or 9
        guidance_scale = 0.0 if guidance_scale is None else guidance_scale
        backend = backend or self.backend
        compile_model = self.compile_model if compile_model is None else compile_model

        if backend != self.backend or compile_model != self.compile_model:
            logger.info(
                "ZImage overridden settings: backend=%s compile=%s",
                backend,
                compile_model,
            )

        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(self.device).manual_seed(42),
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer


class FluxGenerator:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.repo_id = os.getenv("FLUX_MODEL_ID", "diffusers/FLUX.2-dev")
        self.remote_text_encoder_url = os.getenv(
            "FLUX_REMOTE_ENCODER",
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
        )

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.pipe = Flux2Pipeline.from_pretrained(
            self.repo_id,
            text_encoder=None,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device)

    def _remote_text_encoder(self, prompt: str) -> torch.Tensor:
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or get_token()
        if not token:
            raise RuntimeError("Hugging Face token required to use FLUX text encoder.")

        response = requests.post(
            self.remote_text_encoder_url,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        response.raise_for_status()

        buffer = io.BytesIO(response.content)
        embeds = torch.load(buffer, map_location=self.device)
        return embeds

    def generate(
        self,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **_,
    ) -> io.BytesIO:
        num_inference_steps = num_inference_steps or 50
        guidance_scale = 4.0 if guidance_scale is None else guidance_scale

        prompt_embeds = self._remote_text_encoder(prompt)

        image = self.pipe(
            prompt_embeds=prompt_embeds,
            generator=torch.Generator(self.device).manual_seed(42),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer


class ImageGeneratorManager:
    def __init__(self):
        self._zimage = ZImageGenerator()
        self._flux = FluxGenerator()

    def generate(
        self,
        *,
        model: ModelName,
        prompt: str,
        **kwargs,
    ) -> io.BytesIO:
        generator = self._flux if model == ModelName.FLUX else self._zimage
        return generator.generate(prompt=prompt, **kwargs)


# Instância reutilizável
generator = ImageGeneratorManager()
