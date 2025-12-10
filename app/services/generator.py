import io
import logging
import os
from typing import List, Optional

import torch
from diffusers import QwenImageEditPlusPipeline, ZImagePipeline
from PIL import Image

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _resolve_token() -> Optional[str]:
    return HF_TOKEN


def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class ZImageGenerator:
    def __init__(self):
        model_id = os.getenv("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
        dtype_name = os.getenv("ZIMAGE_TORCH_DTYPE", "float16").lower()
        torch_dtype = torch.float16 if dtype_name in ("float16", "fp16") else torch.float32
        token = _resolve_token()

        self.device = (
            os.getenv("ZIMAGE_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.attention_backend = os.getenv("ZIMAGE_ATTENTION", "_flash_3")
        self.enable_progress = os.getenv("ZIMAGE_PROGRESS", "false").strip().lower() in ("true", "1", "yes")
        self.compile_pipeline = os.getenv("ZIMAGE_COMPILE", "false").strip().lower() in ("true", "1", "yes")

        self.pipeline = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=token,
            low_cpu_mem_usage=True,
            torch_compile=self.compile_pipeline,
        )

        self.pipeline.set_progress_bar_config(disable=not self.enable_progress)
        try:
            self.pipeline.set_attention_backend(self.attention_backend)
        except Exception:
            logger.debug("attention backend %s not supported", self.attention_backend)

        if self.device == "cpu":
            self.pipeline.enable_model_cpu_offload()
        else:
            try:
                self.pipeline.to(self.device)
            except RuntimeError as exc:
                logger.warning("Erro ao mover ZImage para %s (%s); forcando offload.", self.device, exc)
                self.pipeline.enable_model_cpu_offload()
                self.device = "cpu"

    def _build_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(seed)

    def generate(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        n: int = 1,
    ) -> List[bytes]:
        results: List[bytes] = []
        for attempt in range(n):
            generator = self._build_generator(seed + attempt if seed is not None else None)
            with torch.inference_mode():
                image = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            results.append(_image_to_bytes(image))
        return results


class QwenPipelineGenerator:
    def __init__(self):
        model_id = os.getenv("QWEN_MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit")
        dtype_name = os.getenv("QWEN_TORCH_DTYPE", "bfloat16").lower()
        torch_dtype = torch.bfloat16 if dtype_name in ("bf16", "bfloat16") else torch.float16
        token = _resolve_token()

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=token,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        preferred_device = (
            os.getenv("QWEN_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = preferred_device
        self._move_pipeline()

    def _move_pipeline(self):
        if self.device == "cpu":
            self.pipeline.enable_model_cpu_offload()
            return
        try:
            self.pipeline.to(self.device)
        except RuntimeError as exc:
            logger.warning("Não foi possível mover pipeline para %s (%s), usando CPU.", self.device, exc)
            self.pipeline.enable_model_cpu_offload()
            self.device = "cpu"

    def _build_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(seed)

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        *,
        num_inference_steps: int,
        true_cfg_scale: float,
        negative_prompt: str,
        seed: Optional[int],
        strength: float,
        n: int = 1,
    ) -> List[bytes]:
        results: List[bytes] = []
        for idx in range(n):
            generator = self._build_generator(seed + idx if seed is not None else None)
            with torch.inference_mode():
                output = self.pipeline(
                    image=image,
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                )
            results.append(_image_to_bytes(output.images[0]))
        return results


zimage_generator = ZImageGenerator()
qwen_generator = QwenPipelineGenerator()
