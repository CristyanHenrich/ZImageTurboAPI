import io
import logging
import os
from typing import List, Optional

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline
from PIL import Image

from app.services.model_setup import ensure_quantized_model

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _resolve_token() -> Optional[str]:
    return HF_TOKEN


def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


class QwenTextGenerator:
    def __init__(self):
        base_repo = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen-Image")
        dtype_name = os.getenv("QWEN_TEXT_TORCH_DTYPE", "bfloat16").lower()
        torch_dtype = torch.bfloat16 if dtype_name in ("bf16", "bfloat16") else torch.float16
        token = _resolve_token()

        quant_repo = os.getenv("QWEN_TEXT_QUANT_REPO", "nunchaku-tech/nunchaku-qwen-image").strip() or None
        quant_file = os.getenv("QWEN_TEXT_QUANT_FILE", "svdq-int4_r32-qwen-image.safetensors").strip() or None
        model_cache = os.getenv("MODEL_CACHE_DIR", "quantized_models")
        force_quant = _env_enabled("QWEN_QUANT_FORCE")
        target_name = os.getenv("QWEN_TEXT_QUANT_TARGET")
        model_source = base_repo
        if quant_repo and quant_file:
            quant_path = ensure_quantized_model(
                base_repo=base_repo,
                quant_repo=quant_repo,
                quant_filename=quant_file,
                cache_dir=model_cache,
                target_name=target_name,
                force=force_quant,
                token=token,
            )
            model_source = str(quant_path)
        else:
            logger.info("Usando modelo base %s sem quantizacao extra", base_repo)

        self.device = (
            os.getenv("QWEN_TEXT_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.attention_backend = os.getenv("QWEN_TEXT_ATTENTION", "_flash_3")
        self.enable_progress = os.getenv("QWEN_TEXT_PROGRESS", "false").strip().lower() in ("true", "1", "yes")
        self.compile_pipeline = os.getenv("QWEN_TEXT_COMPILE", "false").strip().lower() in ("true", "1", "yes")

        self.pipeline = QwenImagePipeline.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            token=token,
            low_cpu_mem_usage=True,
            torch_compile=self.compile_pipeline,
        )
        self.pipeline.set_progress_bar_config(disable=not self.enable_progress)
        try:
            self.pipeline.set_attention_backend(self.attention_backend)
        except Exception:
            logger.debug("attention backend %s not supported for text pipeline", self.attention_backend)

        self._move_pipeline()

    def _move_pipeline(self):
        if self.device == "cpu":
            self.pipeline.enable_model_cpu_offload()
            return
        try:
            self.pipeline.to(self.device)
        except RuntimeError as exc:
            logger.warning("Erro ao mover text pipeline para %s (%s); ativando offload.", self.device, exc)
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
        for idx in range(n):
            generator = self._build_generator(seed + idx if seed is not None else None)
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            results.append(_image_to_bytes(output.images[0]))
        return results


class QwenEditGenerator:
    def __init__(self):
        base_repo = os.getenv("QWEN_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit")
        dtype_name = os.getenv("QWEN_EDIT_TORCH_DTYPE", "bfloat16").lower()
        torch_dtype = torch.bfloat16 if dtype_name in ("bf16", "bfloat16") else torch.float16
        token = _resolve_token()
        quant_repo = os.getenv("QWEN_EDIT_QUANT_REPO", "nunchaku-tech/nunchaku-qwen-image-edit").strip() or None
        quant_file = os.getenv("QWEN_EDIT_QUANT_FILE", "svdq-int4_r32-qwen-image-edit.safetensors").strip() or None
        model_cache = os.getenv("MODEL_CACHE_DIR", "quantized_models")
        force_quant = _env_enabled("QWEN_QUANT_FORCE")
        target_name = os.getenv("QWEN_EDIT_QUANT_TARGET")
        model_source = base_repo
        if quant_repo and quant_file:
            edit_path = ensure_quantized_model(
                base_repo=base_repo,
                quant_repo=quant_repo,
                quant_filename=quant_file,
                cache_dir=model_cache,
                target_name=target_name,
                force=force_quant,
                token=token,
            )
            model_source = str(edit_path)
        else:
            logger.info("Usando modelo de edição base %s sem quantizacao extra", base_repo)

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            token=token,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        preferred_device = (
            os.getenv("QWEN_EDIT_DEVICE")
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
            logger.warning("Não foi possível mover edição para %s (%s), usando CPU.", self.device, exc)
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
                )
            results.append(_image_to_bytes(output.images[0]))
        return results


text_generator = QwenTextGenerator()
edit_generator = QwenEditGenerator()
