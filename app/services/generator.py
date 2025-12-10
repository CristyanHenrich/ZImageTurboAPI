import logging
from io import BytesIO
import os
from typing import List, Optional

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

logger = logging.getLogger(__name__)


_TORCH_DTYPES = {
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class QwenPipelineGenerator:
    def __init__(self):
        model_id = os.getenv("QWEN_MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit")
        dtype_name = os.getenv("QWEN_TORCH_DTYPE", "bfloat16").lower()
        torch_dtype = _TORCH_DTYPES.get(dtype_name, torch.bfloat16)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

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
        self._move_to_device()

    def _move_to_device(self):
        if self.device == "cpu":
            self.pipeline.enable_model_cpu_offload()
            return

        try:
            self.pipeline.to(self.device)
        except Exception as exc:
            logger.warning(
                "Falha ao mover pipeline para %s (%s); usando cpu/offload.",
                self.device,
                exc,
            )
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
            generator = self._build_generator(
                seed + idx if seed is not None else None
            )
            with torch.inference_mode():
                output = self.pipeline(
                    image=image,
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                )
            buffer = BytesIO()
            output.images[0].save(buffer, format="PNG")
            results.append(buffer.getvalue())
        return results


generator = QwenPipelineGenerator()
