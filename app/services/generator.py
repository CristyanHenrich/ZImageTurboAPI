import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class GGUFFluxGenerator:
    def __init__(self):
        self.model_path = Path(os.getenv("FLUX_GGUF_MODEL_PATH", "flux2-dev-gguf.Q4_K_M.gguf"))
        if not self.model_path.exists():
            raise RuntimeError(
                "O arquivo GGUF não foi encontrado em "
                f"'{self.model_path}'. Defina FLUX_GGUF_MODEL_PATH."
            )

        cli_path = Path(os.getenv("FLUX_GGUF_CLI", "flux2-gguf"))
        resolved_cli = shutil.which(str(cli_path))
        if resolved_cli:
            self.cli = Path(resolved_cli)
        elif cli_path.exists():
            self.cli = cli_path
        else:
            raise RuntimeError(
                f"CLI GGUF não disponível em '{cli_path}'. "
                "Instale o runtime (p. ex. ggml/flux2) e defina FLUX_GGUF_CLI."
            )

        self.device = os.getenv("FLUX_GGUF_DEVICE", "cuda:0")
        self.default_height = int(os.getenv("FLUX_GGUF_HEIGHT", "1024"))
        self.default_width = int(os.getenv("FLUX_GGUF_WIDTH", "1024"))
        self.default_steps = int(os.getenv("FLUX_GGUF_STEPS", "28"))
        self.default_guidance = float(os.getenv("FLUX_GGUF_GUIDANCE", "4.0"))
        self.default_strength = float(os.getenv("FLUX_GGUF_STRENGTH", "0.8"))
        self.max_retries = int(os.getenv("FLUX_GGUF_RETRIES", "2"))
        self.extra_args = shlex.split(os.getenv("FLUX_GGUF_EXTRA_ARGS", ""))

        self.env = os.environ.copy()
        cuda_devices = os.getenv("FLUX_GGUF_CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            self.env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    def _build_command(
        self,
        prompt: str,
        output_path: Path,
        height: int,
        width: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        init_image_path: Optional[Path],
        strength: float,
    ) -> List[str]:
        cmd = [
            str(self.cli),
            "--model",
            str(self.model_path),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(steps),
            "--guidance",
            str(guidance),
            "--device",
            self.device,
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if init_image_path:
            cmd.extend(
                [
                    "--init-image",
                    str(init_image_path),
                    "--strength",
                    str(strength),
                ]
            )

        if self.extra_args:
            cmd.extend(self.extra_args)

        return cmd

    def _dump_temp_image(self, image_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            tmp.write(image_bytes)
            tmp.flush()
        finally:
            tmp.close()
        return Path(tmp.name)

    def _generate_once(
        self,
        prompt: str,
        height: int,
        width: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        init_image: Optional[bytes],
        strength: float,
    ) -> bytes:
        init_path: Optional[Path] = None
        if init_image:
            init_path = self._dump_temp_image(init_image)

        out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        out_tmp.close()
        output_path = Path(out_tmp.name)

        last_error: Optional[subprocess.CalledProcessError] = None

        try:
            for attempt in range(1, self.max_retries + 1):
                cmd = self._build_command(
                    prompt=prompt,
                    output_path=output_path,
                    height=height,
                    width=width,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    init_image_path=init_path,
                    strength=strength,
                )

                logger.info("Flux GGUF runner: %s", " ".join(shlex.quote(p) for p in cmd))

                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        env=self.env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    return output_path.read_bytes()
                except subprocess.CalledProcessError as err:
                    last_error = err
                    logger.warning(
                        "Tentativa %s/%s falhou: %s",
                        attempt,
                        self.max_retries,
                        err.stderr.strip(),
                    )
                    if attempt == self.max_retries:
                        raise
        finally:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            if init_path and init_path.exists():
                init_path.unlink(missing_ok=True)

        assert last_error is not None  # mypy
        raise last_error

    def generate(
        self,
        prompt: str,
        n: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        image_bytes: Optional[bytes] = None,
        strength: Optional[float] = None,
    ) -> List[bytes]:
        height = height or self.default_height
        width = width or self.default_width
        steps = num_inference_steps or self.default_steps
        guidance = guidance_scale or self.default_guidance
        strength = strength or self.default_strength

        results: List[bytes] = []
        for _ in range(n):
            results.append(
                self._generate_once(
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    init_image=image_bytes,
                    strength=strength,
                )
            )

        return results


generator = GGUFFluxGenerator()
