import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def ensure_quantized_model(
    base_repo: str,
    quant_repo: str,
    quant_filename: str,
    *,
    cache_dir: Optional[str] = None,
    target_name: Optional[str] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Path:
    """
    Garante que um modelo quantizado tenha a mesma estrutura de um repo Diffusers.
    Copia o repo base (base_repo) para um diretório local e sobrescreve os pesos com o safetensors
    quantizado baixado de quant_repo/quant_filename.
    """
    cache_root = Path(cache_dir or os.getenv("MODEL_CACHE_DIR", "quantized_models"))
    cache_root.mkdir(parents=True, exist_ok=True)

    target = cache_root / (target_name or quant_repo.replace("/", "_"))
    if target.exists() and not force:
        return target

    logger.info("Preparando quantizado %s para %s (base %s)", quant_filename, target, base_repo)

    base_path = Path(
        snapshot_download(
            repo_id=base_repo,
            cache_dir=str(cache_root),
            resume_download=True,
            token=token,
        )
    )

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base_path, target)

    quant_path = Path(
        hf_hub_download(
            repo_id=quant_repo,
            filename=quant_filename,
            cache_dir=str(cache_root),
            resume_download=True,
            token=token,
        )
    )

    unet_dir = target / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    for old in unet_dir.glob("diffusion_pytorch_model*.safetensors"):
        old.unlink()
    dest = unet_dir / "diffusion_pytorch_model.safetensors"
    shutil.copyfile(quant_path, dest)

    logger.info("Modelo quantizado disponível em %s", target)
    return target
