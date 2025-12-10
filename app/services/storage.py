import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv()
logger = logging.getLogger(__name__)


class MinioStorage:
    _client = None
    _bucket = None
    _max_attempts = 3
    _retry_delay = 2.0

    def __init__(self):
        use_minio_flag = os.getenv("MINIO_ENABLED", "true").strip().lower()
        self.use_minio = use_minio_flag not in ("0", "false", "no")

        self.minio_url = os.getenv("MINIO_URL")
        api_source = (
            os.getenv("MINIO_API_URL")
            or os.getenv("MINIO_ENDPOINT")
            or self.minio_url
        )
        self.endpoint, self.secure = self._parse_endpoint(api_source)
        self.public_url = self.minio_url or api_source

        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = os.getenv("MINIO_BUCKET", "genius")
        self._bucket_checked = False
        self.local_image_dir = Path(os.getenv("LOCAL_IMAGE_DIR", "images/generated"))
        self._minio_enabled = self.use_minio and bool(self.endpoint and self.access_key and self.secret_key)
        secure = self.secure

        if self._minio_enabled and not MinioStorage._client:
            MinioStorage._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=secure,
            )
            MinioStorage._bucket = self.bucket_name

    @staticmethod
    def _parse_endpoint(url: Optional[str]) -> tuple[str, bool]:
        if not url:
            return "", False

        normalized = url if "://" in url else f"https://{url}"
        parsed = urlparse(normalized)
        host = parsed.netloc
        secure = parsed.scheme == "https"
        return host, secure

    def _ensure_bucket_exists(self):
        if not self._minio_enabled or not MinioStorage._client:
            self._bucket_checked = True
            return

        try:
            if self._bucket_checked:
                return
            if not MinioStorage._client.bucket_exists(self.bucket_name):
                MinioStorage._client.make_bucket(self.bucket_name)
                logger.info(f"Bucket criado: {self.bucket_name}")
            else:
                logger.info(f"Bucket já existe: {self.bucket_name}")
            self._bucket_checked = True
        except S3Error as err:
            logger.error(f"Erro ao garantir bucket: {err}")
            raise err

    def _build_file_url(self, filename: str) -> str:
        """
        Monta a URL pública final, no mesmo padrão do seu código anterior.
        """
        base_url = self.public_url

        if not base_url and self.endpoint:
            proto = "https" if self.secure else "http"
            base_url = f"{proto}://{self.endpoint}"

        # Ex.: https://minio.playground.dev.br/genius/arquivo.png
        return f"{base_url}/{self.bucket_name}/{filename}"

    def _ensure_local_dir(self) -> Path:
        self.local_image_dir.mkdir(parents=True, exist_ok=True)
        return self.local_image_dir

    def _save_locally(self, filename: str, payload: bytes) -> str:
        dir_path = self._ensure_local_dir()
        file_path = dir_path / filename
        file_path.write_bytes(payload)
        logger.info(f"Imagem salva localmente em {file_path}")
        return str(file_path)

    def upload_image(self, image_data: io.BytesIO, content_type: str = "image/png") -> str:
        """
        Faz upload robusto de uma imagem (BytesIO) com retries,
        logs e URL formatada como no exemplo anterior.
        """
        if not isinstance(image_data, io.BytesIO):
            raise ValueError("image_data deve ser um BytesIO")

        filename = f"{uuid.uuid4()}.png"
        payload = image_data.getvalue()
        file_size = len(payload)
        local_image_path = self._save_locally(filename, payload)

        if not self._minio_enabled:
            return local_image_path

        try:
            self._ensure_bucket_exists()
        except S3Error:
            logger.warning("Falha ao garantir bucket MinIO; retornando caminho local.")
            return local_image_path

        for attempt in range(1, self._max_attempts + 1):
            try:
                stream = io.BytesIO(payload)

                MinioStorage._client.put_object(
                    self.bucket_name,
                    filename,
                    stream,
                    length=file_size,
                    content_type=content_type,
                )

                logger.info(f"Upload concluído: {filename} ({file_size} bytes)")
                return self._build_file_url(filename)

            except S3Error as err:
                if attempt == self._max_attempts:
                    logger.error(f"Erro no upload (tentativa final {attempt}): {err}")
                    logger.warning("Upload MinIO falhou; retornando caminho local.")
                    return local_image_path
                logger.warning(
                    f"Falha no upload (tentativa {attempt}/{self._max_attempts}): {err}. "
                    "Retentando..."
                )
                time.sleep(self._retry_delay * attempt)

            except Exception as err:
                if attempt == self._max_attempts:
                    logger.error(f"Erro inesperado no upload (tentativa final {attempt}): {err}")
                    logger.warning("Upload inesperado falhou; retornando caminho local.")
                    return local_image_path
                logger.warning(
                    f"Erro inesperado no upload (tentativa {attempt}/{self._max_attempts}): {err}. "
                    "Retentando..."
                )
                time.sleep(self._retry_delay * attempt)

        logger.warning("Não foi possível subir para MinIO; usando caminho local.")
        return local_image_path


# Instância reutilizável
storage = MinioStorage()
