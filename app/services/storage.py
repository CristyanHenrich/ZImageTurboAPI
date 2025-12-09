import io
import logging
import os
import time
import uuid

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
        self.minio_url = os.getenv("MINIO_URL")
        self.endpoint = self._extract_endpoint(os.getenv("MINIO_ENDPOINT") or self.minio_url)
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = os.getenv("MINIO_BUCKET", "genius")

        secure = str(self.minio_url).startswith("https")

        if not MinioStorage._client:
            MinioStorage._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=secure,
            )
            MinioStorage._bucket = self.bucket_name
            self._ensure_bucket_exists()

    @staticmethod
    def _extract_endpoint(url: str) -> str:
        if not url:
            return ""
        return url.replace("https://", "").replace("http://", "")

    def _ensure_bucket_exists(self):
        try:
            if not MinioStorage._client.bucket_exists(self.bucket_name):
                MinioStorage._client.make_bucket(self.bucket_name)
                logger.info(f"Bucket criado: {self.bucket_name}")
            else:
                logger.info(f"Bucket já existe: {self.bucket_name}")
        except S3Error as err:
            logger.error(f"Erro ao garantir bucket: {err}")
            raise err

    def _build_file_url(self, filename: str) -> str:
        """
        Monta a URL pública final, no mesmo padrão do seu código anterior.
        """
        base_url = os.getenv("MINIO_URL")

        if not base_url:
            base_url = f"https://{self.endpoint}"

        # Ex.: https://minio.playground.dev.br/genius/arquivo.png
        return f"{base_url}/{self.bucket_name}/{filename}"

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
                    raise err
                logger.warning(
                    f"Falha no upload (tentativa {attempt}/{self._max_attempts}): {err}. "
                    "Retentando..."
                )
                time.sleep(self._retry_delay * attempt)

            except Exception as err:
                if attempt == self._max_attempts:
                    logger.error(f"Erro inesperado no upload (tentativa final {attempt}): {err}")
                    raise err
                logger.warning(
                    f"Erro inesperado no upload (tentativa {attempt}/{self._max_attempts}): {err}. "
                    "Retentando..."
                )
                time.sleep(self._retry_delay * attempt)

        raise RuntimeError("Falha inesperada: não deveria chegar aqui.")


# Instância reutilizável
storage = MinioStorage()
