import os
import io
import uuid
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

class MinioStorage:
    def __init__(self):
        self.minio_url = os.getenv("MINIO_URL")
        self.endpoint = os.getenv("MINIO_ENDPOINT")
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = os.getenv("MINIO_BUCKET", "genius")
        
        secure = os.getenv("MINIO_URL", "").startswith("https")

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=secure
        )
        
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as err:
            print(f"Error ensuring bucket exists: {err}")

    def upload_image(self, image_data: io.BytesIO, content_type: str = "image/png") -> str:
        filename = f"{uuid.uuid4()}.png"
        file_size = image_data.getbuffer().nbytes
        image_data.seek(0)
        
        try:
            self.client.put_object(
                self.bucket_name,
                filename,
                image_data,
                file_size,
                content_type=content_type
            )
            # return formatted url or presigned url
            # Constructing public URL directly as requested implies public bucket or direct link usage
            # Use MINIO_URL or MINIO_ENDPOINT to construct the full URL
            base_url = os.getenv("MINIO_URL")
            if not base_url:
                 # fallback to endpoint/bucket style if no specific view url
                 base_url = f"https://{self.endpoint}/{self.bucket_name}"
            
            # If using playground dev, it might be bucket.domain or domain/bucket.
            # Based on user input: MINIO_URL=https://minio.playground.dev.br
            # Let's assume standard access: https://minio.playground.dev.br/genius/filename
            
            return f"{base_url}/{self.bucket_name}/{filename}"
            
        except S3Error as err:
            print(f"Error uploading file: {err}")
            raise err

storage = MinioStorage()
