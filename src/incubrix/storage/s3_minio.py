"""
MinIO / S3 Object Storage provider for IncuBrix AI Background Music Engine.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Optional
from minio import Minio
from minio.error import S3Error
from incubrix.storage.base import BaseStorage
from incubrix.errors import IncubrixError, ErrorCode


class S3MinioStorage(BaseStorage):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "incubrix-audio",
        secure: bool = False,
    ):
        self.bucket = bucket
        self.endpoint = endpoint
        try:
            self.client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception as e:
            # We don't crash initialization, but log/save state
            self.init_error = str(e)
        else:
            self.init_error = None

    def _object_name(self, job_id: str, filename: str) -> str:
        return f"{job_id}/{filename}"

    async def save_bytes(self, job_id: str, filename: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        if self.init_error:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"S3/MinIO connection error: {self.init_error}",
                status_code=500,
                retryable=True,
            )
        try:
            obj_name = self._object_name(job_id, filename)
            stream = io.BytesIO(data)
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=obj_name,
                data=stream,
                length=len(data),
                content_type=content_type,
            )
            return f"s3://{self.bucket}/{obj_name}"
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"Failed to upload asset '{filename}' to S3/MinIO: {str(e)}",
                status_code=500,
                retryable=True,
            )

    async def save_file(self, job_id: str, filename: str, src_path: Path, content_type: str = "application/octet-stream") -> str:
        data = src_path.read_bytes()
        return await self.save_bytes(job_id, filename, data, content_type)

    async def get_bytes(self, job_id: str, filename: str) -> bytes:
        if self.init_error:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"S3/MinIO connection error: {self.init_error}",
                status_code=500,
                retryable=True,
            )
        obj_name = self._object_name(job_id, filename)
        try:
            response = self.client.get_object(self.bucket, obj_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise IncubrixError(
                    code=ErrorCode.ASSET_NOT_FOUND,
                    message=f"Asset '{filename}' for job '{job_id}' not found in S3.",
                    status_code=404,
                )
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"S3 get_object failed: {str(e)}",
                status_code=500,
                retryable=True,
            )
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"Failed to fetch asset from S3: {str(e)}",
                status_code=500,
                retryable=True,
            )

    async def file_exists(self, job_id: str, filename: str) -> bool:
        obj_name = self._object_name(job_id, filename)
        try:
            self.client.stat_object(self.bucket, obj_name)
            return True
        except Exception:
            return False

    def get_local_path(self, job_id: str, filename: str) -> Optional[Path]:
        return None
