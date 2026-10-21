"""
Local disk storage provider for IncuBrix AI Background Music Engine.
Stores assets organized by job_id under a configured directory.
"""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional
from incubrix.storage.base import BaseStorage
from incubrix.errors import IncubrixError, ErrorCode


class LocalStorage(BaseStorage):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_target_dir(self, job_id: str) -> Path:
        target_dir = self.base_dir / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_local_path(self, job_id: str, filename: str) -> Path:
        return self._get_target_dir(job_id) / filename

    async def save_bytes(self, job_id: str, filename: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        try:
            target_path = self.get_local_path(job_id, filename)
            target_path.write_bytes(data)
            return f"local://{job_id}/{filename}"
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"Failed to save asset '{filename}' to local storage: {str(e)}",
                status_code=500,
                retryable=True,
            )

    async def save_file(self, job_id: str, filename: str, src_path: Path, content_type: str = "application/octet-stream") -> str:
        try:
            target_path = self.get_local_path(job_id, filename)
            shutil.copy2(src_path, target_path)
            return f"local://{job_id}/{filename}"
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"Failed to copy asset '{filename}' to local storage: {str(e)}",
                status_code=500,
                retryable=True,
            )

    async def get_bytes(self, job_id: str, filename: str) -> bytes:
        target_path = self.get_local_path(job_id, filename)
        if not target_path.exists():
            raise IncubrixError(
                code=ErrorCode.ASSET_NOT_FOUND,
                message=f"Asset '{filename}' for job '{job_id}' not found.",
                status_code=404,
                retryable=False,
            )
        return target_path.read_bytes()

    async def file_exists(self, job_id: str, filename: str) -> bool:
        return self.get_local_path(job_id, filename).exists()
