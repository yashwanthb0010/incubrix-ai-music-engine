"""
Storage abstract base class interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from pathlib import Path


class BaseStorage(ABC):
    @abstractmethod
    async def save_bytes(self, job_id: str, filename: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Save bytes to storage and return the relative or storage URI."""
        pass

    @abstractmethod
    async def save_file(self, job_id: str, filename: str, src_path: Path, content_type: str = "application/octet-stream") -> str:
        """Save an existing file from disk into storage."""
        pass

    @abstractmethod
    async def get_bytes(self, job_id: str, filename: str) -> bytes:
        """Read bytes for a given job asset."""
        pass

    @abstractmethod
    async def file_exists(self, job_id: str, filename: str) -> bool:
        """Check if asset exists in storage."""
        pass

    @abstractmethod
    def get_local_path(self, job_id: str, filename: str) -> Optional[Path]:
        """Return a local filesystem path if stored locally, or None if remote."""
        pass
