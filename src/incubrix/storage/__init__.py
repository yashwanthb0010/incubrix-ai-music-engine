"""
Storage factory for IncuBrix.
"""

from typing import Optional
from incubrix.config import Settings, get_settings
from incubrix.storage.base import BaseStorage
from incubrix.storage.local import LocalStorage
from incubrix.storage.s3_minio import S3MinioStorage

_storage_singleton: Optional[BaseStorage] = None


def get_storage(settings: Optional[Settings] = None) -> BaseStorage:
    global _storage_singleton
    if _storage_singleton is not None:
        return _storage_singleton

    cfg = settings or get_settings()
    if cfg.storage_type == "s3":
        _storage_singleton = S3MinioStorage(
            endpoint=cfg.s3_endpoint,
            access_key=cfg.s3_access_key,
            secret_key=cfg.s3_secret_key,
            bucket=cfg.s3_bucket,
            secure=cfg.s3_secure,
        )
    else:
        _storage_singleton = LocalStorage(cfg.storage_local_dir)

    return _storage_singleton
