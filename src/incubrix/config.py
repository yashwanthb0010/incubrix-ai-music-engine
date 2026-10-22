"""
Configuration settings for IncuBrix AI Background Music Engine.
Loads environment variables safely and provides strongly-typed configuration.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    # App Information
    app_name: str = Field(default_factory=lambda: os.getenv("APP_NAME", "incubrix-music-engine"))
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() in ("true", "1", "yes"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    base_url: str = Field(default_factory=lambda: os.getenv("BASE_URL", "http://localhost:8000"))

    # Security & Limits
    demo_access_token: Optional[str] = Field(default_factory=lambda: os.getenv("DEMO_ACCESS_TOKEN", None) or None)
    rate_limit_per_minute: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
    max_request_duration_s: float = Field(default_factory=lambda: float(os.getenv("MAX_REQUEST_DURATION_S", "300.0")))
    min_request_duration_s: float = Field(default_factory=lambda: float(os.getenv("MIN_REQUEST_DURATION_S", "5.0")))
    max_narration_file_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_NARRATION_FILE_SIZE_MB", "50")))
    ssrf_protection_enabled: bool = Field(default_factory=lambda: os.getenv("SSRF_PROTECTION_ENABLED", "true").lower() in ("true", "1", "yes"))

    # Music Generation Provider
    # "mock" or "stable_audio"
    music_provider: str = Field(default_factory=lambda: os.getenv("MUSIC_PROVIDER", "mock").lower())
    stable_audio_model_id: str = Field(default_factory=lambda: os.getenv("STABLE_AUDIO_MODEL_ID", "stabilityai/stable-audio-3-small-music"))
    hf_token: Optional[str] = Field(default_factory=lambda: os.getenv("HF_TOKEN", None) or None)
    torch_device: str = Field(default_factory=lambda: os.getenv("TORCH_DEVICE", "cuda"))
    model_timeout_s: float = Field(default_factory=lambda: float(os.getenv("MODEL_TIMEOUT_S", "45.0")))

    # Stem Separator
    # "mock" or "demucs"
    stem_separator: str = Field(default_factory=lambda: os.getenv("STEM_SEPARATOR", "mock").lower())
    demucs_model_name: str = Field(default_factory=lambda: os.getenv("DEMUCS_MODEL_NAME", "htdemucs"))
    demucs_device: str = Field(default_factory=lambda: os.getenv("DEMUCS_DEVICE", "cpu"))

    # Audio Processing Defaults
    target_lufs: float = Field(default_factory=lambda: float(os.getenv("TARGET_LUFS", "-16.0")))
    target_true_peak_dbtp: float = Field(default_factory=lambda: float(os.getenv("TARGET_TRUE_PEAK_DBTP", "-1.0")))
    default_sample_rate: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_SAMPLE_RATE", "48000")))
    duration_tolerance_s: float = Field(default_factory=lambda: float(os.getenv("DURATION_TOLERANCE_S", "0.05")))

    # Storage
    storage_type: str = Field(default_factory=lambda: os.getenv("STORAGE_TYPE", "local").lower())
    storage_local_dir: str = Field(default_factory=lambda: os.getenv("STORAGE_LOCAL_DIR", str(BASE_DIR / "storage_data")))
    s3_endpoint: str = Field(default_factory=lambda: os.getenv("S3_ENDPOINT", "localhost:9000"))
    s3_access_key: str = Field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", "minioadmin"))
    s3_secret_key: str = Field(default_factory=lambda: os.getenv("S3_SECRET_KEY", "minioadmin"))
    s3_bucket: str = Field(default_factory=lambda: os.getenv("S3_BUCKET", "incubrix-audio"))
    s3_secure: bool = Field(default_factory=lambda: os.getenv("S3_SECURE", "false").lower() in ("true", "1", "yes"))

    # Redis & Queue
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    use_redis_queue: bool = Field(default_factory=lambda: os.getenv("USE_REDIS_QUEUE", "false").lower() in ("true", "1", "yes"))
    queue_concurrency: int = Field(default_factory=lambda: int(os.getenv("QUEUE_CONCURRENCY", "2")))

    # Database
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'incubrix.db'}"))


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
