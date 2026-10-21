"""
Music Provider Abstract Base Class Interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict
from incubrix.schemas import CreateMusicJobRequest


class GenerationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: Any  # numpy.ndarray
    sample_rate: int
    duration_s: float
    model_name: str
    model_version: str
    seed_used: Optional[int] = None
    tempo_bpm: int
    key: str
    processing_time_s: float
    estimated_cost_usd: float
    warnings: list[str] = []
    section_measurements: list[Dict[str, Any]] = []


class BaseMusicProvider(ABC):
    @abstractmethod
    async def generate_music(self, request: CreateMusicJobRequest) -> GenerationResult:
        """Generate stereo music array for the requested prompt and specifications."""
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Return provider metadata, model name, version, and licensing terms."""
        pass

    @abstractmethod
    async def health_check(self) -> Tuple[bool, str]:
        """Verify provider availability, credentials, and dependency readiness."""
        pass
