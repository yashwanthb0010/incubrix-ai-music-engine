"""
Stem Separator Abstract Base Class Interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict


class StemSeparationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # stem_name -> (numpy.ndarray, linear_gain)
    stems: Dict[str, Tuple[Any, float]]
    production_method: str
    is_mock: bool
    limitations: Optional[str] = None


class BaseStemSeparator(ABC):
    @abstractmethod
    async def separate_stems(
        self,
        mix_audio: np.ndarray,
        sample_rate: int,
        provider_ref: Optional[Any] = None,
    ) -> StemSeparationResult:
        """Extract genuine distinct instrument stems from the music mix."""
        pass

    @abstractmethod
    def get_separator_info(self) -> Dict[str, Any]:
        """Return information on model checkpoint, dependencies, and license."""
        pass

    @abstractmethod
    async def health_check(self) -> Tuple[bool, str]:
        """Check availability and readiness of stem separation model."""
        pass
