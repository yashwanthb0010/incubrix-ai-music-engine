"""
Demucs Pretrained Music Source-Separation Adapter for IncuBrix AI Background Music Engine.
Model: Meta AI Demucs v4 (Hybrid Transformer Demucs - 'htdemucs')
License: MIT License
Sources produced: drums, bass, other, vocals
"""

from __future__ import annotations
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from incubrix.stems.base import BaseStemSeparator, StemSeparationResult
from incubrix.errors import IncubrixError, ErrorCode


class DemucsStemSeparator(BaseStemSeparator):
    def __init__(self, model_name: str = "htdemucs", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._is_loaded = False

    def get_separator_info(self) -> Dict[str, Any]:
        return {
            "separator_name": "Meta AI Demucs v4",
            "model_name": self.model_name,
            "architecture": "Hybrid Transformer Demucs (HTDemucs)",
            "license": "MIT License",
            "commercial_use": "Permitted under MIT License for software and MusDB18 checkpoints",
            "sources_produced": ["drums", "bass", "other", "vocals"],
            "native_sample_rate": 44100,
            "is_mock": False,
        }

    async def health_check(self) -> Tuple[bool, str]:
        try:
            import torch
            import demucs
            return True, f"Demucs '{self.model_name}' available on device '{self.device}'."
        except ImportError as e:
            return False, f"Demucs dependencies not available: {str(e)}. Set STEM_SEPARATOR=mock for local mock testing."

    def _ensure_loaded(self):
        if self._is_loaded and self._model is not None:
            return

        try:
            import torch
            from demucs.pretrained import get_model
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
        except ImportError:
            raise IncubrixError(
                code=ErrorCode.STEM_SEPARATION_FAILED,
                message=(
                    "Demucs library is not installed in the current environment. "
                    "Install via: pip install demucs torch torchaudio. "
                    "Alternatively, set STEM_SEPARATOR=mock to run deterministic mock stems."
                ),
                status_code=503,
                retryable=False,
                details={"model": self.model_name},
            )
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STEM_SEPARATION_FAILED,
                message=f"Failed to load Demucs model '{self.model_name}': {str(e)}",
                status_code=500,
                retryable=True,
                details={"error": str(e)},
            )

    async def separate_stems(
        self,
        mix_audio: np.ndarray,
        sample_rate: int,
        provider_ref: Optional[Any] = None,
    ) -> StemSeparationResult:
        self._ensure_loaded()

        try:
            import torch
            from demucs.apply import apply_model

            # Demucs expects tensor of shape (batch, channels, samples)
            # mix_audio is (samples, 2)
            tensor_audio = torch.from_numpy(mix_audio.T).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                # sources shape: (batch, num_sources, channels, samples)
                sources = apply_model(self._model, tensor_audio, device=self.device)

            source_names = self._model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
            stems: Dict[str, Tuple[np.ndarray, float]] = {}

            sources_np = sources[0].cpu().numpy()  # shape: (num_sources, channels, samples)

            for idx, name in enumerate(source_names):
                stem_arr = sources_np[idx].T  # shape: (samples, channels)
                stems[name] = (stem_arr, 1.0)

            # In instrumental background music, vocals is typically empty or residual bleed
            # Only claim the actual genuine instrument stems
            limitations = (
                "Demucs separates into drums, bass, other, and vocals. "
                "For instrumental music, 'other' encompasses guitars, pianos, and lead synths. "
                "Vocals stem represents speech/vocal leakage or silence."
            )

            return StemSeparationResult(
                stems=stems,
                production_method=f"Demucs v4 HTDemucs ({self.model_name}) Pretrained Music Source Separation",
                is_mock=False,
                limitations=limitations,
            )
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.STEM_SEPARATION_FAILED,
                message=f"Demucs source separation execution failed: {str(e)}",
                status_code=500,
                retryable=True,
                details={"error": str(e)},
            )
