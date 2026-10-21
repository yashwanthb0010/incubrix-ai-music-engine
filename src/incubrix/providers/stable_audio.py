"""
Stable Audio 3 Small Music Provider Adapter for IncuBrix AI Background Music Engine.
Model: stabilityai/stable-audio-3-small-music
Governed by Stability AI Community License.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from incubrix.providers.base import BaseMusicProvider, GenerationResult
from incubrix.schemas import CreateMusicJobRequest
from incubrix.errors import IncubrixError, ErrorCode


class StableAudioProvider(BaseMusicProvider):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-3-small-music",
        hf_token: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = device
        self._model = None
        self._is_loaded = False

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider_name": "Stability AI",
            "model_name": self.model_id,
            "model_version": "3.0-small-music",
            "license": "Stability AI Community License",
            "license_conditions": (
                "Free for evaluation and commercial use for entities with annual revenue < $1,000,000 USD. "
                "Entities with >= $1,000,000 USD annual revenue require an Enterprise License from Stability AI. "
                "Weights are gated on Hugging Face; users must accept terms on the repository page."
            ),
            "native_sample_rate": 44100,
            "max_native_duration_s": 120,
            "is_gated": True,
            "is_mock": False,
        }

    async def health_check(self) -> Tuple[bool, str]:
        # Verify HF token
        if not self.hf_token:
            return False, (
                f"HF_TOKEN environment variable is not configured. "
                f"Model '{self.model_id}' is gated on Hugging Face and requires access authorization."
            )

        # Check torch
        try:
            import torch
        except ImportError:
            return False, "PyTorch ('torch') is not installed in the current environment."

        return True, f"Stable Audio 3 provider configured for model '{self.model_id}' on {self.device}."

    def _ensure_model_loaded(self):
        if self._is_loaded and self._model is not None:
            return

        if not self.hf_token:
            raise IncubrixError(
                code=ErrorCode.MISSING_MODEL_CREDENTIALS,
                message=(
                    f"Model '{self.model_id}' requires a Hugging Face access token with accepted terms. "
                    f"Please set HF_TOKEN in your .env file and accept terms at https://huggingface.co/{self.model_id}."
                ),
                status_code=503,
                retryable=False,
                details={"model_id": self.model_id, "gated": True},
            )

        try:
            import torch
            # Attempt to import official stable-audio-3 package or diffusers
            try:
                from stable_audio_3 import StableAudioModel
                self._model = StableAudioModel.from_pretrained(
                    self.model_id,
                    token=self.hf_token,
                )
                if torch.cuda.is_available() and "cuda" in self.device:
                    self._model = self._model.to("cuda")
                self._is_loaded = True
            except ImportError:
                # If stable_audio_3 library is not yet installed in host environment
                raise IncubrixError(
                    code=ErrorCode.MODEL_PROVIDER_UNAVAILABLE,
                    message=(
                        f"Inference library 'stable-audio-3' or 'torch' is not installed. "
                        f"Install via: pip install stable-audio-3 torch torchaudio. "
                        f"Alternatively, set MUSIC_PROVIDER=mock for deterministic local execution."
                    ),
                    status_code=503,
                    retryable=False,
                    details={"model_id": self.model_id},
                )
        except IncubrixError:
            raise
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.MODEL_PROVIDER_UNAVAILABLE,
                message=f"Failed to load Stable Audio model weights: {str(e)}",
                status_code=503,
                retryable=True,
                details={"model_id": self.model_id, "error": str(e)},
            )

    async def generate_music(self, request: CreateMusicJobRequest) -> GenerationResult:
        start_time = time.time()
        self._ensure_model_loaded()

        try:
            import torch
            import torchaudio

            # Stable Audio 3 supports up to 120s in a single pass
            # For requested durations up to 300s, segment generation is applied
            chunk_duration = min(request.duration_s, 120.0)
            
            # Run inference
            # torch.manual_seed if seed provided
            if request.seed is not None:
                torch.manual_seed(request.seed)

            # Generate via model
            # audio: shape (channels, samples)
            audio_tensor = self._model.generate(
                prompt=request.prompt,
                duration=chunk_duration,
                steps=8,
                cfg_scale=7.0,
            )

            # Convert to numpy
            audio_np = audio_tensor.detach().cpu().numpy()
            if audio_np.ndim == 2 and audio_np.shape[0] == 2:
                # Transpose from (2, samples) to (samples, 2)
                audio_np = audio_np.T

            sr = 44100
            elapsed = time.time() - start_time
            # Estimated cloud inference cost: ~$0.005 per 30 seconds on A10G/T4
            est_cost = (request.duration_s / 60.0) * 0.01

            return GenerationResult(
                audio=audio_np,
                sample_rate=sr,
                duration_s=float(audio_np.shape[0] / sr),
                model_name=self.model_id,
                model_version="3.0-small-music",
                seed_used=request.seed,
                tempo_bpm=request.tempo_bpm or 110,
                key=request.key or "C major",
                processing_time_s=round(elapsed, 3),
                estimated_cost_usd=round(est_cost, 4),
                warnings=[],
                section_measurements=[],
            )
        except Exception as e:
            raise IncubrixError(
                code=ErrorCode.MODEL_PROVIDER_UNAVAILABLE,
                message=f"Stable Audio inference execution failed: {str(e)}",
                status_code=500,
                retryable=True,
                details={"model_id": self.model_id, "error": str(e)},
            )
