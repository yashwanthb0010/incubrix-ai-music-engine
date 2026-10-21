"""
Music provider factory.
"""

from typing import Optional
from incubrix.config import Settings, get_settings
from incubrix.providers.base import BaseMusicProvider
from incubrix.providers.mock_provider import MockMusicProvider
from incubrix.providers.stable_audio import StableAudioProvider


def get_music_provider(settings: Optional[Settings] = None) -> BaseMusicProvider:
    cfg = settings or get_settings()
    if cfg.music_provider == "stable_audio":
        return StableAudioProvider(
            model_id=cfg.stable_audio_model_id,
            hf_token=cfg.hf_token,
            device=cfg.torch_device,
        )
    return MockMusicProvider(sample_rate=cfg.default_sample_rate)
