"""
Stem Separator Factory for IncuBrix.
"""

from typing import Optional
from incubrix.config import Settings, get_settings
from incubrix.stems.base import BaseStemSeparator
from incubrix.stems.mock_separator import MockStemSeparator
from incubrix.stems.demucs_separator import DemucsStemSeparator


def get_stem_separator(settings: Optional[Settings] = None) -> BaseStemSeparator:
    cfg = settings or get_settings()
    if cfg.stem_separator == "demucs":
        return DemucsStemSeparator(
            model_name=cfg.demucs_model_name,
            device=cfg.demucs_device,
        )
    return MockStemSeparator()
