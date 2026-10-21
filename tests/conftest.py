"""
Pytest configuration and fixtures for IncuBrix AI Background Music Engine.
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Force mock mode and temporary database for test suite
test_db_path = Path(__file__).resolve().parent / "test_incubrix.db"
os.environ["MUSIC_PROVIDER"] = "mock"
os.environ["STEM_SEPARATOR"] = "mock"
os.environ["STORAGE_TYPE"] = "local"
os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

from incubrix.api.app import app
from incubrix.config import get_settings


@pytest.fixture(scope="session")
def client():
    # Clean previous db file if exists
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception:
            pass

    with TestClient(app) as test_client:
        yield test_client

    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception:
            pass
