"""
FFmpeg utilities for audio format conversion.
Automatically locates ffmpeg (via imageio-ffmpeg or system PATH).
"""

from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def get_ffmpeg_path() -> Optional[str]:
    """Locate ffmpeg binary on the system or inside imageio_ffmpeg."""
    # 1. Check environment variable
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Check system PATH
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path

    # 3. Check imageio_ffmpeg bundled executable
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    return None


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "256k") -> bool:
    """Convert a WAV file to MP3 using ffmpeg."""
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        return False

    cmd = [
        ffmpeg_exe,
        "-y",               # Overwrite output
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(mp3_path),
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return mp3_path.exists()
    except Exception:
        return False
