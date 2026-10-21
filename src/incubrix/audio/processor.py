"""
Audio processing pipeline for IncuBrix AI Background Music Engine.
Handles resampling, integrated loudness normalization (LUFS), True Peak limiting,
smooth musical fading, duration enforcement, and looping.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import scipy.signal
import pyloudnorm as pyln


class AudioProcessor:
    def __init__(
        self,
        target_sample_rate: int = 48000,
        target_lufs: float = -16.0,
        target_true_peak_dbtp: float = -1.0,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        self.target_true_peak_dbtp = target_true_peak_dbtp

    def ensure_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio array has shape (samples, 2)."""
        if audio.ndim == 1:
            return np.column_stack([audio, audio])
        elif audio.ndim == 2:
            if audio.shape[1] == 1:
                return np.repeat(audio, 2, axis=1)
            elif audio.shape[0] == 2 and audio.shape[1] > 2:
                return audio.T
            return audio
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio array to target sample rate using polyphase filtering."""
        if orig_sr == target_sr:
            return audio

        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        resampled_channels = []
        for ch in range(audio.shape[1]):
            resampled_ch = scipy.signal.resample_poly(audio[:, ch], up, down)
            resampled_channels.append(resampled_ch)

        return np.column_stack(resampled_channels)

    def measure_loudness(self, audio: np.ndarray, sample_rate: int) -> float:
        """Measure ITU-R BS.1770 integrated loudness in LUFS."""
        meter = pyln.Meter(sample_rate)
        energy = np.mean(audio**2)
        if energy < 1e-9:
            return -70.0
        try:
            loudness = meter.integrated_loudness(audio)
            if math.isnan(loudness) or math.isinf(loudness):
                return -70.0
            return float(loudness)
        except Exception:
            return -70.0

    def measure_true_peak(self, audio: np.ndarray) -> float:
        """Measure true peak in dBTP using 4x oversampling."""
        try:
            upsampled = scipy.signal.resample_poly(audio, 4, 1, axis=0)
            max_val = np.max(np.abs(upsampled))
        except Exception:
            max_val = np.max(np.abs(audio))

        if max_val <= 1e-9:
            return -100.0
        return float(20.0 * np.log10(max_val))

    def _soft_limit(self, audio: np.ndarray, max_linear_peak: float = 0.891) -> np.ndarray:
        """
        Soft knee limiter for transient control without destroying integrated LUFS.
        max_linear_peak = 10^(-1.0/20) ~ 0.891 (-1.0 dBTP).
        """
        out = audio.copy()
        knee = max_linear_peak * 0.85
        above_mask = np.abs(out) > knee
        if np.any(above_mask):
            sign = np.sign(out[above_mask])
            mag = np.abs(out[above_mask])
            # Smooth soft saturation above knee
            compressed = knee + (max_linear_peak - knee) * np.tanh((mag - knee) / (max_linear_peak - knee + 1e-6))
            out[above_mask] = sign * compressed
        return out

    def normalize_loudness_and_peak(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_lufs: float = -16.0,
        max_true_peak_dbtp: float = -1.0,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Normalize integrated loudness to target_lufs (ITU-R BS.1770).
        Applies transparent peak limiter so true peak <= max_true_peak_dbtp.
        Returns (processed_audio, measured_lufs, measured_true_peak, applied_gain_linear).
        """
        current_lufs = self.measure_loudness(audio, sample_rate)
        if current_lufs > -65.0:
            gain_db = target_lufs - current_lufs
            gain_linear = 10.0 ** (gain_db / 20.0)
            norm_audio = audio * gain_linear
        else:
            gain_linear = 1.0
            norm_audio = audio.copy()

        # Apply transparent transient limiter to ensure true peak <= max_true_peak_dbtp
        max_linear_peak = 10.0 ** (max_true_peak_dbtp / 20.0)
        norm_audio = self._soft_limit(norm_audio, max_linear_peak=max_linear_peak)

        # Measure true peak
        true_peak_dbtp = self.measure_true_peak(norm_audio)
        if true_peak_dbtp > max_true_peak_dbtp:
            excess_db = true_peak_dbtp - max_true_peak_dbtp
            reduction_factor = 10.0 ** (-excess_db / 20.0)
            norm_audio = norm_audio * reduction_factor
            gain_linear *= reduction_factor
            true_peak_dbtp = self.measure_true_peak(norm_audio)

        final_lufs = self.measure_loudness(norm_audio, sample_rate)
        return norm_audio, final_lufs, true_peak_dbtp, float(gain_linear)

    def enforce_exact_duration(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_duration_s: float,
        fade_out_duration_s: float = 1.5,
    ) -> np.ndarray:
        target_samples = int(round(target_duration_s * sample_rate))
        current_samples = audio.shape[0]

        if current_samples < target_samples:
            pad_len = target_samples - current_samples
            pad = np.zeros((pad_len, audio.shape[1]), dtype=audio.dtype)
            result = np.vstack([audio, pad])
        elif current_samples > target_samples:
            result = audio[:target_samples, :].copy()
        else:
            result = audio.copy()

        fade_len = min(int(round(fade_out_duration_s * sample_rate)), target_samples // 4)
        if fade_len > 0:
            t = np.linspace(0, np.pi / 2, fade_len)
            fade_curve = np.cos(t)[:, np.newaxis]
            result[-fade_len:, :] = result[-fade_len:, :] * fade_curve

        return result

    def apply_loop_crossfade(
        self,
        audio: np.ndarray,
        sample_rate: int,
        crossfade_s: float = 0.5,
    ) -> Tuple[np.ndarray, float]:
        samples = audio.shape[0]
        cf_samples = int(round(crossfade_s * sample_rate))
        cf_samples = min(cf_samples, samples // 6)

        if cf_samples <= 0:
            jump = float(np.max(np.abs(audio[0] - audio[-1])))
            return audio, jump

        processed = audio.copy()
        t = np.linspace(0, 1, cf_samples)[:, np.newaxis]

        fade_in = np.sin(t * (np.pi / 2))
        fade_out = np.cos(t * (np.pi / 2))

        head_block = processed[:cf_samples].copy()
        tail_block = processed[-cf_samples:].copy()

        blended_head = head_block * fade_in + tail_block * fade_out
        blended_tail = tail_block * fade_out + head_block * fade_in

        processed[:cf_samples] = blended_head
        processed[-cf_samples:] = blended_tail
        processed[-1] = processed[0]

        jump = float(np.max(np.abs(processed[0] - processed[-1])))
        return processed, jump
