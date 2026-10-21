"""
Narration Ducking Engine for IncuBrix AI Background Music Engine.
Detects speech activity, calculates a smooth dynamic attenuation envelope,
and applies ducking to the music track without including the narration audio.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import scipy.ndimage


class DuckingEngine:
    def __init__(
        self,
        attenuation_db: float = 10.0,
        attack_ms: float = 50.0,
        release_ms: float = 250.0,
        speech_threshold_ratio: float = 0.05,
    ):
        self.attenuation_db = attenuation_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.speech_threshold_ratio = speech_threshold_ratio

    def detect_speech_envelope(
        self,
        narration_audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Calculates normalized speech activity envelope [0.0 = silence, 1.0 = loud speech].
        Returns (envelope, speech_detected, speech_duration_s).
        """
        # Convert to mono if multichannel
        if narration_audio.ndim > 1:
            mono = np.mean(narration_audio, axis=1)
        else:
            mono = narration_audio

        # Instantaneous amplitude
        amplitude = np.abs(mono)
        max_amp = float(np.max(amplitude)) if len(amplitude) > 0 else 0.0

        if max_amp < 1e-4:
            envelope = np.zeros(len(mono), dtype=np.float32)
            return envelope, False, 0.0

        # Moving RMS energy window (50ms window)
        window_size = max(1, int(round(0.05 * sample_rate)))
        power = scipy.ndimage.uniform_filter1d(mono**2, size=window_size)
        rms = np.sqrt(np.maximum(power, 0.0))

        # Peak normalization
        rms_norm = rms / (np.max(rms) + 1e-9)

        # Threshold to detect speech
        threshold = max(0.02, self.speech_threshold_ratio)
        speech_mask = rms_norm > threshold

        # Speech duration in seconds
        speech_samples = int(np.count_nonzero(speech_mask))
        speech_duration_s = float(speech_samples / sample_rate)
        speech_detected = bool(speech_samples > int(0.2 * sample_rate))

        # Build smooth activity curve
        raw_activity = np.where(speech_mask, 1.0, 0.0).astype(np.float32)

        # Smooth activity with attack and release filter
        attack_samples = max(1, int(round((self.attack_ms / 1000.0) * sample_rate)))
        release_samples = max(1, int(round((self.release_ms / 1000.0) * sample_rate)))

        alpha_attack = np.exp(-1.0 / attack_samples)
        alpha_release = np.exp(-1.0 / release_samples)

        smooth_env = np.zeros_like(raw_activity)
        current = 0.0
        for i in range(len(raw_activity)):
            target = raw_activity[i]
            if target > current:
                current = target + alpha_attack * (current - target)
            else:
                current = target + alpha_release * (current - target)
            smooth_env[i] = current

        return smooth_env, speech_detected, speech_duration_s

    def apply_ducking(
        self,
        music_audio: np.ndarray,
        narration_audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies speech envelope ducking to music_audio.
        Music is attenuated by self.attenuation_db during speech.
        Returns (ducked_music, ducking_manifest_data).
        Note: The returned audio contains ONLY music, NOT the narration voice.
        """
        music_len = music_audio.shape[0]
        narr_len = narration_audio.shape[0]

        # Match narration length to music length for envelope calculation
        if narr_len < music_len:
            pad_len = music_len - narr_len
            if narration_audio.ndim > 1:
                padded_narr = np.pad(narration_audio, ((0, pad_len), (0, 0)))
            else:
                padded_narr = np.pad(narration_audio, (0, pad_len))
        else:
            padded_narr = narration_audio[:music_len]

        envelope, speech_detected, speech_duration_s = self.detect_speech_envelope(
            padded_narr, sample_rate
        )

        min_gain = 10.0 ** (-self.attenuation_db / 20.0)
        gain_curve = 1.0 - envelope * (1.0 - min_gain)

        if music_audio.ndim > 1:
            gain_curve_2d = gain_curve[:, np.newaxis]
            ducked_music = music_audio * gain_curve_2d
        else:
            ducked_music = music_audio * gain_curve

        num_summary_points = 50
        step = max(1, len(gain_curve) // num_summary_points)
        sampled_curve = [
            round(float(20.0 * np.log10(max(g, 1e-4))), 2)
            for g in gain_curve[::step][:num_summary_points]
        ]

        ducking_summary = {
            "applied": True,
            "attenuation_db": float(self.attenuation_db),
            "speech_detected": bool(speech_detected),
            "speech_duration_s": round(float(speech_duration_s), 2),
            "attack_ms": float(self.attack_ms),
            "release_ms": float(self.release_ms),
            "volume_curve_summary": {
                "points_count": len(sampled_curve),
                "gain_reduction_db_samples": sampled_curve,
                "max_reduction_db": round(float(20.0 * np.log10(np.min(gain_curve))), 2),
                "average_gain_db": round(float(20.0 * np.log10(np.mean(gain_curve))), 2),
            },
        }

        return ducked_music, ducking_summary
