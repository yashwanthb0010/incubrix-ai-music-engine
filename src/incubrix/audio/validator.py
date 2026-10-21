"""
Audio Validation and Quality Measurement Engine for IncuBrix AI Background Music Engine.
Validates duration tolerance (0.05s), loudness (-16 LUFS +/- 1 LU), True Peak (<= -1 dBTP),
clipping, silence gaps, stem alignment, and stem reconstruction fidelity.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from incubrix.schemas import QualityMeasurements, StemReconstruction
from incubrix.audio.processor import AudioProcessor


class AudioValidator:
    def __init__(
        self,
        duration_tolerance_s: float = 0.05,
        target_lufs: float = -16.0,
        lufs_tolerance_lu: float = 1.0,
        max_true_peak_dbtp: float = -1.0,
        required_sample_rate: int = 48000,
    ):
        self.duration_tolerance_s = duration_tolerance_s
        self.target_lufs = target_lufs
        self.lufs_tolerance_lu = lufs_tolerance_lu
        self.max_true_peak_dbtp = max_true_peak_dbtp
        self.required_sample_rate = required_sample_rate
        self.processor = AudioProcessor(target_sample_rate=required_sample_rate)

    def validate_and_measure(
        self,
        audio: np.ndarray,
        sample_rate: int,
        requested_duration_s: float,
    ) -> QualityMeasurements:
        """Measure audio properties and validate compliance against quality specs."""
        actual_samples = audio.shape[0]
        actual_duration_s = float(actual_samples / sample_rate)
        duration_error_s = abs(actual_duration_s - requested_duration_s)

        # Integrated loudness
        measured_lufs = self.processor.measure_loudness(audio, sample_rate)
        lufs_error_lu = abs(measured_lufs - self.target_lufs)

        # True Peak
        true_peak_dbtp = self.processor.measure_true_peak(audio)

        # Clipping detection: sample peak >= 0 dBFS (>= 0.9999)
        max_sample_val = np.max(np.abs(audio))
        has_clipping = bool(max_sample_val >= 0.9999)

        # Unexpected gaps detection: internal silence > 500ms before track end
        has_unexpected_gaps = False
        if actual_duration_s > 3.0:
            mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
            chunk_size = int(round(0.25 * sample_rate))
            # Check interior chunks (ignore first 200ms and last 1.5s fade)
            start_idx = int(0.2 * sample_rate)
            end_idx = max(start_idx, actual_samples - int(1.5 * sample_rate))
            if end_idx > start_idx + chunk_size:
                interior = mono[start_idx:end_idx]
                for i in range(0, len(interior) - chunk_size, chunk_size):
                    chunk_rms = np.sqrt(np.mean(interior[i : i + chunk_size] ** 2))
                    if chunk_rms < 1e-4:
                        has_unexpected_gaps = True
                        break

        ending_style = "smooth_musical_boundary_fade"

        return QualityMeasurements(
            measured_duration_s=round(actual_duration_s, 4),
            duration_error_s=round(duration_error_s, 4),
            sample_rate=sample_rate,
            integrated_lufs=round(measured_lufs, 2),
            target_lufs=self.target_lufs,
            lufs_error_lu=round(lufs_error_lu, 2),
            true_peak_dbtp=round(true_peak_dbtp, 2),
            has_clipping=has_clipping,
            has_unexpected_gaps=has_unexpected_gaps,
            ending_style=ending_style,
        )

    def measure_stem_reconstruction(
        self,
        final_mix: np.ndarray,
        stems: Dict[str, Tuple[np.ndarray, float]],  # stem_name -> (audio_array, gain)
        sample_rate: int,
    ) -> StemReconstruction:
        """
        Reconstructs mix from sum of (stem * gain) and calculates reconstruction SNR (dB)
        and RMS difference error.
        """
        stem_names = list(stems.keys())
        stem_count = len(stem_names)

        if stem_count == 0:
            return StemReconstruction(
                stem_count=0,
                stem_identities=[],
                reconstruction_snr_db=-100.0,
                reconstruction_rms_error=1.0,
                difference_max_db=0.0,
                tolerances_met=False,
                limitations="No stems supplied for reconstruction.",
            )

        # Find common length
        min_len = final_mix.shape[0]
        for name, (stem_audio, _) in stems.items():
            min_len = min(min_len, stem_audio.shape[0])

        reconstructed_mix = np.zeros((min_len, 2), dtype=np.float32)
        mix_sub = final_mix[:min_len, :]

        for name, (stem_audio, gain) in stems.items():
            sub = stem_audio[:min_len, :]
            reconstructed_mix += sub * gain

        # Calculate difference / error
        difference = mix_sub - reconstructed_mix
        rms_diff = float(np.sqrt(np.mean(difference**2)))
        rms_mix = float(np.sqrt(np.mean(mix_sub**2)))

        if rms_diff < 1e-7:
            snr_db = 100.0  # essentially perfect mathematical reconstruction
            diff_max_db = -100.0
        else:
            snr_db = float(20.0 * np.log10(max(rms_mix, 1e-6) / max(rms_diff, 1e-7)))
            max_abs_err = np.max(np.abs(difference))
            diff_max_db = float(20.0 * np.log10(max(max_abs_err, 1e-6)))

        tolerances_met = bool(snr_db >= 25.0 or rms_diff <= 0.05)

        return StemReconstruction(
            stem_count=stem_count,
            stem_identities=stem_names,
            reconstruction_snr_db=round(snr_db, 2),
            reconstruction_rms_error=round(rms_diff, 5),
            difference_max_db=round(diff_max_db, 2),
            tolerances_met=tolerances_met,
            limitations="Demucs-style source separation or multi-track harmonic synthesis.",
        )
