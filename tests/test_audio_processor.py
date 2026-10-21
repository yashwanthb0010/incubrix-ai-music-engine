"""
Unit tests for Audio Processing, Loudness Normalization, Ducking, and Stem Reconstruction.
"""

import numpy as np
import pytest
from incubrix.audio.processor import AudioProcessor
from incubrix.audio.ducking import DuckingEngine
from incubrix.audio.validator import AudioValidator


def test_ensure_stereo_and_resampling():
    processor = AudioProcessor(target_sample_rate=48000)
    # Test mono to stereo
    mono = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32)
    stereo = processor.ensure_stereo(mono)
    assert stereo.ndim == 2
    assert stereo.shape == (44100, 2)

    # Test resampling from 44100 to 48000
    resampled = processor.resample(stereo, orig_sr=44100, target_sr=48000)
    assert resampled.shape[1] == 2
    expected_samples = int(round(44100 * (48000 / 44100)))
    assert abs(resampled.shape[0] - expected_samples) <= 2


def test_exact_duration_enforcement_and_musical_fade():
    processor = AudioProcessor(target_sample_rate=48000)
    sr = 48000
    # Create 10-second audio
    audio = np.random.randn(sr * 10, 2).astype(np.float32) * 0.1

    # Request 5.0 seconds
    exact_5s = processor.enforce_exact_duration(audio, sr, target_duration_s=5.0)
    assert exact_5s.shape[0] == sr * 5
    # The last sample should be near 0 due to cosine fade
    assert abs(exact_5s[-1, 0]) < 1e-4

    # Request 12.0 seconds (padding)
    exact_12s = processor.enforce_exact_duration(audio, sr, target_duration_s=12.0)
    assert exact_12s.shape[0] == sr * 12


def test_loudness_and_true_peak_normalization():
    processor = AudioProcessor(target_sample_rate=48000, target_lufs=-16.0, target_true_peak_dbtp=-1.0)
    sr = 48000
    t = np.linspace(0, 5, sr * 5)
    # Generate 1 kHz sine tone
    signal = np.sin(2 * np.pi * 1000 * t) * 0.5
    stereo = np.column_stack([signal, signal])

    norm_audio, measured_lufs, true_peak, _ = processor.normalize_loudness_and_peak(
        stereo, sr, target_lufs=-16.0, max_true_peak_dbtp=-1.0
    )

    # Should be within 1 LU of target -16 LUFS
    assert abs(measured_lufs - (-16.0)) <= 1.0
    # True peak should not exceed -1.0 dBTP
    assert true_peak <= -0.9


def test_loop_crossfade_boundary():
    processor = AudioProcessor(target_sample_rate=48000)
    sr = 48000
    audio = np.random.randn(sr * 4, 2).astype(np.float32) * 0.2
    looped, jump = processor.apply_loop_crossfade(audio, sr, crossfade_s=0.5)

    assert looped.shape == audio.shape
    # Boundary discontinuity should be virtually zero
    assert jump < 0.001


def test_ducking_with_speech_and_silence():
    engine = DuckingEngine(attenuation_db=10.0, attack_ms=50.0, release_ms=250.0)
    sr = 48000
    duration_s = 4.0
    t = np.linspace(0, duration_s, int(sr * duration_s))

    # Background music: constant amplitude tone
    music = np.column_stack([np.sin(2 * np.pi * 440 * t) * 0.5, np.sin(2 * np.pi * 440 * t) * 0.5])

    # Narration: speech between 1.0s and 2.5s, silence elsewhere
    narration = np.zeros(len(t), dtype=np.float32)
    speech_start = int(1.0 * sr)
    speech_end = int(2.5 * sr)
    narration[speech_start:speech_end] = np.random.randn(speech_end - speech_start) * 0.6

    ducked_music, manifest = engine.apply_ducking(music, narration, sr)

    assert manifest["applied"] is True
    assert manifest["speech_detected"] is True
    assert manifest["speech_duration_s"] > 1.0

    # During silence (e.g. at 0.5s), music should be full volume (gain ~ 1.0)
    rms_before = np.sqrt(np.mean(ducked_music[int(0.2*sr):int(0.6*sr)]**2))
    # During speech (e.g. at 1.8s), music should be ducked by ~10 dB (gain ~ 0.316)
    rms_during = np.sqrt(np.mean(ducked_music[int(1.5*sr):int(2.0*sr)]**2))

    assert rms_during < rms_before * 0.5


def test_ducking_with_no_speech():
    engine = DuckingEngine()
    sr = 48000
    music = np.ones((sr * 3, 2), dtype=np.float32) * 0.5
    silent_narration = np.zeros(sr * 3, dtype=np.float32)

    ducked, manifest = engine.apply_ducking(music, silent_narration, sr)
    assert manifest["speech_detected"] is False
    assert manifest["speech_duration_s"] == 0.0
    # Music should remain virtually unchanged
    np.testing.assert_allclose(ducked, music, atol=1e-3)


def test_stem_reconstruction_validation():
    validator = AudioValidator()
    sr = 48000
    samples = sr * 4

    drums = np.random.randn(samples, 2).astype(np.float32) * 0.3
    bass = np.random.randn(samples, 2).astype(np.float32) * 0.25
    other = np.random.randn(samples, 2).astype(np.float32) * 0.2

    mix = drums + bass + other

    stems = {
        "drums": (drums, 1.0),
        "bass": (bass, 1.0),
        "other": (other, 1.0),
    }

    recon = validator.measure_stem_reconstruction(mix, stems, sr)
    assert recon.stem_count == 3
    assert recon.reconstruction_snr_db > 80.0
    assert recon.reconstruction_rms_error < 1e-4
    assert recon.tolerances_met is True
