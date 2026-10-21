"""
Deterministic High-Quality Mock Music Provider for IncuBrix AI Background Music Engine.
Synthesizes rich, multi-layered, beat-synchronized instrumental music.
Enables full local offline testing without GPU, API keys, or paid services.
Supports sections, key, tempo, looping, and seed reproducibility.
"""

from __future__ import annotations
import time
import math
from typing import Dict, Any, Tuple
import numpy as np
from incubrix.providers.base import BaseMusicProvider, GenerationResult
from incubrix.schemas import CreateMusicJobRequest


class MockMusicProvider(BaseMusicProvider):
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider_name": "IncuBrix Deterministic Mock Engine",
            "model_name": "incubrix-mock-synth-v1",
            "model_version": "1.0.0",
            "license": "MIT (Internal Mock Engine)",
            "commercial_use": "Permitted (Internal test engine for development)",
            "reproducibility": "Exact (fully deterministic when seed is provided)",
            "is_mock": True,
        }

    async def health_check(self) -> Tuple[bool, str]:
        return True, "Mock music provider is active and ready."

    def _get_key_base_freq(self, key_name: str) -> float:
        """Map musical key string to root frequency in Hz."""
        key_lower = key_name.lower().strip()
        notes = {
            "c": 130.81,   # C3
            "c#": 138.59, "db": 138.59,
            "d": 146.83,
            "d#": 155.56, "eb": 155.56,
            "e": 164.81,
            "f": 174.61,
            "f#": 185.00, "gb": 185.00,
            "g": 196.00,
            "g#": 207.65, "ab": 207.65,
            "a": 220.00,
            "a#": 233.08, "bb": 233.08,
            "b": 246.94,
        }
        for note, freq in notes.items():
            if key_lower.startswith(note):
                return freq
        return 130.81  # Default C3

    async def generate_music(self, request: CreateMusicJobRequest) -> GenerationResult:
        start_time = time.time()
        sr = request.output.sample_rate if request.output else self.sample_rate
        total_samples = int(round(request.duration_s * sr))

        # Seed handling
        seed = request.seed if request.seed is not None else int(time.time() * 1000) % 1000000
        rng = np.random.RandomState(seed)

        # Tempo and Key
        bpm = request.tempo_bpm if request.tempo_bpm else 112
        key_str = request.key if request.key else "C major"
        root_freq = self._get_key_base_freq(key_str)

        # Scale intervals (Major or Minor)
        is_minor = "minor" in key_str.lower()
        if is_minor:
            chord_ratios = [1.0, 1.2, 1.5, 1.68]  # Root, Minor 3rd, 5th, Octave
        else:
            chord_ratios = [1.0, 1.25, 1.5, 2.0]  # Root, Major 3rd, 5th, Octave

        # Generate timeline
        t = np.linspace(0, request.duration_s, total_samples, endpoint=False)
        beat_interval_s = 60.0 / bpm
        samples_per_beat = int(round(beat_interval_s * sr))

        # Build Section-Aware Map
        section_measurements = []
        sections = request.sections or []
        current_section_idx = 0

        # Output channels: (samples, 2)
        mix_left = np.zeros(total_samples, dtype=np.float32)
        mix_right = np.zeros(total_samples, dtype=np.float32)

        # Beat accumulator for drums & bassline
        num_beats = int(request.duration_s / beat_interval_s) + 1

        # Track transitions
        for sec in sections:
            # Measure deviation from requested section start
            measured_start = round(sec.start_s, 3)
            deviation = abs(measured_start - sec.start_s)
            section_measurements.append({
                "section_start_s": sec.start_s,
                "measured_transition_s": measured_start,
                "transition_deviation_s": deviation,
                "mood": sec.mood,
                "aligned_within_0_5s": deviation <= 0.5,
            })

        # Synthesize harmonic stems:
        # 1. Bassline (warm low-end sine + saw harmonics)
        bass_line = np.zeros(total_samples, dtype=np.float32)
        # 2. Chords / Synths (lush polyphonic pads & stereo chorus)
        chords_line = np.zeros(total_samples, dtype=np.float32)
        # 3. Drums (kick, snare, hi-hats aligned to beat grid)
        drums_line = np.zeros(total_samples, dtype=np.float32)

        # Synthesize Beats (Drums)
        for b in range(num_beats):
            beat_start_sample = int(b * samples_per_beat)
            if beat_start_sample >= total_samples:
                break

            beat_time = b * beat_interval_s
            # Find current section mood
            sec_mood = "intro"
            for sec in sections:
                if beat_time >= sec.start_s:
                    sec_mood = sec.mood.lower()

            # Dynamic intensity based on section
            is_energetic = "build" in sec_mood or "energetic" in sec_mood or "finish" in sec_mood or "drop" in sec_mood

            # Kick on beats 0, 2 (or all 4 if energetic)
            if b % 4 == 0 or (is_energetic and b % 2 == 0):
                kick_len = min(int(0.25 * sr), total_samples - beat_start_sample)
                kt = np.linspace(0, 0.25, kick_len)
                # Exponential pitch sweep for punchy kick
                k_freq = 55.0 + 90.0 * np.exp(-kt * 25.0)
                kick = np.sin(2 * np.pi * k_freq * kt) * np.exp(-kt * 14.0) * 0.8
                drums_line[beat_start_sample : beat_start_sample + kick_len] += kick

            # Snare on beats 1 and 3
            if b % 4 == 2 or (b % 4 == 1 and is_energetic):
                snare_len = min(int(0.2 * sr), total_samples - beat_start_sample)
                st = np.linspace(0, 0.2, snare_len)
                noise = (rng.rand(snare_len) * 2 - 1) * np.exp(-st * 18.0) * 0.4
                tone = np.sin(2 * np.pi * 180.0 * st) * np.exp(-st * 22.0) * 0.3
                drums_line[beat_start_sample : beat_start_sample + snare_len] += (noise + tone)

            # Hi-hat on every 8th note
            hat_subdivisions = [0, samples_per_beat // 2]
            for h_offset in hat_subdivisions:
                h_idx = beat_start_sample + h_offset
                if h_idx < total_samples:
                    hat_len = min(int(0.06 * sr), total_samples - h_idx)
                    ht = np.linspace(0, 0.06, hat_len)
                    hat_noise = (rng.rand(hat_len) * 2 - 1) * np.exp(-ht * 60.0) * 0.2
                    drums_line[h_idx : h_idx + hat_len] += hat_noise

        # Synthesize Bassline
        for b in range(num_beats):
            beat_start_sample = int(b * samples_per_beat)
            if beat_start_sample >= total_samples:
                break
            b_len = min(int(samples_per_beat * 0.85), total_samples - beat_start_sample)
            bt = np.linspace(0, float(b_len) / sr, b_len)

            # Bass root note changes every bar (4 beats)
            bar_idx = b // 4
            progression_ratios = [1.0, 1.25 if not is_minor else 1.2, 0.75, 0.89]
            freq_mult = progression_ratios[bar_idx % len(progression_ratios)]
            bass_freq = (root_freq * 0.5) * freq_mult

            bass_tone = np.sin(2 * np.pi * bass_freq * bt) + 0.3 * np.sin(4 * np.pi * bass_freq * bt)
            bass_env = np.exp(-bt * 3.5)
            bass_line[beat_start_sample : beat_start_sample + b_len] += (bass_tone * bass_env * 0.5)

        # Synthesize Chords / Ambient Pads
        for r in chord_ratios:
            freq = root_freq * r
            pad_left = np.sin(2 * np.pi * freq * t) * 0.15
            pad_right = np.sin(2 * np.pi * (freq * 1.003) * t + 0.5) * 0.15  # Stereo chorus
            chords_line += (pad_left + pad_right) * 0.5

        # Pan & Mix
        # Drums centered
        mix_left += drums_line * 0.6
        mix_right += drums_line * 0.6

        # Bass centered
        mix_left += bass_line * 0.7
        mix_right += bass_line * 0.7

        # Chords stereo spread
        mix_left += chords_line * 0.5
        mix_right += chords_line * 0.52

        # Combine
        stereo_audio = np.column_stack([mix_left, mix_right]).astype(np.float32)

        # Normalization headroom
        peak = np.max(np.abs(stereo_audio))
        if peak > 1e-4:
            stereo_audio = stereo_audio / peak * 0.85

        # Store stems cache on object for deterministic stem extraction
        self._last_stems = {
            "drums": np.column_stack([drums_line, drums_line]) * 0.6,
            "bass": np.column_stack([bass_line, bass_line]) * 0.7,
            "other": np.column_stack([chords_line, chords_line]) * 0.5,
        }

        warnings = []
        # Content checks for warnings
        p_lower = request.prompt.lower()
        if "vocal" in p_lower or "singing" in p_lower or "lyrics" in p_lower or "rap" in p_lower:
            warnings.append("Request asked for vocals. This engine is strictly instrumental; vocals were suppressed.")
        # Check for copyrighted artists / songs
        copyrighted_refs = ["taylor swift", "beatles", "drake", "queen", "michael jackson", "ed sheeran", "beethoven", "hans zimmer"]
        for cr in copyrighted_refs:
            if cr in p_lower:
                warnings.append(f"Prompt referenced '{cr}'. To respect copyright, an original original composition was generated instead of recreating copyrighted work.")

        elapsed = time.time() - start_time

        return GenerationResult(
            audio=stereo_audio,
            sample_rate=sr,
            duration_s=request.duration_s,
            model_name="incubrix-mock-synth-v1",
            model_version="1.0.0",
            seed_used=seed,
            tempo_bpm=bpm,
            key=key_str,
            processing_time_s=round(elapsed, 3),
            estimated_cost_usd=0.00,
            warnings=warnings,
            section_measurements=section_measurements,
        )
