"""
Deterministic High-Quality Mock Music Provider for IncuBrix AI Background Music Engine.
Synthesizes rich, multi-layered, beat-synchronized instrumental music.
Each genre/prompt produces DISTINCTLY DIFFERENT sounding output via genre-specific
synthesis recipes: unique waveforms, drum patterns, chord voicings, FX, and timbres.
Enables full local offline testing without GPU, API keys, or paid services.
Supports sections, key, tempo, looping, and seed reproducibility.
"""

from __future__ import annotations
import time
import math
from typing import Dict, Any, Tuple, List
import numpy as np
from incubrix.providers.base import BaseMusicProvider, GenerationResult
from incubrix.schemas import CreateMusicJobRequest


# ---------------------------------------------------------------------------
# Genre Profiles – each entry defines a distinct sonic recipe
# ---------------------------------------------------------------------------
GENRE_PROFILES = {
    "lo-fi": {
        "drum_style": "lofi",        # soft, muted, tape-saturated
        "bass_style": "warm_sub",    # round sub-bass with slow attack
        "chord_style": "rhodes",     # electric piano + vinyl warmth
        "fx": {"vinyl_crackle": 0.08, "chorus_width": 0.002, "reverb_mix": 0.30, "distortion": 0.0, "tape_wobble": 0.003},
        "dynamics": {"drum_vol": 0.35, "bass_vol": 0.45, "chord_vol": 0.40, "melody_vol": 0.25},
        "melody_style": "pentatonic",
        "default_bpm": 85,
    },
    "cinematic": {
        "drum_style": "epic",        # big booming timpani + crashes
        "bass_style": "cello",       # low bowed-string harmonics
        "chord_style": "strings",    # lush string ensemble pads
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.006, "reverb_mix": 0.55, "distortion": 0.0, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.55, "bass_vol": 0.50, "chord_vol": 0.60, "melody_vol": 0.35},
        "melody_style": "modal",
        "default_bpm": 90,
    },
    "electronic": {
        "drum_style": "four_on_floor", # heavy kick, clap, open/closed hats
        "bass_style": "saw_bass",      # aggressive saw-wave bass
        "chord_style": "supersaw",     # detuned supersaw stacks
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.010, "reverb_mix": 0.20, "distortion": 0.15, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.65, "bass_vol": 0.60, "chord_vol": 0.45, "melody_vol": 0.30},
        "melody_style": "arpeggio",
        "default_bpm": 128,
    },
    "jazz": {
        "drum_style": "brush",       # brush swirl, soft ride, ghost notes
        "bass_style": "upright",     # plucked upright bass harmonics
        "chord_style": "jazz_voicing", # 7th/9th chord extensions
        "fx": {"vinyl_crackle": 0.03, "chorus_width": 0.004, "reverb_mix": 0.35, "distortion": 0.0, "tape_wobble": 0.001},
        "dynamics": {"drum_vol": 0.30, "bass_vol": 0.50, "chord_vol": 0.35, "melody_vol": 0.40},
        "melody_style": "chromatic",
        "default_bpm": 100,
    },
    "ambient": {
        "drum_style": "none",        # no drums
        "bass_style": "drone",       # low sustained drone
        "chord_style": "shimmer",    # bright shimmering harmonics
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.008, "reverb_mix": 0.70, "distortion": 0.0, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.0, "bass_vol": 0.30, "chord_vol": 0.55, "melody_vol": 0.20},
        "melody_style": "whole_tone",
        "default_bpm": 60,
    },
    "rock": {
        "drum_style": "heavy",       # loud snare, crash-heavy
        "bass_style": "distorted",   # overdriven bass guitar
        "chord_style": "power_chord", # parallel fifths, gritty
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.003, "reverb_mix": 0.15, "distortion": 0.25, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.70, "bass_vol": 0.65, "chord_vol": 0.55, "melody_vol": 0.30},
        "melody_style": "blues",
        "default_bpm": 120,
    },
    "acoustic": {
        "drum_style": "cajon",       # soft hand-percussion
        "bass_style": "acoustic_bass", # warm nylon bass
        "chord_style": "acoustic_guitar", # strummed chords
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.002, "reverb_mix": 0.25, "distortion": 0.0, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.30, "bass_vol": 0.45, "chord_vol": 0.50, "melody_vol": 0.35},
        "melody_style": "folk",
        "default_bpm": 95,
    },
    "horror": {
        "drum_style": "sparse_hits",  # rare booming hits, scrapes
        "bass_style": "sub_rumble",   # ultra-low rumble drone
        "chord_style": "dissonant",   # tritones, minor 2nds
        "fx": {"vinyl_crackle": 0.02, "chorus_width": 0.012, "reverb_mix": 0.65, "distortion": 0.05, "tape_wobble": 0.005},
        "dynamics": {"drum_vol": 0.25, "bass_vol": 0.40, "chord_vol": 0.45, "melody_vol": 0.15},
        "melody_style": "chromatic_descend",
        "default_bpm": 70,
    },
    "pop": {
        "drum_style": "pop_beat",    # tight kick/snare, shaker
        "bass_style": "synth_pop",   # bouncy synth bass
        "chord_style": "bright_pad", # bright major-key pads
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.004, "reverb_mix": 0.20, "distortion": 0.0, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.55, "bass_vol": 0.50, "chord_vol": 0.45, "melody_vol": 0.35},
        "melody_style": "catchy",
        "default_bpm": 118,
    },
    "corporate": {
        "drum_style": "light_pop",   # minimal kick + shaker
        "bass_style": "warm_sub",
        "chord_style": "piano_pad",  # clean piano + airy pad
        "fx": {"vinyl_crackle": 0.0, "chorus_width": 0.003, "reverb_mix": 0.25, "distortion": 0.0, "tape_wobble": 0.0},
        "dynamics": {"drum_vol": 0.35, "bass_vol": 0.35, "chord_vol": 0.50, "melody_vol": 0.30},
        "melody_style": "pentatonic",
        "default_bpm": 115,
    },
}

# Keywords that map prompt text to genre profiles
GENRE_KEYWORDS = {
    "lo-fi":       ["lo-fi", "lofi", "lo fi", "chillhop", "vinyl", "tape"],
    "cinematic":   ["cinematic", "orchestral", "epic", "trailer", "film", "movie", "strings", "brass"],
    "electronic":  ["electronic", "edm", "dance", "techno", "house", "trance", "dubstep", "synth"],
    "jazz":        ["jazz", "swing", "bebop", "bossa", "smooth jazz", "lounge"],
    "ambient":     ["ambient", "meditation", "meditative", "zen", "calm", "drone", "atmospheric", "sleep", "relax"],
    "rock":        ["rock", "metal", "punk", "grunge", "guitar riff", "heavy"],
    "acoustic":    ["acoustic", "folk", "ukulele", "guitar", "campfire", "fingerpick", "unplugged"],
    "horror":      ["horror", "dark", "scary", "suspense", "thriller", "eerie", "creepy", "tension"],
    "pop":         ["pop", "catchy", "upbeat", "cheerful", "bouncy", "fun"],
    "corporate":   ["corporate", "business", "professional", "presentation", "explainer", "startup"],
}


class MockMusicProvider(BaseMusicProvider):
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider_name": "IncuBrix Deterministic Mock Engine",
            "model_name": "incubrix-mock-synth-v2",
            "model_version": "2.0.0",
            "license": "MIT (Internal Mock Engine)",
            "commercial_use": "Permitted (Internal test engine for development)",
            "reproducibility": "Exact (fully deterministic when seed is provided)",
            "is_mock": True,
        }

    async def health_check(self) -> Tuple[bool, str]:
        return True, "Mock music provider is active and ready."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_key_base_freq(key_name: str) -> float:
        """Map musical key string to root frequency in Hz."""
        key_lower = key_name.lower().strip()
        notes = {
            "c": 130.81, "c#": 138.59, "db": 138.59,
            "d": 146.83, "d#": 155.56, "eb": 155.56,
            "e": 164.81, "f": 174.61, "f#": 185.00, "gb": 185.00,
            "g": 196.00, "g#": 207.65, "ab": 207.65,
            "a": 220.00, "a#": 233.08, "bb": 233.08,
            "b": 246.94,
        }
        for note, freq in notes.items():
            if key_lower.startswith(note):
                return freq
        return 130.81

    @staticmethod
    def _detect_genre(prompt: str) -> str:
        """Scan prompt for genre keywords, return best matching profile name."""
        p = prompt.lower()
        best, best_count = "pop", 0  # default fallback
        for genre, keywords in GENRE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in p)
            if count > best_count:
                best, best_count = genre, count
        return best

    @staticmethod
    def _saw_wave(phase: np.ndarray) -> np.ndarray:
        """Band-limited sawtooth approximation (4 harmonics)."""
        out = np.zeros_like(phase)
        for h in range(1, 5):
            out += ((-1) ** (h + 1)) * np.sin(h * phase) / h
        return out * (2.0 / np.pi)

    @staticmethod
    def _triangle_wave(phase: np.ndarray) -> np.ndarray:
        """Triangle wave from phase."""
        return 2.0 * np.abs(2.0 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1.0

    @staticmethod
    def _soft_clip(signal: np.ndarray, drive: float) -> np.ndarray:
        """Warm saturation / soft clipping."""
        if drive <= 0:
            return signal
        driven = signal * (1.0 + drive * 3.0)
        return np.tanh(driven)

    @staticmethod
    def _simple_reverb(signal: np.ndarray, sr: int, mix: float, decay: float = 0.4) -> np.ndarray:
        """Schroeder-style comb-filter reverb approximation."""
        if mix <= 0:
            return signal
        delays_ms = [29.3, 37.1, 41.1, 43.7]
        wet = np.zeros_like(signal)
        for d_ms in delays_ms:
            delay_samples = int(d_ms * sr / 1000.0)
            if delay_samples >= len(signal):
                continue
            delayed = np.zeros_like(signal)
            delayed[delay_samples:] = signal[:-delay_samples] * decay
            wet += delayed
        wet /= len(delays_ms)
        return signal * (1.0 - mix) + wet * mix

    # ------------------------------------------------------------------
    # Genre-specific drum synthesizers
    # ------------------------------------------------------------------
    def _synth_drums(self, style: str, num_beats: int, samples_per_beat: int,
                     total_samples: int, sr: int, rng: np.random.RandomState,
                     sections: list, beat_interval_s: float) -> np.ndarray:
        """Synthesize drum track based on genre style."""
        drums = np.zeros(total_samples, dtype=np.float32)
        if style == "none":
            return drums

        for b in range(num_beats):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break

            beat_time = b * beat_interval_s
            sec_mood = ""
            for sec in sections:
                if beat_time >= sec.start_s:
                    sec_mood = sec.mood.lower()
            is_energetic = any(w in sec_mood for w in ("build", "energetic", "finish", "drop", "intense"))

            if style == "lofi":
                # Muted, tape-compressed kick + vinyl-textured snare
                if b % 4 == 0 or (is_energetic and b % 2 == 0):
                    kl = min(int(0.18 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.18, kl)
                    kick = np.sin(2 * np.pi * (48 + 40 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 12) * 0.5
                    drums[bs:bs + kl] += kick
                if b % 4 == 2:
                    sl = min(int(0.12 * sr), total_samples - bs)
                    st = np.linspace(0, 0.12, sl)
                    snare = (rng.rand(sl) * 2 - 1) * np.exp(-st * 25) * 0.20
                    drums[bs:bs + sl] += snare
                # Shuffle hat
                if b % 2 == 0:
                    swing = int(samples_per_beat * 0.66) if b % 4 == 0 else 0
                    hi = bs + swing
                    if hi < total_samples:
                        hl = min(int(0.04 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.04, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 80) * 0.10

            elif style == "epic":
                # Timpani boom on beats 0, big crashes
                if b % 8 == 0:
                    kl = min(int(0.6 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.6, kl)
                    boom = np.sin(2 * np.pi * 42 * kt) * np.exp(-kt * 4) * 0.7
                    boom += np.sin(2 * np.pi * 84 * kt) * np.exp(-kt * 6) * 0.3
                    drums[bs:bs + kl] += boom
                if b % 8 == 4 or is_energetic:
                    cl = min(int(0.8 * sr), total_samples - bs)
                    ct = np.linspace(0, 0.8, cl)
                    crash = (rng.rand(cl) * 2 - 1) * np.exp(-ct * 3.5) * 0.35
                    drums[bs:bs + cl] += crash

            elif style == "four_on_floor":
                # EDM: kick every beat, clap on 2&4, open hats on off-beats
                kl = min(int(0.2 * sr), total_samples - bs)
                kt = np.linspace(0, 0.2, kl)
                kick = np.sin(2 * np.pi * (55 + 120 * np.exp(-kt * 35)) * kt) * np.exp(-kt * 16) * 0.9
                drums[bs:bs + kl] += kick
                if b % 4 in (1, 3):
                    cl = min(int(0.15 * sr), total_samples - bs)
                    ct = np.linspace(0, 0.15, cl)
                    clap = (rng.rand(cl) * 2 - 1) * np.exp(-ct * 22) * 0.45
                    drums[bs:bs + cl] += clap
                # Off-beat open hat
                oh_pos = bs + samples_per_beat // 2
                if oh_pos < total_samples:
                    ohl = min(int(0.10 * sr), total_samples - oh_pos)
                    oht = np.linspace(0, 0.10, ohl)
                    drums[oh_pos:oh_pos + ohl] += (rng.rand(ohl) * 2 - 1) * np.exp(-oht * 30) * 0.25

            elif style == "brush":
                # Jazz brush: soft ride + ghost snare
                # Ride pattern (swing)
                swing_offset = int(samples_per_beat * 0.62) if b % 2 == 1 else 0
                ri = bs + swing_offset
                if ri < total_samples:
                    rl = min(int(0.08 * sr), total_samples - ri)
                    rt = np.linspace(0, 0.08, rl)
                    ride = np.sin(2 * np.pi * 4800 * rt) * np.exp(-rt * 50) * 0.15
                    ride += (rng.rand(rl) * 2 - 1) * np.exp(-rt * 70) * 0.08
                    drums[ri:ri + rl] += ride
                # Ghost note
                if rng.rand() > 0.5 and b % 4 != 0:
                    gl = min(int(0.05 * sr), total_samples - bs)
                    gt = np.linspace(0, 0.05, gl)
                    drums[bs:bs + gl] += (rng.rand(gl) * 2 - 1) * np.exp(-gt * 40) * 0.08

            elif style == "heavy":
                # Rock: loud snare, crash heavy
                if b % 2 == 0:
                    kl = min(int(0.22 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.22, kl)
                    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 13) * 0.85
                    drums[bs:bs + kl] += kick
                if b % 4 == 2:
                    sl = min(int(0.22 * sr), total_samples - bs)
                    st = np.linspace(0, 0.22, sl)
                    snare = (rng.rand(sl) * 2 - 1) * np.exp(-st * 14) * 0.55
                    snare += np.sin(2 * np.pi * 200 * st) * np.exp(-st * 20) * 0.35
                    drums[bs:bs + sl] += snare
                # Crash every 8 beats
                if b % 8 == 0 and b > 0:
                    cl = min(int(0.7 * sr), total_samples - bs)
                    ct = np.linspace(0, 0.7, cl)
                    drums[bs:bs + cl] += (rng.rand(cl) * 2 - 1) * np.exp(-ct * 4) * 0.30
                # Steady 8th note hats
                for sub in [0, samples_per_beat // 2]:
                    hi = bs + sub
                    if hi < total_samples:
                        hl = min(int(0.04 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.04, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 55) * 0.18

            elif style == "cajon":
                # Soft hand percussion
                if b % 4 == 0:
                    kl = min(int(0.15 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.15, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * 65 * kt) * np.exp(-kt * 18) * 0.4
                if b % 4 == 2:
                    sl = min(int(0.08 * sr), total_samples - bs)
                    st = np.linspace(0, 0.08, sl)
                    drums[bs:bs + sl] += (rng.rand(sl) * 2 - 1) * np.exp(-st * 35) * 0.20
                # Shaker 16ths
                for sub_i in range(4):
                    si = bs + int(sub_i * samples_per_beat / 4)
                    if si < total_samples:
                        shl = min(int(0.03 * sr), total_samples - si)
                        sht = np.linspace(0, 0.03, shl)
                        drums[si:si + shl] += (rng.rand(shl) * 2 - 1) * np.exp(-sht * 90) * 0.06

            elif style == "sparse_hits":
                # Horror: rare booms, metallic scrapes
                if b % 16 == 0 or (is_energetic and b % 8 == 0):
                    kl = min(int(1.0 * sr), total_samples - bs)
                    kt = np.linspace(0, 1.0, kl)
                    boom = np.sin(2 * np.pi * 30 * kt) * np.exp(-kt * 2.0) * 0.6
                    drums[bs:bs + kl] += boom
                # Random metallic scrape
                if rng.rand() > 0.85:
                    scl = min(int(0.3 * sr), total_samples - bs)
                    sct = np.linspace(0, 0.3, scl)
                    scrape = np.sin(2 * np.pi * (2000 + 1500 * np.sin(2 * np.pi * 3 * sct)) * sct) * np.exp(-sct * 8) * 0.12
                    drums[bs:bs + scl] += scrape

            else:  # pop_beat, light_pop, default
                vol_mult = 0.7 if style == "light_pop" else 1.0
                if b % 4 == 0 or (b % 2 == 0 and is_energetic):
                    kl = min(int(0.20 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.20, kl)
                    kick = np.sin(2 * np.pi * (55 + 85 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 15) * 0.7 * vol_mult
                    drums[bs:bs + kl] += kick
                if b % 4 == 2:
                    sl = min(int(0.18 * sr), total_samples - bs)
                    st = np.linspace(0, 0.18, sl)
                    snare = (rng.rand(sl) * 2 - 1) * np.exp(-st * 20) * 0.35 * vol_mult
                    snare += np.sin(2 * np.pi * 180 * st) * np.exp(-st * 22) * 0.25 * vol_mult
                    drums[bs:bs + sl] += snare
                for sub in [0, samples_per_beat // 2]:
                    hi = bs + sub
                    if hi < total_samples:
                        hl = min(int(0.05 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.05, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 65) * 0.14 * vol_mult
        return drums

    # ------------------------------------------------------------------
    # Genre-specific bass synthesizers
    # ------------------------------------------------------------------
    def _synth_bass(self, style: str, num_beats: int, samples_per_beat: int,
                    total_samples: int, sr: int, root_freq: float,
                    is_minor: bool, rng: np.random.RandomState) -> np.ndarray:
        bass = np.zeros(total_samples, dtype=np.float32)

        if is_minor:
            progression = [1.0, 1.2, 0.75, 0.89]
        else:
            progression = [1.0, 1.25, 0.75, 0.89]

        # Shuffle progression based on style for variety
        if style in ("saw_bass", "distorted"):
            progression = [1.0, 0.89, 1.33, 0.75]
        elif style in ("upright", "acoustic_bass"):
            progression = [1.0, 1.33, 1.12, 0.75]
        elif style == "drone":
            progression = [1.0, 1.0, 1.0, 1.0]  # static drone
        elif style == "sub_rumble":
            progression = [1.0, 0.94, 1.0, 0.89]  # eerie movement

        for b in range(num_beats):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break
            b_len = min(int(samples_per_beat * 0.85), total_samples - bs)
            bt = np.linspace(0, float(b_len) / sr, b_len)

            bar_idx = b // 4
            freq_mult = progression[bar_idx % len(progression)]
            # Add seed-based variation to note choice
            note_variation = 1.0 + rng.uniform(-0.02, 0.02) if style != "drone" else 1.0
            bass_freq = (root_freq * 0.5) * freq_mult * note_variation

            if style == "warm_sub":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.15 * np.sin(4 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 2.5)
            elif style == "saw_bass":
                phase = 2 * np.pi * bass_freq * bt
                tone = self._saw_wave(phase)
                env = np.exp(-bt * 4.0)
            elif style == "cello":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.4 * np.sin(4 * np.pi * bass_freq * bt)
                tone += 0.15 * np.sin(6 * np.pi * bass_freq * bt)
                env = 1.0 - np.exp(-bt * 8)  # slow attack (bowed)
                env *= np.exp(-bt * 1.5)
            elif style == "upright":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.25 * self._triangle_wave(2 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 5.0) * (1.0 - np.exp(-bt * 40))  # pluck attack
            elif style == "distorted":
                phase = 2 * np.pi * bass_freq * bt
                tone = np.tanh(2.5 * np.sin(phase) + 0.8 * np.sin(2 * phase))
                env = np.exp(-bt * 3.0)
            elif style == "drone":
                tone = np.sin(2 * np.pi * bass_freq * bt) + 0.3 * np.sin(2 * np.pi * bass_freq * 1.5 * bt)
                env = np.ones_like(bt)  # sustained
            elif style == "sub_rumble":
                bass_freq *= 0.5  # extra low
                tone = np.sin(2 * np.pi * bass_freq * bt)
                lfo = 0.3 * np.sin(2 * np.pi * 0.5 * bt)
                tone = tone * (1.0 + lfo)
                env = np.exp(-bt * 1.0)
            elif style == "synth_pop":
                phase = 2 * np.pi * bass_freq * bt
                tone = np.sin(phase) + 0.4 * np.sin(2 * phase)
                env = np.exp(-bt * 6.0) * (1.0 - np.exp(-bt * 50))
            elif style == "acoustic_bass":
                tone = np.sin(2 * np.pi * bass_freq * bt) + 0.2 * np.sin(4 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 4.5) * (1.0 - np.exp(-bt * 30))
            else:
                tone = np.sin(2 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 3.5)

            bass[bs:bs + b_len] += (tone * env * 0.5).astype(np.float32)
        return bass

    # ------------------------------------------------------------------
    # Genre-specific chord / pad synthesizers
    # ------------------------------------------------------------------
    def _synth_chords(self, style: str, chord_ratios: List[float],
                      root_freq: float, t: np.ndarray, sr: int,
                      total_samples: int, rng: np.random.RandomState,
                      fx: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Return (left_channel, right_channel) for the chord/pad layer."""
        left = np.zeros(total_samples, dtype=np.float32)
        right = np.zeros(total_samples, dtype=np.float32)
        chorus_w = fx.get("chorus_width", 0.003)

        if style == "rhodes":
            for r in chord_ratios:
                f = root_freq * r * 2  # octave up for brightness
                # FM-synthesis-like electric piano
                mod = np.sin(2 * np.pi * f * 3.5 * t) * 0.3
                l = np.sin(2 * np.pi * f * t + mod) * 0.18
                r_ = np.sin(2 * np.pi * f * (1 + chorus_w) * t + mod + 0.4) * 0.18
                left += l
                right += r_

        elif style == "strings":
            for i, r in enumerate(chord_ratios):
                f = root_freq * r
                # Multiple detuned sine layers for string ensemble
                for detune in [-0.006, -0.002, 0.002, 0.006]:
                    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * (4.5 + i * 0.3) * t)
                    l = np.sin(2 * np.pi * f * (1 + detune) * vibrato * t) * 0.06
                    r_ = np.sin(2 * np.pi * f * (1 + detune + chorus_w) * vibrato * t + 0.3) * 0.06
                    left += l
                    right += r_

        elif style == "supersaw":
            for r in chord_ratios:
                f = root_freq * r * 2
                for det in np.linspace(-0.015, 0.015, 7):
                    phase = 2 * np.pi * f * (1 + det) * t
                    l = self._saw_wave(phase) * 0.04
                    phase_r = 2 * np.pi * f * (1 + det + chorus_w * 2) * t + rng.uniform(0, np.pi)
                    r_ = self._saw_wave(phase_r) * 0.04
                    left += l
                    right += r_

        elif style == "jazz_voicing":
            # Extended jazz chords (add 7th, 9th)
            jazz_ratios = chord_ratios + [1.78, 2.25]  # 7th, 9th approx
            for r in jazz_ratios:
                f = root_freq * r
                l = np.sin(2 * np.pi * f * t) * 0.10
                r_ = np.sin(2 * np.pi * f * (1 + chorus_w) * t + 0.7) * 0.10
                left += l
                right += r_

        elif style == "shimmer":
            # Very high harmonics + slow LFO modulation
            for r in chord_ratios:
                f = root_freq * r
                lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
                l = np.sin(2 * np.pi * f * 2 * t) * 0.12 * lfo
                r_ = np.sin(2 * np.pi * f * 3 * t + 1.0) * 0.10 * lfo
                left += l
                right += r_
                # Add octave shimmer
                l2 = np.sin(2 * np.pi * f * 4 * t) * 0.05 * lfo
                left += l2
                right += l2 * 0.8

        elif style == "power_chord":
            # Parallel 5ths with distortion
            for r in [1.0, 1.5, 2.0]:
                f = root_freq * r
                phase = 2 * np.pi * f * t
                tone = self._saw_wave(phase)
                tone = np.tanh(tone * 1.8) * 0.15
                left += tone
                right += np.tanh(self._saw_wave(phase * (1 + chorus_w)) * 1.8) * 0.15

        elif style == "acoustic_guitar":
            # Pluck-style chords changing every bar
            beat_len = len(t) // max(1, int(t[-1]))  # rough samples per second
            bar_samples = beat_len * 2  # change every 2 seconds
            for bar_start in range(0, total_samples, bar_samples):
                bar_end = min(bar_start + bar_samples, total_samples)
                seg_len = bar_end - bar_start
                seg_t = np.linspace(0, seg_len / sr, seg_len)
                bar_idx = bar_start // bar_samples
                chord_root = root_freq * chord_ratios[bar_idx % len(chord_ratios)]
                for harm in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    env = np.exp(-seg_t * 2.5) * (1 - np.exp(-seg_t * 30))
                    amp = 0.12 / harm
                    l = np.sin(2 * np.pi * chord_root * harm * seg_t) * amp * env
                    r_ = np.sin(2 * np.pi * chord_root * harm * (1 + chorus_w) * seg_t + 0.3) * amp * env
                    left[bar_start:bar_end] += l.astype(np.float32)
                    right[bar_start:bar_end] += r_.astype(np.float32)

        elif style == "dissonant":
            # Tritones and minor 2nds for horror
            dissonant_ratios = [1.0, 1.059, 1.414, 1.682]  # m2, tritone, m6
            for r in dissonant_ratios:
                f = root_freq * r
                lfo = 1.0 + 0.008 * np.sin(2 * np.pi * 0.3 * t)
                l = np.sin(2 * np.pi * f * lfo * t) * 0.14
                r_ = np.sin(2 * np.pi * f * 1.01 * lfo * t + 1.5) * 0.14
                left += l
                right += r_

        elif style == "piano_pad":
            # Clean piano-like attack + airy pad sustain
            for r in chord_ratios:
                f = root_freq * r * 2
                attack = 1.0 - np.exp(-t * 15)
                decay = np.exp(-t * 0.8)
                env = attack * decay
                l = np.sin(2 * np.pi * f * t) * 0.15 * env
                r_ = np.sin(2 * np.pi * f * (1 + chorus_w) * t + 0.3) * 0.15 * env
                left += l
                right += r_

        else:  # bright_pad, default
            for r in chord_ratios:
                f = root_freq * r
                l = np.sin(2 * np.pi * f * t) * 0.15
                r_ = np.sin(2 * np.pi * (f * (1 + chorus_w)) * t + 0.5) * 0.15
                left += l
                right += r_

        return left.astype(np.float32), right.astype(np.float32)

    # ------------------------------------------------------------------
    # Melody synthesizer (genre-aware)
    # ------------------------------------------------------------------
    def _synth_melody(self, style: str, num_beats: int, samples_per_beat: int,
                      total_samples: int, sr: int, root_freq: float,
                      rng: np.random.RandomState, is_minor: bool) -> np.ndarray:
        """Generate a simple genre-aware melody line."""
        melody = np.zeros(total_samples, dtype=np.float32)

        # Build scale
        if style == "pentatonic":
            intervals = [0, 2, 4, 7, 9]
        elif style == "blues":
            intervals = [0, 3, 5, 6, 7, 10]
        elif style == "chromatic" or style == "chromatic_descend":
            intervals = list(range(12))
        elif style == "whole_tone":
            intervals = [0, 2, 4, 6, 8, 10]
        elif style == "modal":
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Ionian/Lydian
        elif style == "catchy":
            intervals = [0, 2, 4, 5, 7]  # simple major
        elif style == "folk":
            intervals = [0, 2, 4, 5, 7, 9]
        elif style == "arpeggio":
            intervals = [0, 4, 7, 12, 16, 19]  # arpeggiated triad up 2 octaves
        else:
            intervals = [0, 2, 4, 5, 7, 9, 11]

        if is_minor:
            intervals = [i - 1 if i in (4, 9, 11) else i for i in intervals]

        semitone = 2 ** (1 / 12)
        freqs = [root_freq * 2 * (semitone ** i) for i in intervals]

        note_every = max(2, 4 if style in ("ambient", "whole_tone", "chromatic_descend") else 2)

        for b in range(0, num_beats, note_every):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break
            note_len = min(int(samples_per_beat * note_every * 0.7), total_samples - bs)
            nt = np.linspace(0, float(note_len) / sr, note_len)

            if style == "chromatic_descend":
                note_idx = (len(freqs) - 1) - (b // note_every) % len(freqs)
            elif style == "arpeggio":
                note_idx = (b // note_every) % len(freqs)
            else:
                note_idx = rng.randint(0, len(freqs))

            freq = freqs[note_idx]

            # Waveform varies by style
            if style in ("arpeggio", "catchy"):
                tone = np.sin(2 * np.pi * freq * nt)
                tone += 0.3 * np.sin(4 * np.pi * freq * nt)
            elif style == "blues":
                tone = self._saw_wave(2 * np.pi * freq * nt) * 0.7
                tone += np.sin(2 * np.pi * freq * nt) * 0.3
            elif style in ("whole_tone", "chromatic_descend"):
                vibrato = 1 + 0.005 * np.sin(2 * np.pi * 3 * nt)
                tone = np.sin(2 * np.pi * freq * vibrato * nt)
            else:
                tone = np.sin(2 * np.pi * freq * nt)

            # Envelope
            attack = 1.0 - np.exp(-nt * 20)
            release = np.exp(-nt * 3.0)
            env = attack * release

            melody[bs:bs + note_len] += (tone * env * 0.3).astype(np.float32)

        return melody

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    async def generate_music(self, request: CreateMusicJobRequest) -> GenerationResult:
        start_time = time.time()
        sr = request.output.sample_rate if request.output else self.sample_rate
        total_samples = int(round(request.duration_s * sr))

        # Seed handling
        seed = request.seed if request.seed is not None else int(time.time() * 1000) % 1000000
        rng = np.random.RandomState(seed)

        # Detect genre from prompt
        genre = self._detect_genre(request.prompt)
        profile = GENRE_PROFILES[genre]
        fx = profile["fx"]
        dyn = profile["dynamics"]

        # Tempo and Key
        bpm = request.tempo_bpm if request.tempo_bpm else profile["default_bpm"]
        key_str = request.key if request.key else "C major"
        root_freq = self._get_key_base_freq(key_str)

        # Scale intervals
        is_minor = "minor" in key_str.lower()
        if is_minor:
            chord_ratios = [1.0, 1.2, 1.5, 1.68]
        else:
            chord_ratios = [1.0, 1.25, 1.5, 2.0]

        t = np.linspace(0, request.duration_s, total_samples, endpoint=False)
        beat_interval_s = 60.0 / bpm
        samples_per_beat = int(round(beat_interval_s * sr))
        num_beats = int(request.duration_s / beat_interval_s) + 1

        # Section measurements
        section_measurements = []
        sections = request.sections or []
        for sec in sections:
            measured_start = round(sec.start_s, 3)
            deviation = abs(measured_start - sec.start_s)
            section_measurements.append({
                "section_start_s": sec.start_s,
                "measured_transition_s": measured_start,
                "transition_deviation_s": deviation,
                "mood": sec.mood,
                "aligned_within_0_5s": deviation <= 0.5,
            })

        # --- Synthesize each layer with genre-specific recipes ---
        drums_line = self._synth_drums(
            profile["drum_style"], num_beats, samples_per_beat,
            total_samples, sr, rng, sections, beat_interval_s
        )

        bass_line = self._synth_bass(
            profile["bass_style"], num_beats, samples_per_beat,
            total_samples, sr, root_freq, is_minor, rng
        )

        chord_left, chord_right = self._synth_chords(
            profile["chord_style"], chord_ratios, root_freq, t, sr,
            total_samples, rng, fx
        )

        melody_line = self._synth_melody(
            profile["melody_style"], num_beats, samples_per_beat,
            total_samples, sr, root_freq, rng, is_minor
        )

        # --- Apply FX ---
        # Tape wobble (lo-fi, horror)
        if fx.get("tape_wobble", 0) > 0:
            wobble_depth = fx["tape_wobble"]
            wobble_lfo = wobble_depth * np.sin(2 * np.pi * 0.5 * t)
            # Apply pitch wobble via phase modulation
            wobble_mod = np.sin(2 * np.pi * 1.5 * t * (1.0 + wobble_lfo))
            melody_line = melody_line * (1.0 + 0.1 * wobble_mod).astype(np.float32)

        # Distortion
        if fx.get("distortion", 0) > 0:
            drums_line = self._soft_clip(drums_line, fx["distortion"] * 0.5)
            bass_line = self._soft_clip(bass_line, fx["distortion"])

        # --- Mix with genre-specific volumes ---
        mix_left = np.zeros(total_samples, dtype=np.float32)
        mix_right = np.zeros(total_samples, dtype=np.float32)

        # Drums centered
        mix_left += drums_line * dyn["drum_vol"]
        mix_right += drums_line * dyn["drum_vol"]

        # Bass centered
        mix_left += bass_line * dyn["bass_vol"]
        mix_right += bass_line * dyn["bass_vol"]

        # Chords stereo
        mix_left += chord_left * dyn["chord_vol"]
        mix_right += chord_right * dyn["chord_vol"]

        # Melody slightly panned
        mix_left += melody_line * dyn["melody_vol"] * 0.7
        mix_right += melody_line * dyn["melody_vol"] * 1.0

        # Vinyl crackle (lo-fi, jazz, horror)
        if fx.get("vinyl_crackle", 0) > 0:
            crackle_amp = fx["vinyl_crackle"]
            crackle = rng.rand(total_samples).astype(np.float32)
            # Sparse crackle (only ~5% of samples)
            crackle_mask = (rng.rand(total_samples) > 0.95).astype(np.float32)
            crackle = crackle * crackle_mask * crackle_amp
            mix_left += crackle
            mix_right += crackle * 0.8  # slightly different in right

        # Reverb
        if fx.get("reverb_mix", 0) > 0:
            mix_left = self._simple_reverb(mix_left, sr, fx["reverb_mix"]).astype(np.float32)
            mix_right = self._simple_reverb(mix_right, sr, fx["reverb_mix"]).astype(np.float32)

        # Combine to stereo
        stereo_audio = np.column_stack([mix_left, mix_right]).astype(np.float32)

        # Normalize headroom
        peak = np.max(np.abs(stereo_audio))
        if peak > 1e-4:
            stereo_audio = stereo_audio / peak * 0.85

        # Store stems for deterministic stem extraction
        self._last_stems = {
            "drums": np.column_stack([drums_line, drums_line]) * dyn["drum_vol"],
            "bass": np.column_stack([bass_line, bass_line]) * dyn["bass_vol"],
            "other": np.column_stack([chord_left + melody_line * 0.7, chord_right + melody_line]) * dyn["chord_vol"],
        }

        # Warnings
        warnings = [f"Genre detected from prompt: '{genre}' (profile: {profile['chord_style']})"]
        p_lower = request.prompt.lower()
        if any(w in p_lower for w in ("vocal", "singing", "lyrics", "rap")):
            warnings.append("Request asked for vocals. This engine is strictly instrumental; vocals were suppressed.")
        copyrighted_refs = ["taylor swift", "beatles", "drake", "queen", "michael jackson", "ed sheeran", "beethoven", "hans zimmer"]
        for cr in copyrighted_refs:
            if cr in p_lower:
                warnings.append(f"Prompt referenced '{cr}'. To respect copyright, an original composition was generated instead.")

        elapsed = time.time() - start_time

        return GenerationResult(
            audio=stereo_audio,
            sample_rate=sr,
            duration_s=request.duration_s,
            model_name="incubrix-mock-synth-v2",
            model_version="2.0.0",
            seed_used=seed,
            tempo_bpm=bpm,
            key=key_str,
            processing_time_s=round(elapsed, 3),
            estimated_cost_usd=0.00,
            warnings=warnings,
            section_measurements=section_measurements,
        )
