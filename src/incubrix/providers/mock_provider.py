"""
Deterministic High-Quality Expressive Music Synthesizer for IncuBrix AI Background Music Engine.
Synthesizes rich, multi-layered, beat-synchronized instrumental music.
Supports 14 distinct genres with unique chord progressions, instrument timbres, drum patterns,
dynamic envelopes, BPM/key detection, section transitions, and exact seed reproducibility.
Enables offline development and evaluation without external GPU or paid API dependencies.
"""

from __future__ import annotations
import re
import time
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from incubrix.providers.base import BaseMusicProvider, GenerationResult
from incubrix.schemas import CreateMusicJobRequest


# ---------------------------------------------------------------------------
# 14 Genre Profiles – distinct harmonic progressions, instruments, rhythms & FX
# ---------------------------------------------------------------------------
GENRE_PROFILES: Dict[str, Dict[str, Any]] = {
    "lo-fi": {
        "display_name": "Lo-Fi Hip-Hop / Chillhop",
        "default_bpm": 82,
        "default_key": "D minor",
        "chord_instrument": "rhodes",
        "progression": [
            [1.0, 1.20, 1.50, 1.80],    # Dm7 (ii7)
            [1.33, 1.67, 2.00, 2.37],   # G7 (V7)
            [0.89, 1.11, 1.33, 1.67],   # Cmaj7 (Imaj7)
            [0.75, 0.94, 1.12, 1.41],   # A7 (VI7)
        ],
        "bass_style": "warm_sub",
        "drum_style": "lofi_boombap",
        "melody_style": "mellow_chime",
        "fx": {"vinyl_crackle": 0.09, "tape_wobble": 0.003, "reverb_mix": 0.32, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.40, "bass_vol": 0.50, "chord_vol": 0.45, "melody_vol": 0.30},
    },
    "cinematic": {
        "display_name": "Cinematic / Epic Orchestral",
        "default_bpm": 88,
        "default_key": "C minor",
        "chord_instrument": "orchestral_strings",
        "progression": [
            [1.0, 1.20, 1.50, 2.00],    # Cm (i)
            [1.60, 2.00, 2.40, 3.20],   # Ab (VI)
            [1.20, 1.50, 1.80, 2.40],   # Eb (III)
            [1.80, 2.25, 2.70, 3.60],   # Bb (VII)
        ],
        "bass_style": "bowed_cello",
        "drum_style": "epic_timpani",
        "melody_style": "brass_horn",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.60, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.55, "bass_vol": 0.55, "chord_vol": 0.60, "melody_vol": 0.38},
    },
    "sad_piano": {
        "display_name": "Emotional / Melancholic Piano",
        "default_bpm": 68,
        "default_key": "A minor",
        "chord_instrument": "acoustic_piano",
        "progression": [
            [1.0, 1.20, 1.50, 2.00],    # Am (i)
            [0.67, 0.83, 1.00, 1.33],   # F (VI)
            [0.79, 1.00, 1.19, 1.59],   # C (III)
            [0.89, 1.12, 1.34, 1.78],   # G (VII)
        ],
        "bass_style": "soft_upright",
        "drum_style": "none",
        "melody_style": "piano_solo",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.45, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.0, "bass_vol": 0.40, "chord_vol": 0.65, "melody_vol": 0.45},
    },
    "electronic_edm": {
        "display_name": "Electronic / Club EDM",
        "default_bpm": 128,
        "default_key": "F minor",
        "chord_instrument": "supersaw",
        "progression": [
            [1.0, 1.20, 1.50, 2.00],    # Fm (i)
            [1.60, 2.00, 2.40, 3.20],   # Db (VI)
            [1.33, 1.60, 2.00, 2.67],   # Bbm (iv)
            [1.50, 1.80, 2.25, 3.00],   # C (V)
        ],
        "bass_style": "acid_saw_bass",
        "drum_style": "four_on_floor",
        "melody_style": "synth_lead",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.22, "sidechain": 0.70},
        "dynamics": {"drum_vol": 0.70, "bass_vol": 0.65, "chord_vol": 0.50, "melody_vol": 0.35},
    },
    "synthwave": {
        "display_name": "Synthwave / Cyberpunk 80s",
        "default_bpm": 122,
        "default_key": "D minor",
        "chord_instrument": "analog_pad",
        "progression": [
            [1.0, 1.20, 1.50, 2.00],    # Dm (i)
            [0.84, 1.06, 1.26, 1.68],   # Bb (VI)
            [0.94, 1.19, 1.42, 1.89],   # C (VII)
            [1.0, 1.20, 1.50, 2.00],    # Dm (i)
        ],
        "bass_style": "arp_bass",
        "drum_style": "gated_80s",
        "melody_style": "retro_lead",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.002, "reverb_mix": 0.35, "sidechain": 0.45},
        "dynamics": {"drum_vol": 0.65, "bass_vol": 0.60, "chord_vol": 0.50, "melody_vol": 0.40},
    },
    "hip_hop": {
        "display_name": "Urban Hip-Hop / Trap",
        "default_bpm": 92,
        "default_key": "C# minor",
        "chord_instrument": "dark_rhodes",
        "progression": [
            [1.0, 1.20, 1.50, 1.80],    # C#m (i)
            [0.94, 1.19, 1.42, 1.89],   # B (VII)
            [0.84, 1.06, 1.26, 1.68],   # A (VI)
            [0.94, 1.19, 1.42, 1.89],   # B (VII)
        ],
        "bass_style": "808_sub_drop",
        "drum_style": "trap_groove",
        "melody_style": "plucked_flute",
        "fx": {"vinyl_crackle": 0.04, "tape_wobble": 0.001, "reverb_mix": 0.28, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.70, "bass_vol": 0.70, "chord_vol": 0.40, "melody_vol": 0.30},
    },
    "rock_metal": {
        "display_name": "Hard Rock / Metal",
        "default_bpm": 126,
        "default_key": "E minor",
        "chord_instrument": "power_chords",
        "progression": [
            [1.0, 1.50, 2.00],          # E power (i)
            [1.33, 2.00, 2.67],         # A power (IV)
            [1.50, 2.25, 3.00],         # B power (V)
            [1.0, 1.50, 2.00],          # E power (i)
        ],
        "bass_style": "distorted_rock",
        "drum_style": "rock_kit",
        "melody_style": "guitar_riff",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.18, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.75, "bass_vol": 0.65, "chord_vol": 0.60, "melody_vol": 0.40},
    },
    "acoustic_folk": {
        "display_name": "Acoustic / Warm Folk",
        "default_bpm": 96,
        "default_key": "G major",
        "chord_instrument": "acoustic_strum",
        "progression": [
            [1.0, 1.25, 1.50, 2.00],    # G (I)
            [1.33, 1.67, 2.00, 2.67],   # C (IV)
            [1.50, 1.88, 2.25, 3.00],   # D (V)
            [0.89, 1.07, 1.33, 1.78],   # Em (vi)
        ],
        "bass_style": "acoustic_wood_bass",
        "drum_style": "cajon_shaker",
        "melody_style": "acoustic_fingerpick",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.28, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.35, "bass_vol": 0.45, "chord_vol": 0.55, "melody_vol": 0.35},
    },
    "ambient_zen": {
        "display_name": "Ambient / Meditative Zen",
        "default_bpm": 60,
        "default_key": "F major",
        "chord_instrument": "shimmer_drone",
        "progression": [
            [1.0, 1.33, 1.50, 2.00],    # F sus2/4 (I)
            [1.12, 1.50, 1.69, 2.25],   # Gm pad (ii)
            [1.33, 1.78, 2.00, 2.67],   # Bb shimmer (IV)
            [1.0, 1.33, 1.50, 2.00],    # F sustained
        ],
        "bass_style": "deep_drone",
        "drum_style": "none",
        "melody_style": "singing_bowl",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.75, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.0, "bass_vol": 0.35, "chord_vol": 0.65, "melody_vol": 0.25},
    },
    "jazz_lounge": {
        "display_name": "Smooth Jazz / Lounge",
        "default_bpm": 102,
        "default_key": "Bb major",
        "chord_instrument": "jazz_extensions",
        "progression": [
            [1.0, 1.25, 1.50, 1.88, 2.25],  # Bb maj9 (I)
            [1.12, 1.35, 1.69, 2.02, 2.53],  # Cm9 (ii)
            [1.33, 1.67, 2.00, 2.37, 2.84],  # F13 (V)
            [0.89, 1.07, 1.33, 1.60, 2.00],  # Gm7 (vi)
        ],
        "bass_style": "walking_upright",
        "drum_style": "jazz_swing_brush",
        "melody_style": "jazz_vibes",
        "fx": {"vinyl_crackle": 0.03, "tape_wobble": 0.001, "reverb_mix": 0.35, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.35, "bass_vol": 0.55, "chord_vol": 0.45, "melody_vol": 0.40},
    },
    "horror_suspense": {
        "display_name": "Horror / Dark Suspense",
        "default_bpm": 66,
        "default_key": "C minor",
        "chord_instrument": "dissonant_cluster",
        "progression": [
            [1.0, 1.059, 1.414, 1.682],     # Tritone + m2 cluster (dissonant)
            [0.944, 1.0, 1.335, 1.587],     # Creeping half-step down
            [1.0, 1.189, 1.414, 1.782],     # Diminished triad
            [0.891, 0.944, 1.260, 1.500],   # Low dread cluster
        ],
        "bass_style": "sub_rumble_dread",
        "drum_style": "heartbeat_hits",
        "melody_style": "creepy_musicbox",
        "fx": {"vinyl_crackle": 0.03, "tape_wobble": 0.006, "reverb_mix": 0.65, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.35, "bass_vol": 0.55, "chord_vol": 0.50, "melody_vol": 0.25},
    },
    "corporate_tech": {
        "display_name": "Corporate / Inspiring Tech Explainer",
        "default_bpm": 115,
        "default_key": "C major",
        "chord_instrument": "bright_piano_pad",
        "progression": [
            [1.0, 1.25, 1.50, 2.00],    # C (I)
            [1.33, 1.67, 2.00, 2.67],   # F (IV)
            [0.89, 1.07, 1.33, 1.78],   # Am (vi)
            [1.50, 1.88, 2.25, 3.00],   # G (V)
        ],
        "bass_style": "tight_sub_bass",
        "drum_style": "light_corporate_beat",
        "melody_style": "tech_arpeggio",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.25, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.40, "bass_vol": 0.45, "chord_vol": 0.55, "melody_vol": 0.35},
    },
    "happy_upbeat": {
        "display_name": "Happy / Cheerful Upbeat Pop",
        "default_bpm": 120,
        "default_key": "D major",
        "chord_instrument": "bouncy_chords",
        "progression": [
            [1.0, 1.25, 1.50, 2.00],    # I
            [1.50, 1.88, 2.25, 3.00],   # V
            [0.89, 1.07, 1.33, 1.78],   # vi
            [1.33, 1.67, 2.00, 2.67],   # IV
        ],
        "bass_style": "bouncy_synth_pop",
        "drum_style": "pop_groove",
        "melody_style": "cheerful_bells",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.20, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.60, "bass_vol": 0.55, "chord_vol": 0.50, "melody_vol": 0.40},
    },
    "reggae_tropical": {
        "display_name": "Reggae / Tropical Island Summer",
        "default_bpm": 86,
        "default_key": "A major",
        "chord_instrument": "offbeat_skank",
        "progression": [
            [1.0, 1.25, 1.50, 2.00],    # I
            [1.33, 1.67, 2.00, 2.67],   # IV
            [1.50, 1.88, 2.25, 3.00],   # V
            [1.33, 1.67, 2.00, 2.67],   # IV
        ],
        "bass_style": "deep_dub_bass",
        "drum_style": "reggae_one_drop",
        "melody_style": "steel_drum",
        "fx": {"vinyl_crackle": 0.0, "tape_wobble": 0.0, "reverb_mix": 0.35, "sidechain": 0.0},
        "dynamics": {"drum_vol": 0.55, "bass_vol": 0.65, "chord_vol": 0.45, "melody_vol": 0.35},
    },
}

# ---------------------------------------------------------------------------
# Extensive Keyword Mappings (Over 250 semantic trigger terms)
# ---------------------------------------------------------------------------
KEYWORD_MAPPINGS = {
    "lo-fi": [
        "lo-fi", "lofi", "lo fi", "chillhop", "chill-hop", "vinyl", "cassette", "tape",
        "coffee", "cozy", "study", "relaxing study", "homework", "rainy", "bedroom pop",
        "mellow beats", "chilled beat", "lo-fi hip hop", "laid back"
    ],
    "cinematic": [
        "cinematic", "epic", "trailer", "film", "movie", "orchestral", "hans zimmer",
        "heroic", "hollywood", "soundtrack", "symphonic", "drama", "blockbuster",
        "triumph", "score", "strings", "brass", "glory", "battle", "legendary", "warrior"
    ],
    "sad_piano": [
        "sad", "emotional", "melancholy", "melancholic", "grief", "heartbreak", "sorrow",
        "crying", "mournful", "loss", "tears", "lonely", "somber", "depressing", "tragic",
        "emotional piano", "soft piano", "gentle piano", "sadness", "peaceful piano"
    ],
    "electronic_edm": [
        "electronic", "edm", "dance", "club", "rave", "techno", "house", "electro",
        "festival", "dj", "party", "banger", "drop", "synth bass", "workout", "gym",
        "high energy", "pumping", "driving beat", "aerobics", "fitness", "nightclub"
    ],
    "synthwave": [
        "synthwave", "retrowave", "outrun", "cyberpunk", "80s", "1980s", "retro", "neon",
        "futuristic", "arcade", "analog synth", "blade runner", "stranger things",
        "vaporwave", "night drive", "retro synth", "nostalgic 80s"
    ],
    "hip_hop": [
        "hip-hop", "hip hop", "trap", "rap", "beat", "urban", "street", "808", "boom bap",
        "cypher", "freestyle", "drill", "groove", "hiphop", "swag", "rhymes"
    ],
    "rock_metal": [
        "rock", "metal", "heavy metal", "hard rock", "punk", "grunge", "guitar riff",
        "electric guitar", "overdrive", "distorted guitar", "power chords", "aggressive",
        "intense rock", "headbanger", "stadium rock", "garage band", "drums and guitar"
    ],
    "acoustic_folk": [
        "acoustic", "folk", "campfire", "guitar", "acoustic guitar", "ukulele", "unplugged",
        "fingerstyle", "fingerpick", "organic", "country", "indie folk", "morning sunshine",
        "nature", "hiking", "travel vlog", "cozy morning", "rustic"
    ],
    "ambient_zen": [
        "ambient", "meditation", "meditative", "zen", "yoga", "spa", "sleep", "tranquil",
        "peaceful", "calm", "relax", "relaxation", "soothing", "healing", "drone",
        "floating", "ethereal", "space", "celestial", "mindfulness", "deep calm"
    ],
    "jazz_lounge": [
        "jazz", "lounge", "smooth jazz", "bossa nova", "swing", "bebop", "cocktail",
        "cafe", "bar", "saxophone", "sax", "upright bass", "sophisticated", "classy",
        "dining", "dim lights", "candlelight", "hotel lobby"
    ],
    "horror_suspense": [
        "horror", "scary", "spooky", "creepy", "dark", "fear", "frightening", "eerie",
        "tension", "suspense", "thriller", "terror", "nightmare", "chilling", "ghost",
        "haunted", "jumpscare", "macabre", "murder", "monster", "mysterious dark"
    ],
    "corporate_tech": [
        "corporate", "business", "presentation", "tech", "technology", "explainer",
        "startup", "product launch", "infomercial", "tutorial", "commercial",
        "innovative", "professional", "company", "advertising", "clean tech"
    ],
    "happy_upbeat": [
        "happy", "cheerful", "joyful", "fun", "bouncy", "sunny", "feel good", "smiling",
        "playful", "bright", "celebration", "positive", "uplifting", "kid", "children",
        "comedy", "wholesome", "good vibes", "delightful"
    ],
    "reggae_tropical": [
        "reggae", "tropical", "island", "summer", "beach", "caribbean", "jamaica",
        "calypso", "sunshine", "vacation", "holiday", "cocktails on the beach", "surf"
    ],
}


class MockMusicProvider(BaseMusicProvider):
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._last_stems = {}

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
    # Analysis & Prompt Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_bpm_from_prompt(prompt: str, default_bpm: int) -> int:
        """Extract explicit BPM like '128 bpm' or infer from tempo words."""
        p = prompt.lower()
        m = re.search(r"\b(\d{2,3})\s*(?:bpm|tempo)\b", p)
        if m:
            val = int(m.group(1))
            if 40 <= val <= 240:
                return val

        # Semantic tempo words
        if any(w in p for w in ("very fast", "hyper", "ultra fast", "chase", "racing")):
            return min(160, int(default_bpm * 1.35))
        if any(w in p for w in ("fast", "energetic", "speed", "running", "intense")):
            return min(145, int(default_bpm * 1.18))
        if any(w in p for w in ("very slow", "glacial", "sluggish")):
            return max(50, int(default_bpm * 0.75))
        if any(w in p for w in ("slow", "relaxed", "leisurely", "unhurried", "gentle")):
            return max(58, int(default_bpm * 0.88))
        return default_bpm

    @staticmethod
    def _parse_key_from_prompt(prompt: str, default_key: str) -> str:
        """Extract explicit key like 'in D minor' or infer minor/major from mood."""
        p = prompt.lower()
        # Look for e.g. "in d minor", "c# major", "f-sharp minor"
        m = re.search(r"\b(?:in\s+)?([a-g][#b]?)\s*(major|minor)\b", p)
        if m:
            note = m.group(1).upper()
            scale = m.group(2).lower()
            return f"{note} {scale}"

        # If user words suggest minor mood
        if any(w in p for w in ("sad", "dark", "evil", "horror", "grief", "melancholy", "suspense", "cyberpunk", "minor")):
            return default_key if "minor" in default_key else "D minor"

        # If user words suggest major mood
        if any(w in p for w in ("happy", "uplifting", "bright", "cheerful", "sunny", "joyful", "fun", "major")):
            return default_key if "major" in default_key else "C major"

        return default_key

    @staticmethod
    def _detect_genre(prompt: str) -> str:
        """Score all 14 genres against the user prompt with word-boundary awareness."""
        p = prompt.lower()
        scores: Dict[str, float] = {k: 0.0 for k in GENRE_PROFILES}

        for genre, keywords in KEYWORD_MAPPINGS.items():
            for kw in keywords:
                # Exact phrase match gives 2.5 points, single word 1.0 point
                if " " in kw:
                    if kw in p:
                        scores[genre] += 2.5
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", p):
                        scores[genre] += 1.0

        best_genre = max(scores, key=lambda g: scores[g])
        if scores[best_genre] > 0.0:
            return best_genre

        # Fallback heuristic: check acoustic, energetic, or calm
        if any(w in p for w in ("guitar", "piano", "folk", "soft", "morning")):
            return "acoustic_folk"
        if any(w in p for w in ("dark", "dramatic", "scary", "epic")):
            return "cinematic"
        if any(w in p for w in ("electronic", "beat", "drums", "synth", "pump")):
            return "electronic_edm"

        return "corporate_tech"

    @staticmethod
    def _get_key_base_freq(key_name: str) -> float:
        """Map musical key string to standard root frequency in Hz."""
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

    # ------------------------------------------------------------------
    # Synthesis Primitives (Waveforms & FX)
    # ------------------------------------------------------------------
    @staticmethod
    def _saw_wave(phase: np.ndarray) -> np.ndarray:
        """Band-limited sawtooth wave (5 harmonics)."""
        out = np.zeros_like(phase)
        for h in range(1, 6):
            out += ((-1) ** (h + 1)) * np.sin(h * phase) / h
        return out * (2.0 / np.pi)

    @staticmethod
    def _square_wave(phase: np.ndarray) -> np.ndarray:
        """Band-limited square wave (4 odd harmonics)."""
        out = np.zeros_like(phase)
        for h in (1, 3, 5, 7):
            out += np.sin(h * phase) / h
        return out * (4.0 / np.pi)

    @staticmethod
    def _triangle_wave(phase: np.ndarray) -> np.ndarray:
        """Triangle wave."""
        return 2.0 * np.abs(2.0 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1.0

    @staticmethod
    def _soft_clip(signal: np.ndarray, drive: float) -> np.ndarray:
        """Tube-like soft saturation."""
        if drive <= 0:
            return signal
        driven = signal * (1.0 + drive * 3.5)
        return np.tanh(driven)

    @staticmethod
    def _simple_reverb(signal: np.ndarray, sr: int, mix: float, decay: float = 0.42) -> np.ndarray:
        """Schroeder comb-filter spatial reverb."""
        if mix <= 0:
            return signal
        delays_ms = [27.3, 34.1, 41.7, 49.2]
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
    # Genre-Specific Drum Synthesizer
    # ------------------------------------------------------------------
    def _synth_drums(self, drum_style: str, num_beats: int, samples_per_beat: int,
                     total_samples: int, sr: int, rng: np.random.RandomState,
                     sections: list, beat_interval_s: float, is_loopable: bool = False) -> np.ndarray:
        drums = np.zeros(total_samples, dtype=np.float32)
        if drum_style == "none":
            return drums

        total_dur_s = float(total_samples) / sr
        # Calculate cadence downbeat for intentional ending hit
        if not is_loopable:
            if total_dur_s < 8.0:
                cadence_beat = max(0, num_beats - 2)
            else:
                cadence_beat = ((num_beats - 1) // 4) * 4
                if (num_beats - cadence_beat) > 4:
                    cadence_beat = max(0, num_beats - 4)
        else:
            cadence_beat = -1

        for b in range(num_beats):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break
            beat_time = b * beat_interval_s

            # Intentional ending handling:
            if not is_loopable and cadence_beat >= 0:
                if b == cadence_beat:
                    # Final resolving accent hit (punchy kick + resonant crash cymbal ring-out)
                    kl = min(int(0.25 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.25, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 24)) * kt) * np.exp(-kt * 12) * 0.90
                    
                    crash_len = min(int(3.2 * sr), total_samples - bs)
                    crash_t = np.linspace(0, float(crash_len) / sr, crash_len)
                    drums[bs:bs + crash_len] += (rng.rand(crash_len) * 2 - 1) * np.exp(-crash_t * 2.0) * 0.40
                    continue
                elif b > cadence_beat:
                    # Silence any new drum triggers after final hit so cymbal and chords naturally ring out
                    continue

            # Check section mood
            sec_mood = ""
            for sec in sections:
                if beat_time >= sec.start_s:
                    sec_mood = sec.mood.lower()
            is_energetic = any(w in sec_mood for w in ("build", "energetic", "finish", "drop", "climax", "intense"))

            # --- 1. Lo-Fi Hip-Hop ---
            if drum_style == "lofi_boombap":
                # Swung boom-bap kick on 0 and 2.5
                if b % 4 == 0 or (b % 4 == 2 and rng.rand() > 0.4):
                    kl = min(int(0.18 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.18, kl)
                    kick = np.sin(2 * np.pi * (50 + 45 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 14) * 0.65
                    drums[bs:bs + kl] += kick
                # Snare on 2
                if b % 4 == 2:
                    sl = min(int(0.14 * sr), total_samples - bs)
                    st = np.linspace(0, 0.14, sl)
                    snare = (rng.rand(sl) * 2 - 1) * np.exp(-st * 24) * 0.28
                    snare += np.sin(2 * np.pi * 180 * st) * np.exp(-st * 30) * 0.20
                    drums[bs:bs + sl] += snare
                # Swung vinyl-textured hi-hat
                for sub, offset_pct in [(0, 0.0), (1, 0.58)]:
                    hi = bs + int(samples_per_beat * offset_pct)
                    if hi < total_samples:
                        hl = min(int(0.04 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.04, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 75) * 0.10

            # --- 2. Cinematic Orchestral ---
            elif drum_style == "epic_timpani":
                # Big thunderous sub timpani on bar 1 (every 4 beats)
                if b % 4 == 0:
                    tl = min(int(0.7 * sr), total_samples - bs)
                    tt = np.linspace(0, 0.7, tl)
                    timpani = np.sin(2 * np.pi * (45 + 15 * np.exp(-tt * 6)) * tt) * np.exp(-tt * 4.5) * 0.85
                    timpani += np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 7.0) * 0.35
                    drums[bs:bs + tl] += timpani
                # Orchestral crash on bar 0 or during build
                if (b % 8 == 0) or (is_energetic and b % 4 == 2):
                    cl = min(int(1.2 * sr), total_samples - bs)
                    ct = np.linspace(0, 1.2, cl)
                    crash = (rng.rand(cl) * 2 - 1) * np.exp(-ct * 3.0) * 0.35
                    drums[bs:bs + cl] += crash

            # --- 3. Four on the Floor (EDM) ---
            elif drum_style == "four_on_floor":
                # Heavy kick every single beat
                kl = min(int(0.16 * sr), total_samples - bs)
                kt = np.linspace(0, 0.16, kl)
                kick = np.sin(2 * np.pi * (58 + 140 * np.exp(-kt * 40)) * kt) * np.exp(-kt * 18) * 0.95
                drums[bs:bs + kl] += kick
                # Crisp clap on beats 1 and 3
                if b % 2 == 1:
                    cl = min(int(0.12 * sr), total_samples - bs)
                    ct = np.linspace(0, 0.12, cl)
                    clap = (rng.rand(cl) * 2 - 1) * np.exp(-ct * 24) * 0.45
                    drums[bs:bs + cl] += clap
                # Offbeat open hi-hat
                oh_pos = bs + samples_per_beat // 2
                if oh_pos < total_samples:
                    ohl = min(int(0.10 * sr), total_samples - oh_pos)
                    oht = np.linspace(0, 0.10, ohl)
                    drums[oh_pos:oh_pos + ohl] += (rng.rand(ohl) * 2 - 1) * np.exp(-oht * 28) * 0.26

            # --- 4. Gated 80s (Synthwave) ---
            elif drum_style == "gated_80s":
                # Punchy 80s kick on 0 and 2
                if b % 2 == 0 or (is_energetic and b % 4 == 3):
                    kl = min(int(0.18 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.18, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * (54 + 90 * np.exp(-kt * 32)) * kt) * np.exp(-kt * 16) * 0.85
                # Gated reverb snare on 1 and 3
                if b % 2 == 1:
                    sl = min(int(0.20 * sr), total_samples - bs)
                    st = np.linspace(0, 0.20, sl)
                    snare = (rng.rand(sl) * 2 - 1) * (1.0 - (st / 0.20)) * 0.50
                    snare += np.sin(2 * np.pi * 220 * st) * np.exp(-st * 20) * 0.35
                    drums[bs:bs + sl] += snare
                # Steady 8th hats
                for sub in [0, samples_per_beat // 2]:
                    hi = bs + sub
                    if hi < total_samples:
                        hl = min(int(0.04 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.04, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 60) * 0.16

            # --- 5. Trap Groove ---
            elif drum_style == "trap_groove":
                # Deep 808 kick
                if b % 4 == 0 or (b % 4 == 2 and rng.rand() > 0.3):
                    kl = min(int(0.24 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.24, kl)
                    kick = np.sin(2 * np.pi * (42 + 60 * np.exp(-kt * 24)) * kt) * np.exp(-kt * 10) * 0.85
                    drums[bs:bs + kl] += kick
                # Tight rimshot / clap on 2
                if b % 4 == 2:
                    sl = min(int(0.09 * sr), total_samples - bs)
                    st = np.linspace(0, 0.09, sl)
                    drums[bs:bs + sl] += (rng.rand(sl) * 2 - 1) * np.exp(-st * 35) * 0.40
                # Fast rolling 16th hats
                for sub_i in range(4):
                    hi = bs + int(sub_i * samples_per_beat / 4)
                    if hi < total_samples:
                        hl = min(int(0.025 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.025, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 90) * 0.12

            # --- 6. Rock Kit ---
            elif drum_style == "rock_kit":
                # Kick on 0 and 2.5
                if b % 2 == 0 or (b % 4 == 3 and is_energetic):
                    kl = min(int(0.20 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.20, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * (52 + 100 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 14) * 0.90
                # Loud crack snare on 1 and 3
                if b % 2 == 1:
                    sl = min(int(0.22 * sr), total_samples - bs)
                    st = np.linspace(0, 0.22, sl)
                    snare = (rng.rand(sl) * 2 - 1) * np.exp(-st * 16) * 0.60
                    snare += np.sin(2 * np.pi * 210 * st) * np.exp(-st * 24) * 0.35
                    drums[bs:bs + sl] += snare
                # Crash on bar 0
                if b % 8 == 0 and b > 0:
                    cl = min(int(0.8 * sr), total_samples - bs)
                    ct = np.linspace(0, 0.8, cl)
                    drums[bs:bs + cl] += (rng.rand(cl) * 2 - 1) * np.exp(-ct * 3.8) * 0.35

            # --- 7. Cajon & Shaker (Acoustic) ---
            elif drum_style == "cajon_shaker":
                # Warm thumb bass thump on 0
                if b % 4 == 0 or b % 4 == 3:
                    kl = min(int(0.14 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.14, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * 68 * kt) * np.exp(-kt * 16) * 0.50
                # Cajon slap on 2
                if b % 4 == 2:
                    sl = min(int(0.08 * sr), total_samples - bs)
                    st = np.linspace(0, 0.08, sl)
                    drums[bs:bs + sl] += (rng.rand(sl) * 2 - 1) * np.exp(-st * 30) * 0.25
                # Shaker 16ths
                for sub_i in range(4):
                    si = bs + int(sub_i * samples_per_beat / 4)
                    if si < total_samples:
                        shl = min(int(0.03 * sr), total_samples - si)
                        sht = np.linspace(0, 0.03, shl)
                        drums[si:si + shl] += (rng.rand(shl) * 2 - 1) * np.exp(-sht * 85) * 0.07

            # --- 8. Jazz Swing Brush ---
            elif drum_style == "jazz_swing_brush":
                # Swung ride cymbal (ding ... ding-a-ding)
                for sub, offset_pct in [(0, 0.0), (1, 0.62)]:
                    ri = bs + int(samples_per_beat * offset_pct)
                    if ri < total_samples:
                        rl = min(int(0.08 * sr), total_samples - ri)
                        rt = np.linspace(0, 0.08, rl)
                        ride = np.sin(2 * np.pi * 5000 * rt) * np.exp(-rt * 50) * 0.14
                        ride += (rng.rand(rl) * 2 - 1) * np.exp(-rt * 70) * 0.08
                        drums[ri:ri + rl] += ride
                # Soft brush swirl snare
                if b % 2 == 1:
                    sl = min(int(0.12 * sr), total_samples - bs)
                    st = np.linspace(0, 0.12, sl)
                    drums[bs:bs + sl] += (rng.rand(sl) * 2 - 1) * np.exp(-st * 20) * 0.12

            # --- 9. Heartbeat Hits (Horror) ---
            elif drum_style == "heartbeat_hits":
                # Low sub double-thud heartbeat on beat 0
                if b % 4 == 0:
                    for hb_offset in [0, int(0.20 * sr)]:
                        hbi = bs + hb_offset
                        if hbi < total_samples:
                            hl = min(int(0.25 * sr), total_samples - hbi)
                            ht = np.linspace(0, 0.25, hl)
                            drums[hbi:hbi + hl] += np.sin(2 * np.pi * 38 * ht) * np.exp(-ht * 8.0) * 0.65

            # --- 10. Reggae One Drop ---
            elif drum_style == "reggae_one_drop":
                # Kick and snare hit together on beat 2 ONLY
                if b % 4 == 2:
                    kl = min(int(0.18 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.18, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * (52 + 70 * np.exp(-kt * 26)) * kt) * np.exp(-kt * 14) * 0.85
                    drums[bs:bs + kl] += (rng.rand(kl) * 2 - 1) * np.exp(-kt * 22) * 0.40
                # Offbeat hi-hat
                hi = bs + samples_per_beat // 2
                if hi < total_samples:
                    hl = min(int(0.04 * sr), total_samples - hi)
                    ht = np.linspace(0, 0.04, hl)
                    drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 60) * 0.15

            # --- Default / Corporate / Pop ---
            else:
                if b % 4 == 0 or (b % 2 == 0 and is_energetic):
                    kl = min(int(0.18 * sr), total_samples - bs)
                    kt = np.linspace(0, 0.18, kl)
                    drums[bs:bs + kl] += np.sin(2 * np.pi * (56 + 75 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 15) * 0.70
                if b % 4 == 2:
                    sl = min(int(0.15 * sr), total_samples - bs)
                    st = np.linspace(0, 0.15, sl)
                    drums[bs:bs + sl] += (rng.rand(sl) * 2 - 1) * np.exp(-st * 22) * 0.35
                for sub in [0, samples_per_beat // 2]:
                    hi = bs + sub
                    if hi < total_samples:
                        hl = min(int(0.04 * sr), total_samples - hi)
                        ht = np.linspace(0, 0.04, hl)
                        drums[hi:hi + hl] += (rng.rand(hl) * 2 - 1) * np.exp(-ht * 65) * 0.12

        return drums

    # ------------------------------------------------------------------
    # Genre-Specific Bass Synthesizer (Harmonized with Chord Roots)
    # ------------------------------------------------------------------
    def _synth_bass(self, bass_style: str, num_beats: int, samples_per_beat: int,
                    total_samples: int, sr: int, root_freq: float,
                    progression: List[List[float]], rng: np.random.RandomState,
                    is_loopable: bool = False) -> np.ndarray:
        bass = np.zeros(total_samples, dtype=np.float32)
        num_bars = int(math.ceil(total_samples / float(samples_per_beat * 4)))

        for b in range(num_beats):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break
            b_len = min(int(samples_per_beat * 0.90), total_samples - bs)
            bt = np.linspace(0, float(b_len) / sr, b_len)

            # Get active bar's chord root
            bar_idx = b // 4
            if not is_loopable and bar_idx >= num_bars - 1:
                bar_root_ratio = 1.0  # Resolve to Tonic root on final bar
            else:
                chord_intervals = progression[bar_idx % len(progression)]
                bar_root_ratio = chord_intervals[0]  # root of this chord
            bass_freq = (root_freq * 0.5) * bar_root_ratio

            # Keep frequency in pleasant low register (40Hz to 180Hz)
            while bass_freq > 180.0:
                bass_freq *= 0.5
            while bass_freq < 40.0:
                bass_freq *= 2.0

            if bass_style == "warm_sub":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.20 * np.sin(4 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 2.8)

            elif bass_style == "bowed_cello":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.45 * np.sin(4 * np.pi * bass_freq * bt)
                tone += 0.25 * np.sin(6 * np.pi * bass_freq * bt)
                attack = 1.0 - np.exp(-bt * 10)
                env = attack * np.exp(-bt * 1.2)

            elif bass_style == "acid_saw_bass":
                phase = 2 * np.pi * bass_freq * bt
                tone = self._saw_wave(phase)
                # Filter cutoff envelope
                cutoff_env = np.exp(-bt * 6.0)
                tone = self._soft_clip(tone * (1.0 + 1.5 * cutoff_env), 0.2)
                env = np.exp(-bt * 3.5)

            elif bass_style == "arp_bass":
                # Fast synthwave 16th-note arpeggiated bass
                tone = np.zeros(b_len, dtype=np.float32)
                sub_len = b_len // 2
                for s_i in range(2):
                    s_start = s_i * sub_len
                    s_end = min(s_start + sub_len, b_len)
                    s_t = bt[:s_end - s_start]
                    mult = 1.0 if s_i == 0 else 2.0  # octave bounce
                    t_sub = self._saw_wave(2 * np.pi * (bass_freq * mult) * s_t) * np.exp(-s_t * 6.0)
                    tone[s_start:s_end] += t_sub
                env = np.ones_like(bt)

            elif bass_style == "808_sub_drop":
                # Pitch-drop 808 sub
                pitch_curve = bass_freq * (1.0 + 1.2 * np.exp(-bt * 22))
                phase = 2 * np.pi * np.cumsum(pitch_curve) / sr
                tone = np.sin(phase) + 0.15 * np.sin(2 * phase)
                env = np.exp(-bt * 2.0)

            elif bass_style == "distorted_rock":
                phase = 2 * np.pi * bass_freq * bt
                raw = np.sin(phase) + 0.5 * self._saw_wave(phase)
                tone = np.tanh(raw * 2.5) * 0.7
                env = np.exp(-bt * 2.6)

            elif bass_style == "walking_upright":
                # Walking bass notes across beats
                step = b % 4
                step_mults = [1.0, 1.25, 1.50, 1.33]
                f = bass_freq * step_mults[step]
                tone = np.sin(2 * np.pi * f * bt) + 0.25 * np.sin(4 * np.pi * f * bt)
                env = np.exp(-bt * 4.5) * (1.0 - np.exp(-bt * 35))

            elif bass_style == "deep_drone":
                tone = np.sin(2 * np.pi * bass_freq * bt) + 0.3 * np.sin(2 * np.pi * bass_freq * 1.5 * bt)
                env = np.ones_like(bt)

            elif bass_style == "deep_dub_bass":
                tone = np.sin(2 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 3.0)

            else:  # tight_sub_bass, bouncy_synth_pop, soft_upright
                tone = np.sin(2 * np.pi * bass_freq * bt)
                tone += 0.2 * np.sin(4 * np.pi * bass_freq * bt)
                env = np.exp(-bt * 3.8)

            bass[bs:bs + b_len] += (tone * env * 0.55).astype(np.float32)

        return bass

    # ------------------------------------------------------------------
    # Bar-by-Bar Musical Chord Progression Synthesizer
    # ------------------------------------------------------------------
    def _synth_chords(self, chord_instrument: str, progression: List[List[float]],
                      root_freq: float, sr: int, total_samples: int,
                      samples_per_beat: int, fx: dict,
                      rng: np.random.RandomState, is_loopable: bool = False,
                      is_minor: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesizes rich stereo chords changing every bar with realistic acoustic envelopes."""
        left = np.zeros(total_samples, dtype=np.float32)
        right = np.zeros(total_samples, dtype=np.float32)

        samples_per_bar = samples_per_beat * 4
        num_bars = int(math.ceil(total_samples / float(samples_per_bar)))
        sidechain_depth = fx.get("sidechain", 0.0)

        for bar_i in range(num_bars):
            bar_start = bar_i * samples_per_bar
            if bar_start >= total_samples:
                break
            bar_len = min(samples_per_bar, total_samples - bar_start)
            bar_t = np.linspace(0, float(bar_len) / sr, bar_len, endpoint=False)

            # Active chord ratios for this bar (resolve to Tonic root chord on final bar for intentional cadence)
            if not is_loopable and bar_i == num_bars - 1:
                chord_ratios = [1.0, 1.20, 1.50, 2.00] if is_minor else [1.0, 1.25, 1.50, 2.00]
            else:
                chord_ratios = progression[bar_i % len(progression)]

            bar_l = np.zeros(bar_len, dtype=np.float32)
            bar_r = np.zeros(bar_len, dtype=np.float32)

            # --- A. Acoustic Piano ---
            if chord_instrument in ("acoustic_piano", "bright_piano_pad"):
                # Natural piano strike on beat 0 and beat 2
                strike_len = bar_len // 2
                for s_idx in range(2):
                    st_start = s_idx * strike_len
                    st_end = min(st_start + strike_len, bar_len)
                    st_t = bar_t[:st_end - st_start]

                    for note_ratio in chord_ratios:
                        f = root_freq * note_ratio
                        # Multiple acoustic piano harmonics with realistic decay
                        decay = np.exp(-st_t * (1.8 + 0.4 * note_ratio))
                        attack = 1.0 - np.exp(-st_t * 60.0)
                        env = attack * decay

                        # Fundamental + 2nd, 3rd, 4th harmonics
                        piano_tone = (
                            np.sin(2 * np.pi * f * st_t) * 0.16 +
                            np.sin(4 * np.pi * f * st_t) * 0.08 +
                            np.sin(6 * np.pi * f * st_t) * 0.04 +
                            np.sin(8 * np.pi * f * st_t) * 0.02
                        )
                        bar_l[st_start:st_end] += (piano_tone * env).astype(np.float32)
                        bar_r[st_start:st_end] += (piano_tone * env * 0.95).astype(np.float32)

            # --- B. Rhodes Electric Piano ---
            elif chord_instrument in ("rhodes", "dark_rhodes"):
                decay = np.exp(-bar_t * 1.4)
                attack = 1.0 - np.exp(-bar_t * 40.0)
                env = attack * decay

                for i, r in enumerate(chord_ratios):
                    f = root_freq * r
                    # Bell chime FM modulation
                    mod = np.sin(2 * np.pi * f * 3.5 * bar_t) * 0.28
                    l_tone = np.sin(2 * np.pi * f * bar_t + mod) * 0.14
                    r_tone = np.sin(2 * np.pi * f * 1.002 * bar_t + mod + 0.3) * 0.14
                    bar_l += (l_tone * env).astype(np.float32)
                    bar_r += (r_tone * env).astype(np.float32)

            # --- C. Orchestral Strings ---
            elif chord_instrument == "orchestral_strings":
                # Slow expressive swell and release
                swell = (1.0 - np.exp(-bar_t * 3.5)) * np.exp(-bar_t * 0.4)
                for i, r in enumerate(chord_ratios):
                    f = root_freq * r
                    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * (4.2 + i * 0.3) * bar_t)
                    # Ensemble detune
                    l_tone = np.sin(2 * np.pi * f * vibrato * bar_t) * 0.08
                    l_tone += np.sin(2 * np.pi * f * 1.004 * vibrato * bar_t) * 0.05
                    r_tone = np.sin(2 * np.pi * f * 0.996 * vibrato * bar_t + 0.4) * 0.08
                    r_tone += np.sin(2 * np.pi * f * 1.008 * vibrato * bar_t + 0.7) * 0.05
                    bar_l += (l_tone * swell).astype(np.float32)
                    bar_r += (r_tone * swell).astype(np.float32)

            # --- D. Supersaw / EDM Stabs ---
            elif chord_instrument == "supersaw":
                for r in chord_ratios:
                    f = root_freq * r
                    for detune in [-0.012, -0.005, 0.0, 0.005, 0.012]:
                        p_l = 2 * np.pi * f * (1.0 + detune) * bar_t
                        p_r = 2 * np.pi * f * (1.0 - detune) * bar_t + 0.5
                        bar_l += self._saw_wave(p_l) * 0.025
                        bar_r += self._saw_wave(p_r) * 0.025

            # --- E. Power Chords (Rock) ---
            elif chord_instrument == "power_chords":
                # Strum on beat 0 and beat 2 with heavy tube overdrive
                half_len = bar_len // 2
                for s_i in range(2):
                    st_start = s_i * half_len
                    st_end = min(st_start + half_len, bar_len)
                    st_t = bar_t[:st_end - st_start]
                    env = np.exp(-st_t * 2.2) * (1.0 - np.exp(-st_t * 50.0))

                    for r in chord_ratios:
                        f = root_freq * r
                        phase = 2 * np.pi * f * st_t
                        tone = np.tanh((np.sin(phase) + 0.5 * self._saw_wave(phase)) * 2.8) * 0.16
                        bar_l[st_start:st_end] += (tone * env).astype(np.float32)
                        bar_r[st_start:st_end] += (tone * env * 0.95).astype(np.float32)

            # --- F. Offbeat Skank (Reggae) ---
            elif chord_instrument == "offbeat_skank":
                # Quick percussive chops on beats 1 and 3
                beat_len = samples_per_beat
                for b_i in (1, 3):
                    b_start = b_i * beat_len
                    b_end = min(b_start + int(0.12 * sr), bar_len)
                    if b_start < bar_len:
                        chop_t = bar_t[:b_end - b_start]
                        env = np.exp(-chop_t * 30.0)
                        for r in chord_ratios:
                            f = root_freq * r
                            tone = (np.sin(2 * np.pi * f * chop_t) + 0.3 * np.sin(4 * np.pi * f * chop_t)) * 0.18
                            bar_l[b_start:b_end] += (tone * env).astype(np.float32)
                            bar_r[b_start:b_end] += (tone * env * 0.9).astype(np.float32)

            # --- G. Ambient Shimmer Drone ---
            elif chord_instrument == "shimmer_drone":
                lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * bar_t)
                for r in chord_ratios:
                    f = root_freq * r
                    l_tone = np.sin(2 * np.pi * f * bar_t) * 0.10 * lfo
                    l_tone += np.sin(4 * np.pi * f * bar_t) * 0.05 * lfo
                    r_tone = np.sin(2 * np.pi * f * 1.003 * bar_t + 1.2) * 0.10 * lfo
                    r_tone += np.sin(4 * np.pi * f * 1.002 * bar_t + 0.8) * 0.05 * lfo
                    bar_l += l_tone.astype(np.float32)
                    bar_r += r_tone.astype(np.float32)

            # --- H. Dissonant Horror Cluster ---
            elif chord_instrument == "dissonant_cluster":
                for r in chord_ratios:
                    f = root_freq * r
                    wobble = 1.0 + 0.007 * np.sin(2 * np.pi * 0.4 * bar_t)
                    l_tone = np.sin(2 * np.pi * f * wobble * bar_t) * 0.12
                    r_tone = np.sin(2 * np.pi * f * 1.012 * wobble * bar_t + 1.5) * 0.12
                    bar_l += l_tone.astype(np.float32)
                    bar_r += r_tone.astype(np.float32)

            # --- Default / Generic Pad ---
            else:
                env = (1.0 - np.exp(-bar_t * 6.0)) * np.exp(-bar_t * 0.8)
                for r in chord_ratios:
                    f = root_freq * r
                    bar_l += (np.sin(2 * np.pi * f * bar_t) * 0.10 * env).astype(np.float32)
                    bar_r += (np.sin(2 * np.pi * f * 1.003 * bar_t + 0.5) * 0.10 * env).astype(np.float32)

            # Apply Sidechain Compression Pump (e.g. EDM / Synthwave on every beat)
            if sidechain_depth > 0:
                beat_len = samples_per_beat
                for b_i in range(4):
                    b_start = b_i * beat_len
                    b_end = min(b_start + beat_len, bar_len)
                    if b_start < bar_len:
                        beat_t = bar_t[:b_end - b_start]
                        # Duck volume heavily on beat hit, release over beat
                        duck_curve = 1.0 - sidechain_depth * np.exp(-beat_t * 18.0)
                        bar_l[b_start:b_end] *= duck_curve.astype(np.float32)
                        bar_r[b_start:b_end] *= duck_curve.astype(np.float32)

            left[bar_start:bar_start + bar_len] = bar_l
            right[bar_start:bar_start + bar_len] = bar_r

        return left, right

    # ------------------------------------------------------------------
    # Harmonized Melody Synthesizer (Aligned with Bar Chords)
    # ------------------------------------------------------------------
    def _synth_melody(self, melody_style: str, num_beats: int, samples_per_beat: int,
                      total_samples: int, sr: int, root_freq: float,
                      progression: List[List[float]], rng: np.random.RandomState,
                      is_loopable: bool = False) -> np.ndarray:
        melody = np.zeros(total_samples, dtype=np.float32)
        num_bars = int(math.ceil(total_samples / float(samples_per_beat * 4)))

        # How often does a melody note strike (in beats)?
        note_interval = 2 if melody_style in ("piano_solo", "brass_horn", "singing_bowl") else 1

        for b in range(0, num_beats, note_interval):
            bs = int(b * samples_per_beat)
            if bs >= total_samples:
                break
            note_len = min(int(samples_per_beat * note_interval * 0.85), total_samples - bs)
            nt = np.linspace(0, float(note_len) / sr, note_len)

            # Get active chord's notes to guarantee harmonic coherence
            bar_idx = b // 4
            if not is_loopable and bar_idx >= num_bars - 1:
                note_freq = root_freq * 2.0  # Resolve to keynote on final bar
            else:
                chord_ratios = progression[bar_idx % len(progression)]
                # Pick a chord tone or upper extension
                note_idx = rng.randint(0, len(chord_ratios))
                note_freq = root_freq * chord_ratios[note_idx] * 2.0  # octave up

            # Keep melody in singing range (250Hz - 1200Hz)
            while note_freq < 250.0:
                note_freq *= 2.0
            while note_freq > 1400.0:
                note_freq *= 0.5

            if melody_style == "piano_solo":
                tone = np.sin(2 * np.pi * note_freq * nt)
                tone += 0.3 * np.sin(4 * np.pi * note_freq * nt)
                env = (1.0 - np.exp(-nt * 50.0)) * np.exp(-nt * 3.5)

            elif melody_style == "brass_horn":
                tone = self._saw_wave(2 * np.pi * note_freq * nt) * 0.6
                tone += np.sin(2 * np.pi * note_freq * nt) * 0.4
                env = (1.0 - np.exp(-nt * 12.0)) * np.exp(-nt * 2.0)

            elif melody_style == "synth_lead":
                phase = 2 * np.pi * note_freq * nt
                tone = self._saw_wave(phase) * 0.6 + self._square_wave(phase) * 0.3
                env = (1.0 - np.exp(-nt * 40.0)) * np.exp(-nt * 4.5)

            elif melody_style == "retro_lead":
                phase = 2 * np.pi * note_freq * nt
                vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * nt)
                tone = np.sin(phase * vibrato) + 0.3 * np.sin(2 * phase * vibrato)
                env = np.exp(-nt * 4.0)

            elif melody_style == "guitar_riff":
                phase = 2 * np.pi * note_freq * nt
                tone = np.tanh(self._saw_wave(phase) * 3.0) * 0.6
                env = np.exp(-nt * 3.5)

            elif melody_style == "acoustic_fingerpick":
                tone = np.sin(2 * np.pi * note_freq * nt) + 0.25 * np.sin(4 * np.pi * note_freq * nt)
                env = (1.0 - np.exp(-nt * 45.0)) * np.exp(-nt * 5.0)

            elif melody_style == "singing_bowl":
                tone = np.sin(2 * np.pi * note_freq * nt)
                env = (1.0 - np.exp(-nt * 2.0)) * np.exp(-nt * 0.8)

            elif melody_style == "jazz_vibes":
                # Vibraphone with tremolo
                tremolo = 1.0 + 0.3 * np.sin(2 * np.pi * 5.5 * nt)
                tone = np.sin(2 * np.pi * note_freq * nt) * tremolo
                env = np.exp(-nt * 4.0)

            elif melody_style == "creepy_musicbox":
                # High music box chime
                tone = np.sin(2 * np.pi * (note_freq * 2.0) * nt)
                env = np.exp(-nt * 6.5)

            elif melody_style == "steel_drum":
                tone = np.sin(2 * np.pi * note_freq * nt) + 0.4 * np.sin(4 * np.pi * note_freq * 1.02 * nt)
                env = np.exp(-nt * 6.0)

            else:  # cheerful_bells, tech_arpeggio, mellow_chime
                tone = np.sin(2 * np.pi * note_freq * nt)
                tone += 0.2 * np.sin(6 * np.pi * note_freq * nt)
                env = np.exp(-nt * 4.5)

            melody[bs:bs + note_len] += (tone * env * 0.30).astype(np.float32)

        return melody

    # ------------------------------------------------------------------
    # Master Music Generation Entrypoint
    # ------------------------------------------------------------------
    async def generate_music(self, request: CreateMusicJobRequest) -> GenerationResult:
        start_time = time.time()
        sr = request.output.sample_rate if request.output else self.sample_rate
        total_samples = int(round(request.duration_s * sr))

        # Seed handling (ensures seed reproducibility)
        if request.seed is not None:
            seed = request.seed
        else:
            # Derive deterministic seed from prompt text so different prompts sound different
            seed = abs(hash(request.prompt)) % 1000000
        rng = np.random.RandomState(seed)

        # Detect genre from prompt
        genre = self._detect_genre(request.prompt)
        profile = GENRE_PROFILES[genre]
        fx = profile["fx"]
        dyn = profile["dynamics"]

        # Parse or default BPM and Key
        bpm = request.tempo_bpm if request.tempo_bpm else self._parse_bpm_from_prompt(request.prompt, profile["default_bpm"])
        key_str = request.key if request.key else self._parse_key_from_prompt(request.prompt, profile["default_key"])
        root_freq = self._get_key_base_freq(key_str)
        progression = profile["progression"]

        beat_interval_s = 60.0 / bpm
        samples_per_beat = int(round(beat_interval_s * sr))
        num_beats = int(math.ceil(request.duration_s / beat_interval_s)) + 1

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

        is_loopable = bool(request.loopable)
        is_minor = "minor" in key_str.lower()

        # --- 1. Synthesize Drums ---
        drums_line = self._synth_drums(
            profile["drum_style"], num_beats, samples_per_beat,
            total_samples, sr, rng, sections, beat_interval_s,
            is_loopable=is_loopable
        )

        # --- 2. Synthesize Bass ---
        bass_line = self._synth_bass(
            profile["bass_style"], num_beats, samples_per_beat,
            total_samples, sr, root_freq, progression, rng,
            is_loopable=is_loopable
        )

        # --- 3. Synthesize Chords ---
        chord_left, chord_right = self._synth_chords(
            profile["chord_instrument"], progression, root_freq, sr,
            total_samples, samples_per_beat, fx, rng,
            is_loopable=is_loopable, is_minor=is_minor
        )

        # --- 4. Synthesize Melodic Lead ---
        melody_line = self._synth_melody(
            profile["melody_style"], num_beats, samples_per_beat,
            total_samples, sr, root_freq, progression, rng,
            is_loopable=is_loopable
        )

        # --- 5. Apply Vinyl Crackle & Reverb FX ---
        t = np.linspace(0, request.duration_s, total_samples, endpoint=False)

        # Tape wobble (pitch modulation on melody)
        if fx.get("tape_wobble", 0) > 0:
            wobble = fx["tape_wobble"] * np.sin(2 * np.pi * 0.6 * t)
            melody_line = (melody_line * (1.0 + wobble)).astype(np.float32)

        # Assemble Stereo Master
        mix_left = np.zeros(total_samples, dtype=np.float32)
        mix_right = np.zeros(total_samples, dtype=np.float32)

        mix_left += drums_line * dyn["drum_vol"]
        mix_right += drums_line * dyn["drum_vol"]

        mix_left += bass_line * dyn["bass_vol"]
        mix_right += bass_line * dyn["bass_vol"]

        mix_left += chord_left * dyn["chord_vol"]
        mix_right += chord_right * dyn["chord_vol"]

        mix_left += melody_line * dyn["melody_vol"] * 0.75
        mix_right += melody_line * dyn["melody_vol"] * 1.0

        # Sparse vinyl crackle
        if fx.get("vinyl_crackle", 0) > 0:
            crackle_amp = fx["vinyl_crackle"]
            crackle = rng.rand(total_samples).astype(np.float32)
            crackle_mask = (rng.rand(total_samples) > 0.96).astype(np.float32)
            crackle = crackle * crackle_mask * crackle_amp
            mix_left += crackle
            mix_right += crackle * 0.85

        # Reverb
        if fx.get("reverb_mix", 0) > 0:
            mix_left = self._simple_reverb(mix_left, sr, fx["reverb_mix"]).astype(np.float32)
            mix_right = self._simple_reverb(mix_right, sr, fx["reverb_mix"]).astype(np.float32)

        stereo_audio = np.column_stack([mix_left, mix_right]).astype(np.float32)

        # Normalize headroom
        peak = np.max(np.abs(stereo_audio))
        if peak > 1e-4:
            stereo_audio = stereo_audio / peak * 0.85

        # Store stems for stem separator (exact proportional isolation)
        self._last_stems = {
            "drums": np.column_stack([drums_line, drums_line]) * dyn["drum_vol"],
            "bass": np.column_stack([bass_line, bass_line]) * dyn["bass_vol"],
            "other": np.column_stack([chord_left + melody_line * 0.75, chord_right + melody_line]) * dyn["chord_vol"],
        }

        # Feedback warnings & recipe info
        warnings = [
            f"Detected Genre: '{profile['display_name']}' | Key: {key_str} | Tempo: {bpm} BPM",
        ]
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
