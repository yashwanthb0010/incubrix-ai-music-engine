"""
Verification script for IncuBrix AI Background Music Engine.
Validates:
1. Exact length: The mix and every stem match requested duration within 0.05s.
2. Intentional ending: The track ends on purpose (final hit or bar-line fade), never a hard cut.
3. Tests across 10 diverse outputs, measuring lengths and analyzing the last 3 seconds.
"""

import asyncio
import os
import soundfile as sf
import numpy as np
from pathlib import Path
from incubrix.schemas import CreateMusicJobRequest
from incubrix.queue.pipeline import execute_job_pipeline
from incubrix.storage.local import LocalStorage
from incubrix.db.database import Database
from incubrix.config import get_settings

TEST_CASES = [
    {"name": "01_podcast_sting", "prompt": "Short punchy podcast sting with electronic drums", "duration_s": 5.0, "seed": 101},
    {"name": "02_synthwave_drive", "prompt": "Fast 80s synthwave neon drive at 128 bpm", "duration_s": 10.0, "seed": 202},
    {"name": "03_lofi_study", "prompt": "Mellow lo-fi hip hop study beats with vinyl crackle", "duration_s": 15.0, "seed": 303},
    {"name": "04_epic_cinematic", "prompt": "Epic dramatic cinematic trailer with timpani and strings", "duration_s": 20.0, "seed": 404},
    {"name": "05_sad_piano", "prompt": "Emotional melancholic piano melody with soft chords", "duration_s": 30.0, "seed": 505},
    {"name": "06_acoustic_folk", "prompt": "Warm acoustic folk guitar morning sunshine with cajon", "duration_s": 45.0, "seed": 606},
    {"name": "07_heavy_rock", "prompt": "Heavy rock with distorted electric guitar power chords", "duration_s": 60.0, "seed": 707},
    {"name": "08_jazz_lounge", "prompt": "Sophisticated smooth midnight jazz lounge with upright bass", "duration_s": 30.0, "seed": 808},
    {"name": "09_dark_horror", "prompt": "Dark scary horror suspense thriller with eerie dissonance", "duration_s": 25.0, "seed": 909},
    {"name": "10_corporate_tech", "prompt": "Modern corporate tech product launch explainer", "duration_s": 40.0, "seed": 1010},
]

async def verify_all():
    settings = get_settings()
    db = Database(settings.database_url)
    storage = LocalStorage(settings.storage_local_dir)
    results = []

    print("=" * 95)
    print("RUNNING 10-OUTPUT DURATION & LAST-3-SECONDS ENDING VERIFICATION")
    print("=" * 95)

    for idx, tc in enumerate(TEST_CASES, 1):
        import uuid
        job_id = f"test_{idx}_{uuid.uuid4().hex[:8]}"
        req = CreateMusicJobRequest(
            prompt=tc["prompt"],
            duration_s=tc["duration_s"],
            seed=tc["seed"],
        )
        db.create_job(job_id, req.model_dump(), initial_warnings=[])

        # Execute pipeline
        await execute_job_pipeline(job_id, settings)

        # Inspect exported files from storage
        job_dir = Path(settings.storage_local_dir) / job_id

        # Check mix.wav
        mix_path = job_dir / "mix.wav"
        assert mix_path.exists(), f"mix.wav missing for {job_id}"
        mix_info = sf.info(str(mix_path))
        mix_dur = mix_info.duration
        mix_data, sr = sf.read(str(mix_path))

        # Check stems
        stems_durs = {}
        stem_files = list(job_dir.glob("stem_*.wav"))
        for sf_path in stem_files:
            s_name = sf_path.stem.replace("stem_", "")
            s_info = sf.info(str(sf_path))
            stems_durs[s_name] = s_info.duration

        # Analyze last 3 seconds of mix
        last_3s_samples = int(min(3.0, tc["duration_s"]) * sr)
        last_3s_audio = mix_data[-last_3s_samples:, :]

        # Check RMS in segments of last 3 seconds
        half_s = int(0.5 * sr)
        ten_ms = int(0.01 * sr)

        rms_start_last3s = np.sqrt(np.mean(last_3s_audio[:half_s, :] ** 2))
        rms_tail_half_s = np.sqrt(np.mean(last_3s_audio[-half_s:, :] ** 2))
        final_sample_abs = np.max(np.abs(mix_data[-1, :]))

        # Smooth taper check: energy at the very tail must be significantly lower than start of last 3s
        is_fading_or_silent = (rms_tail_half_s < rms_start_last3s * 0.3) or (rms_tail_half_s < 1e-4)
        is_hard_cut = (final_sample_abs > 0.05) or (not is_fading_or_silent and rms_tail_half_s > 0.15)

        # Durations check
        dur_err_mix = abs(mix_dur - tc["duration_s"])
        stem_errs = [abs(sd - tc["duration_s"]) for sd in stems_durs.values()]
        max_stem_err = max(stem_errs) if stem_errs else 0.0

        all_under_tolerance = (dur_err_mix <= 0.05) and (max_stem_err <= 0.05)

        res_entry = {
            "idx": idx,
            "name": tc["name"],
            "requested_s": tc["duration_s"],
            "mix_s": round(mix_dur, 5),
            "mix_err_s": round(dur_err_mix, 5),
            "max_stem_err_s": round(max_stem_err, 5),
            "stems_count": len(stem_files),
            "rms_tail": round(float(rms_tail_half_s), 6),
            "final_sample": round(float(final_sample_abs), 6),
            "duration_pass": all_under_tolerance,
            "ending_intentional": not is_hard_cut and final_sample_abs < 1e-5,
        }
        results.append(res_entry)

        status_dur = "PASS" if all_under_tolerance else "FAIL"
        status_end = "PASS (Smooth/Hit)" if res_entry["ending_intentional"] else "FAIL (Hard Cut)"
        print(f"[{idx:02d}/10] {tc['name']:<22} | Req: {tc['duration_s']:>4.1f}s | Mix: {mix_dur:>7.4f}s (err: {dur_err_mix:.5f}s) | Stems: {len(stem_files)} (max err: {max_stem_err:.5f}s) | Duration: {status_dur} | Ending: {status_end}")

    print("=" * 95)
    all_dur_pass = all(r["duration_pass"] for r in results)
    all_end_pass = all(r["ending_intentional"] for r in results)
    print(f"OVERALL SUMMARY:")
    print(f"1. Exact length (within 0.05s) for mix & all stems across 10 outputs: {'10/10 PASSED' if all_dur_pass else 'FAILED'}")
    print(f"2. Intentional ending (never hard cut, resolves and tapers to 0) across 10 outputs: {'10/10 PASSED' if all_end_pass else 'FAILED'}")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(verify_all())
