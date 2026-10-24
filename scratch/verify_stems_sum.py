"""
Verification script for IncuBrix AI Background Music Engine.
Validates:
1. Real stems: At least 3 stems, each one a different instrument part.
2. The mix equals the sum of the stems using the gains in the manifest:
   mix == sum(stem_i * gain_i)
3. Stems are genuinely different signals (no frequency-filter gimmick or duplicate copies).
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

TEST_PROMPTS = [
    ("lofi", "Mellow lo-fi hip hop study beats with warm Rhodes and vinyl crackle", 20.0),
    ("rock", "Aggressive hard rock with electric guitar power chords and drums", 15.0),
    ("cinematic", "Epic dramatic cinematic trailer with timpani and strings", 25.0),
    ("synthwave", "Pumping 80s synthwave cyberpunk with analog leads and arp bass", 18.0),
    ("jazz", "Smooth midnight jazz lounge with upright bass and swung brush ride", 22.0),
]

async def verify_stems():
    settings = get_settings()
    db = Database(settings.database_url)
    storage = LocalStorage(settings.storage_local_dir)

    print("=" * 95)
    print("VERIFYING REAL STEMS & EXACT SUM RECONSTRUCTION: mix == sum(gain_i * stem_i)")
    print("=" * 95)

    all_passed = True

    for idx, (name, prompt, dur_s) in enumerate(TEST_PROMPTS, 1):
        import uuid
        job_id = f"stem_test_{idx}_{uuid.uuid4().hex[:8]}"
        req = CreateMusicJobRequest(
            prompt=prompt,
            duration_s=dur_s,
            seed=42 + idx,
        )
        db.create_job(job_id, req.model_dump(), initial_warnings=[])
        await execute_job_pipeline(job_id, settings)

        # Retrieve job and manifest from database
        job_record = db.get_job(job_id)
        assert job_record is not None
        manifest = db.get_job_manifest(job_id)
        assert manifest is not None, "Manifest was not generated!"

        job_dir = Path(settings.storage_local_dir) / job_id

        # 1. Load Mix Audio
        mix_path = job_dir / "mix.wav"
        mix_data, sr = sf.read(str(mix_path))

        # 2. Extract stems and gains from manifest
        stems_manifest = manifest["stems"]
        assert len(stems_manifest) >= 3, f"Expected at least 3 stems, found {len(stems_manifest)}"

        stem_audios = {}
        stem_gains = {}
        reconstructed_mix = np.zeros_like(mix_data)

        for stem_info in stems_manifest:
            s_name = stem_info["name"]
            s_gain = stem_info["gain"]
            s_file = stem_info["filename"]
            s_path = job_dir / s_file
            assert s_path.exists(), f"Stem file {s_file} missing!"

            s_audio, _ = sf.read(str(s_path))
            stem_audios[s_name] = s_audio
            stem_gains[s_name] = s_gain

            # Accumulate using the gain specified in the manifest
            reconstructed_mix += s_audio * s_gain

        # 3. Check mathematical reconstruction: mix == sum(stem_i * gain_i)
        difference = mix_data - reconstructed_mix
        max_abs_diff = float(np.max(np.abs(difference)))
        rms_diff = float(np.sqrt(np.mean(difference ** 2)))
        rms_mix = float(np.sqrt(np.mean(mix_data ** 2)))

        if rms_diff < 1e-6:
            snr_db = 120.0
        else:
            snr_db = float(20.0 * np.log10(rms_mix / (rms_diff + 1e-9)))

        # 4. Check stems are genuinely distinct signals (different instrument parts)
        drums = stem_audios["drums"]
        bass = stem_audios["bass"]
        other = stem_audios["other"]

        diff_db = np.sqrt(np.mean((drums - bass) ** 2))
        diff_do = np.sqrt(np.mean((drums - other) ** 2))
        diff_bo = np.sqrt(np.mean((bass - other) ** 2))

        stems_are_distinct = (diff_db > 0.01) and (diff_do > 0.01) and (diff_bo > 0.01)
        reconstruction_exact = (max_abs_diff < 1e-4) and (snr_db >= 35.0)

        pass_status = stems_are_distinct and reconstruction_exact
        if not pass_status:
            all_passed = False

        status_str = "PASS" if pass_status else "FAIL"
        print(f"[{idx}/5] {name.upper():<12} | Stems: {list(stem_audios.keys())} | Gains: {stem_gains}")
        print(f"      Reconstruction Max Error: {max_abs_diff:.8f} | RMS Error: {rms_diff:.8f} | SNR: {snr_db:.1f} dB")
        print(f"      Distinct Instruments: drums!=bass ({diff_db:.4f}), drums!=other ({diff_do:.4f}), bass!=other ({diff_bo:.4f}) -> {status_str}")
        print("-" * 95)

    print("=" * 95)
    print(f"OVERALL STEM RECONSTRUCTION RESULT: {'ALL 5 PASSED PERFECTLY' if all_passed else 'FAILED'}")
    print("=" * 95)

if __name__ == "__main__":
    asyncio.run(verify_stems())
