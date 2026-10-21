"""
Job Execution Pipeline for IncuBrix AI Background Music Engine.
Orchestrates:
Music Provider -> Audio Processing (LUFS/True Peak/Fade) -> Stem Separation & Alignment
-> Mix Reconstruction Fidelity -> Narration Ducking -> Multi-format Export -> Validation -> Manifest.
"""

from __future__ import annotations
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import soundfile as sf
import httpx

from incubrix.config import Settings, get_settings
from incubrix.db.database import Database, get_db
from incubrix.schemas import (
    CreateMusicJobRequest,
    JobStatus,
    MusicManifest,
    StemInfo,
    DuckingManifest,
    JobAssets,
)
from incubrix.storage import get_storage
from incubrix.providers import get_music_provider
from incubrix.stems import get_stem_separator
from incubrix.audio.processor import AudioProcessor
from incubrix.audio.validator import AudioValidator
from incubrix.audio.ducking import DuckingEngine
from incubrix.audio.ffmpeg_utils import convert_wav_to_mp3
from incubrix.errors import IncubrixError, ErrorCode


async def fetch_narration_audio(url: str, max_size_mb: int = 50) -> Tuple[np.ndarray, int]:
    """Safely fetch narration audio file from URL with size limits."""
    if not url.startswith("http://") and not url.startswith("https://"):
        raise IncubrixError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid narration URL schema: '{url}'. Must start with http:// or https://",
            status_code=400,
        )

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise IncubrixError(
                    code=ErrorCode.NARRATION_FETCH_FAILED,
                    message=f"Failed to fetch narration from URL: HTTP status {resp.status_code}",
                    status_code=400,
                )
            content = resp.content
            if len(content) > max_size_mb * 1024 * 1024:
                raise IncubrixError(
                    code=ErrorCode.VALIDATION_ERROR,
                    message=f"Narration file exceeds maximum allowed size ({max_size_mb} MB).",
                    status_code=400,
                )

            data, sr = sf.read(io.BytesIO(content))
            return data.astype(np.float32), sr
    except IncubrixError:
        raise
    except Exception as e:
        raise IncubrixError(
            code=ErrorCode.NARRATION_FETCH_FAILED,
            message=f"Error reading narration audio: {str(e)}",
            status_code=400,
        )


async def execute_job_pipeline(job_id: str, settings: Optional[Settings] = None) -> None:
    """Full execution of the audio generation and stem processing pipeline."""
    cfg = settings or get_settings()
    db = get_db(cfg.database_url)
    storage = get_storage(cfg)
    music_provider = get_music_provider(cfg)
    stem_separator = get_stem_separator(cfg)
    processor = AudioProcessor(
        target_sample_rate=cfg.default_sample_rate,
        target_lufs=cfg.target_lufs,
        target_true_peak_dbtp=cfg.target_true_peak_dbtp,
    )
    validator = AudioValidator(
        duration_tolerance_s=cfg.duration_tolerance_s,
        target_lufs=cfg.target_lufs,
        max_true_peak_dbtp=cfg.target_true_peak_dbtp,
        required_sample_rate=cfg.default_sample_rate,
    )
    ducking_engine = DuckingEngine()

    pipeline_start_time = time.time()
    created_at_str = datetime.now(timezone.utc).isoformat()

    try:
        req_dict = db.get_job_request(job_id)
        if not req_dict:
            raise IncubrixError(ErrorCode.JOB_NOT_FOUND, f"Job {job_id} not found", status_code=404)

        request = CreateMusicJobRequest(**req_dict)
        target_sr = request.output.sample_rate if request.output else cfg.default_sample_rate
        target_lufs = request.output.target_lufs if request.output else cfg.target_lufs

        # Step 1: Update status to running (10%)
        db.update_job_progress(job_id, JobStatus.RUNNING, 10)

        # Step 2: Generate Music via Provider (30%)
        gen_result = await music_provider.generate_music(request)
        db.update_job_progress(job_id, JobStatus.RUNNING, 30, warnings=gen_result.warnings)

        # Step 3: Resample, normalize, and enforce exact duration
        raw_audio = processor.ensure_stereo(gen_result.audio)
        if gen_result.sample_rate != target_sr:
            resampled = processor.resample(raw_audio, gen_result.sample_rate, target_sr)
        else:
            resampled = raw_audio

        # Enforce exact duration with musical fade-out
        duration_enforced = processor.enforce_exact_duration(
            resampled, target_sr, request.duration_s, fade_out_duration_s=1.5
        )

        # Handle loopable if requested
        loop_boundary_discontinuity = None
        if request.loopable:
            duration_enforced, loop_boundary_discontinuity = processor.apply_loop_crossfade(
                duration_enforced, target_sr, crossfade_s=0.5
            )

        # Normalize LUFS and True Peak
        normalized_mix, measured_lufs, measured_peak, gain_linear = processor.normalize_loudness_and_peak(
            duration_enforced,
            target_sr,
            target_lufs=target_lufs,
            max_true_peak_dbtp=cfg.target_true_peak_dbtp,
        )

        # Step 4: Stem Separation (50%)
        db.update_job_progress(job_id, JobStatus.RUNNING, 50)
        separation_result = await stem_separator.separate_stems(
            normalized_mix, target_sr, provider_ref=music_provider
        )

        # Align stems to exact duration and sample rate
        aligned_stems: Dict[str, Tuple[np.ndarray, float]] = {}
        for stem_name, (stem_data, stem_gain) in separation_result.stems.items():
            s_stereo = processor.ensure_stereo(stem_data)
            s_dur = processor.enforce_exact_duration(
                s_stereo, target_sr, request.duration_s, fade_out_duration_s=0.0
            )
            aligned_stems[stem_name] = (s_dur, stem_gain)

        # Measure Stem Reconstruction Fidelity
        reconstruction_measurements = validator.measure_stem_reconstruction(
            final_mix=normalized_mix,
            stems=aligned_stems,
            sample_rate=target_sr,
        )
        if separation_result.limitations:
            reconstruction_measurements.limitations = separation_result.limitations

        # Step 5: Narration Ducking (70%)
        db.update_job_progress(job_id, JobStatus.RUNNING, 70)
        ducked_mix: Optional[np.ndarray] = None
        ducking_manifest_data = DuckingManifest(applied=False)

        if request.narration_url:
            narration_audio, narr_sr = await fetch_narration_audio(
                request.narration_url, max_size_mb=cfg.max_narration_file_size_mb
            )
            if narr_sr != target_sr:
                narration_audio = processor.resample(narration_audio, narr_sr, target_sr)

            ducked_mix, d_info = ducking_engine.apply_ducking(
                music_audio=normalized_mix,
                narration_audio=narration_audio,
                sample_rate=target_sr,
            )
            ducking_manifest_data = DuckingManifest(
                applied=True,
                narration_source=request.narration_url,
                attenuation_db=d_info["attenuation_db"],
                speech_detected=d_info["speech_detected"],
                speech_duration_s=d_info["speech_duration_s"],
                attack_ms=d_info["attack_ms"],
                release_ms=d_info["release_ms"],
                volume_curve_summary=d_info["volume_curve_summary"],
            )

        # Step 6: Export Audio Files to Storage (85%)
        db.update_job_progress(job_id, JobStatus.RUNNING, 85)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assets_dict: Dict[str, Any] = {"stems": {}}

            # 6a. Export Mix WAV
            mix_wav_path = temp_path / "mix.wav"
            sf.write(str(mix_wav_path), normalized_mix, target_sr, subtype="PCM_16")
            await storage.save_file(job_id, "mix.wav", mix_wav_path, content_type="audio/wav")
            assets_dict["mix_wav"] = f"/api/v1/jobs/{job_id}/download/mix.wav"

            # 6b. Export Mix MP3 if requested
            formats = request.output.formats if request.output else ["wav", "mp3"]
            if "mp3" in formats:
                mix_mp3_path = temp_path / "mix.mp3"
                if convert_wav_to_mp3(mix_wav_path, mix_mp3_path):
                    await storage.save_file(job_id, "mix.mp3", mix_mp3_path, content_type="audio/mpeg")
                    assets_dict["mix_mp3"] = f"/api/v1/jobs/{job_id}/download/mix.mp3"

            # 6c. Export Stems
            stems_manifest_list: List[StemInfo] = []
            for stem_name, (stem_audio, stem_gain) in aligned_stems.items():
                stem_filename = f"stem_{stem_name}.wav"
                stem_file_path = temp_path / stem_filename
                sf.write(str(stem_file_path), stem_audio, target_sr, subtype="PCM_16")
                await storage.save_file(job_id, stem_filename, stem_file_path, content_type="audio/wav")

                stem_download_url = f"/api/v1/jobs/{job_id}/download/{stem_filename}"
                assets_dict["stems"][stem_name] = stem_download_url
                stems_manifest_list.append(
                    StemInfo(
                        name=stem_name,
                        filename=stem_filename,
                        gain=stem_gain,
                        production_method=separation_result.production_method,
                        sample_rate=target_sr,
                        duration_s=round(float(stem_audio.shape[0] / target_sr), 4),
                        download_url=stem_download_url,
                    )
                )

            # 6d. Export Ducked Mix if present
            if ducked_mix is not None:
                ducked_wav_path = temp_path / "mix_ducked.wav"
                sf.write(str(ducked_wav_path), ducked_mix, target_sr, subtype="PCM_16")
                await storage.save_file(job_id, "mix_ducked.wav", ducked_wav_path, content_type="audio/wav")
                assets_dict["mix_ducked_wav"] = f"/api/v1/jobs/{job_id}/download/mix_ducked.wav"

            # Step 7: Final Audio Quality Validation (95%)
            db.update_job_progress(job_id, JobStatus.RUNNING, 95)
            quality_measurements = validator.validate_and_measure(
                audio=normalized_mix,
                sample_rate=target_sr,
                requested_duration_s=request.duration_s,
            )

            # Step 8: Assemble Manifest
            completed_at_str = datetime.now(timezone.utc).isoformat()
            total_processing_s = round(time.time() - pipeline_start_time, 3)

            provider_info = music_provider.get_provider_info()
            rights_info = {
                "license_name": provider_info.get("license", "Unknown"),
                "commercial_use_permitted": True if provider_info.get("is_mock") else "Conditional (under $1M revenue threshold)",
                "conditions": provider_info.get("license_conditions", "None"),
                "provenance": "Generated by IncuBrix AI Background Music Engine",
            }

            known_limitations = []
            if gen_result.warnings:
                known_limitations.extend(gen_result.warnings)
            if separation_result.limitations:
                known_limitations.append(separation_result.limitations)
            if request.duration_s > 120 and cfg.music_provider == "stable_audio":
                known_limitations.append("Stable Audio native limit is 120s; extended track was produced via segmented generation.")

            manifest = MusicManifest(
                job_id=job_id,
                created_at=created_at_str,
                completed_at=completed_at_str,
                request=req_dict,
                model_provider=provider_info.get("provider_name", "Mock"),
                model_name=provider_info.get("model_name", "mock"),
                model_version=provider_info.get("model_version", "1.0.0"),
                seed=gen_result.seed_used,
                requested_duration_s=request.duration_s,
                actual_duration_s=quality_measurements.measured_duration_s,
                tempo_bpm=gen_result.tempo_bpm,
                key=gen_result.key,
                sections=[s.model_dump() for s in (request.sections or [])],
                section_transitions_measured=gen_result.section_measurements,
                loopable=request.loopable,
                loop_boundary_checked=bool(request.loopable),
                loop_max_boundary_discontinuity=round(loop_boundary_discontinuity, 5) if loop_boundary_discontinuity is not None else None,
                sample_rate=target_sr,
                output_formats=formats,
                quality_measurements=quality_measurements,
                stems=stems_manifest_list,
                stem_reconstruction=reconstruction_measurements,
                ducking=ducking_manifest_data,
                processing_time_s=total_processing_s,
                estimated_cost_usd=gen_result.estimated_cost_usd,
                warnings=gen_result.warnings,
                known_limitations=known_limitations,
                rights_and_license=rights_info,
            )

            manifest_json_str = manifest.model_dump_json(indent=2)
            await storage.save_bytes(
                job_id, "music_manifest.json", manifest_json_str.encode("utf-8"), content_type="application/json"
            )
            assets_dict["manifest"] = f"/api/v1/jobs/{job_id}/manifest"

            db.update_job_progress(
                job_id,
                status=JobStatus.SUCCEEDED,
                progress=100,
                assets=assets_dict,
                manifest=manifest.model_dump(),
            )

    except IncubrixError as e:
        error_dict = {
            "code": e.code,
            "message": e.message,
            "retryable": e.retryable,
            "details": e.details,
        }
        db.update_job_progress(job_id, status=JobStatus.FAILED, progress=0, error=error_dict)
    except Exception as e:
        error_dict = {
            "code": ErrorCode.WORKER_FAILURE,
            "message": f"Unexpected error during job execution: {str(e)}",
            "retryable": True,
            "details": {"exception": type(e).__name__},
        }
        db.update_job_progress(job_id, status=JobStatus.FAILED, progress=0, error=error_dict)
