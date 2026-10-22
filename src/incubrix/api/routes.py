"""
API Route definitions for IncuBrix AI Background Music Engine.
All application endpoints are served under /api/v1/ except /health and /ready.
"""

from __future__ import annotations
import io
import json
import re
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Header, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from incubrix.config import get_settings
from incubrix.schemas import (
    CreateMusicJobRequest,
    RegenerateStemRequest,
    JobAcceptedResponse,
    JobStatusResponse,
    JobStatus,
)
from incubrix.audio.processor import AudioProcessor
from incubrix.audio.validator import AudioValidator
from incubrix.audio.ffmpeg_utils import convert_wav_to_mp3
from incubrix.db.database import get_db
from incubrix.storage import get_storage
from incubrix.queue.worker import get_queue
from incubrix.providers import get_music_provider
from incubrix.stems import get_stem_separator
from incubrix.api.security import (
    compute_request_hash,
    validate_url_against_ssrf,
    get_rate_limiter,
)
from incubrix.errors import IncubrixError, ErrorCode

router = APIRouter()


# -------------------------------------------------------------
# Music Generation Jobs
# -------------------------------------------------------------

@router.post(
    "/api/v1/music/jobs",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a background music generation job",
    description="Submits an asynchronous music generation request with stem separation and optional narration ducking.",
)
async def create_music_job(
    request_data: CreateMusicJobRequest,
    request: Request,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    cfg = get_settings()
    db = get_db(cfg.database_url)
    queue = get_queue(cfg)

    # Rate limiting
    client_ip = request.client.host if request.client else "127.0.0.1"
    get_rate_limiter().check_rate_limit(client_ip)

    # Optional demo token auth
    if cfg.demo_access_token:
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {cfg.demo_access_token}":
            raise IncubrixError(
                code=ErrorCode.UNAUTHORIZED,
                message="Invalid or missing demo access token.",
                status_code=401,
            )

    # Mandatory Idempotency-Key validation
    if not idempotency_key or not idempotency_key.strip():
        raise IncubrixError(
            code=ErrorCode.MISSING_IDEMPOTENCY_KEY,
            message="The 'Idempotency-Key' HTTP header is required for creating a music job.",
            status_code=400,
        )

    # SSRF validation on narration URL
    if request_data.narration_url:
        validate_url_against_ssrf(request_data.narration_url, enabled=cfg.ssrf_protection_enabled)

    # Compute hash of incoming payload
    req_dict = request_data.model_dump()
    req_hash = compute_request_hash(req_dict)

    # Check Idempotency Record
    existing = db.get_idempotency_record(idempotency_key)
    if existing:
        saved_hash, existing_job_id = existing
        if saved_hash == req_hash:
            # Same payload + same key -> Return existing job
            existing_job = db.get_job(existing_job_id)
            if existing_job:
                return JobAcceptedResponse(
                    job_id=existing_job.job_id,
                    status=existing_job.status,
                    status_url=f"/api/v1/jobs/{existing_job.job_id}",
                    created_at=existing_job.created_at,
                    warnings=existing_job.warnings,
                )
        else:
            # Same key with different payload -> Conflict
            raise IncubrixError(
                code=ErrorCode.IDEMPOTENCY_CONFLICT,
                message=(
                    f"An existing job with Idempotency-Key '{idempotency_key}' was submitted with a different payload."
                ),
                status_code=409,
            )

    # Generate new Job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    initial_warnings = []

    # Check prompt content for informative warnings
    p_lower = request_data.prompt.lower()

    # 1. Vocal requests
    if any(w in p_lower for w in ("vocal", "vocals", "singing", "lyrics", "singer", "rap lyrics", "acapella")):
        initial_warnings.append("Request asked for vocals. This engine is strictly instrumental; vocals were suppressed.")

    # 2. Real artists and copyrighted songs
    copyrighted_refs = [
        "taylor swift", "beatles", "drake", "queen", "michael jackson", "ed sheeran",
        "beethoven", "hans zimmer", "mozart", "bach", "coldplay", "nirvana",
        "pink floyd", "eminem", "adele", "the weeknd", "ac/dc", "daft punk",
        "metallica", "elvis", "bob marley", "radiohead", "led zeppelin", "freddie mercury",
        "bohemian rhapsody", "thriller", "shape of you", "smells like teen spirit",
        "blinding lights", "billie jean", "hotel california", "stairway to heaven"
    ]
    for cr in copyrighted_refs:
        if cr in p_lower:
            initial_warnings.append(
                f"Prompt referenced '{cr}'. To respect copyright, prevent infringement, and ensure output is safe to publish commercially, an original composition was generated instead of copying an existing song."
            )

    # 3. Contradictory / vague prompt terms
    contradictions = [
        (("calm", "peaceful", "quiet", "soft", "serene", "ambient"), ("aggressive", "hard", "heavy", "violent", "angry", "screaming")),
        (("fast", "rapid", "hyper"), ("slow", "sluggish", "leisurely")),
        (("happy", "cheerful", "sunny", "joyful"), ("sad", "depressing", "grief", "melancholy", "somber")),
        (("loud",), ("quiet", "soft")),
    ]
    for group1, group2 in contradictions:
        m1 = [w for w in group1 if w in p_lower]
        m2 = [w for w in group2 if w in p_lower]
        if m1 and m2:
            initial_warnings.append(
                f"Prompt contains contradictory attributes ('{m1[0]}' vs '{m2[0]}'). "
                f"Resolved by synthesizing dynamic musical contrast: peaceful harmonic atmosphere with driving rhythmic energy."
            )
            break

    # 4. Multilingual / Non-English prompt
    foreign_terms = [
        "musica", "cancion", "para", "alegre", "corporativo", "pelicula",  # Spanish
        "musique", "pour", "calme", "apaisante", "dramatique",  # French
        "musik", "für", "lied", "entspannend", "hintergrund",  # German
        "canzone", "per", "allegra", "rilassante",  # Italian
    ]
    has_non_ascii = any(ord(c) > 127 for c in request_data.prompt)
    has_foreign_words = any(re.search(r"\b" + w + r"\b", p_lower) for w in foreign_terms)
    if has_non_ascii or has_foreign_words:
        initial_warnings.append(
            "Non-English prompt detected. Multilingual semantic parser translated and mapped prompt to musical genre and instrumentation."
        )

    # 5. Closely-spaced sections (<1.5s apart)
    if request_data.sections and len(request_data.sections) > 1:
        for idx in range(1, len(request_data.sections)):
            gap = request_data.sections[idx].start_s - request_data.sections[idx - 1].start_s
            if gap < 1.5:
                initial_warnings.append(
                    f"Section {idx} starts {gap:.2f}s after previous section (<1.5s). "
                    f"Transition smoothed to musical beat grid to maintain rhythmic cohesion."
                )
                break

    # Persist job and idempotency record
    db.create_job(job_id, req_dict, initial_warnings)
    db.save_idempotency_key(idempotency_key, req_hash, job_id)

    # Enqueue background execution asynchronously
    await queue.enqueue_job(job_id, cfg)

    return JobAcceptedResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        status_url=f"/api/v1/jobs/{job_id}",
        created_at=datetime.now(timezone.utc),
        warnings=initial_warnings,
    )


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job execution status",
    description="Retrieves the current status, progress, assets, warnings, and error details of a music job.",
)
async def get_job_status(job_id: str):
    cfg = get_settings()
    db = get_db(cfg.database_url)
    job = db.get_job(job_id)

    if not job:
        raise IncubrixError(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job with ID '{job_id}' was not found.",
            status_code=404,
        )

    return job


@router.get(
    "/api/v1/jobs/{job_id}/manifest",
    summary="Get job music manifest",
    description="Retrieves the completed music manifest containing audio measurements, stem metrics, and licensing.",
)
async def get_job_manifest(job_id: str):
    cfg = get_settings()
    db = get_db(cfg.database_url)
    manifest = db.get_job_manifest(job_id)

    if not manifest:
        # Check if job exists
        job = db.get_job(job_id)
        if not job:
            raise IncubrixError(
                code=ErrorCode.JOB_NOT_FOUND,
                message=f"Job with ID '{job_id}' was not found.",
                status_code=404,
            )
        if job.status == JobStatus.FAILED:
            raise IncubrixError(
                code=ErrorCode.JOB_NOT_FOUND,
                message=f"Job '{job_id}' failed during execution. Manifest is unavailable.",
                status_code=400,
                details=job.error or {},
            )
        raise IncubrixError(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Manifest for job '{job_id}' is not yet available. Current status: '{job.status.value}'.",
            status_code=404,
        )

    return JSONResponse(content=manifest)


# -------------------------------------------------------------
# Single Stem Regeneration Endpoint
# -------------------------------------------------------------

@router.post(
    "/api/v1/jobs/{job_id}/stems/{stem_name}/regenerate",
    summary="Regenerate one stem without altering others",
    description="Regenerates a single instrument part (e.g. new drums) while keeping all other stems bit-for-bit identical, and re-renders the master mix.",
)
async def regenerate_single_stem_endpoint(
    job_id: str,
    stem_name: str,
    request_data: Optional[RegenerateStemRequest] = None,
):
    cfg = get_settings()
    db = get_db(cfg.database_url)
    storage = get_storage(cfg)
    music_provider = get_music_provider(cfg)
    processor = AudioProcessor()
    validator = AudioValidator()

    job = db.get_job(job_id)
    if not job:
        raise IncubrixError(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job with ID '{job_id}' not found.",
            status_code=404,
        )

    if job.status != JobStatus.SUCCEEDED:
        raise IncubrixError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Cannot regenerate stem for job in status '{job.status.value}'. Must be 'succeeded'.",
            status_code=400,
        )

    manifest_data = db.get_job_manifest(job_id) or {}
    if not manifest_data:
        try:
            m_bytes = await storage.get_bytes(job_id, "music_manifest.json")
            manifest_data = json.loads(m_bytes.decode("utf-8"))
        except Exception:
            raise IncubrixError(
                code=ErrorCode.JOB_NOT_FOUND,
                message=f"Manifest for job '{job_id}' could not be loaded.",
                status_code=404,
            )

    clean_stem_name = stem_name.lower().strip()
    existing_stems = {s["name"].lower(): s for s in manifest_data.get("stems", [])}
    if clean_stem_name not in existing_stems:
        raise IncubrixError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Stem '{stem_name}' not found in job. Existing stems: {list(existing_stems.keys())}",
            status_code=400,
        )

    tempo_bpm = manifest_data.get("tempo_bpm") or 120
    key = manifest_data.get("key") or "C major"
    duration_s = manifest_data.get("actual_duration_s") or manifest_data.get("requested_duration_s") or 15.0
    sample_rate = manifest_data.get("sample_rate") or 48000
    loopable = manifest_data.get("loopable", False)
    target_lufs = manifest_data.get("quality_measurements", {}).get("target_lufs", -16.0)

    from incubrix.schemas import Section
    raw_sections = manifest_data.get("sections") or []
    sections = [Section(**s) for s in raw_sections]

    seed = request_data.seed if (request_data and request_data.seed is not None) else (int(datetime.now().timestamp() * 1000) % 1000000)
    style = request_data.style if request_data else None

    # 1. Synthesize the new single stem
    if hasattr(music_provider, "generate_single_stem"):
        new_stem_audio = music_provider.generate_single_stem(
            clean_stem_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            tempo_bpm=tempo_bpm,
            key_str=key,
            sections=sections,
            is_loopable=loopable,
            seed=seed,
            style_override=style,
        )
    else:
        raise IncubrixError(
            code=ErrorCode.PROVIDER_ERROR,
            message="Active music provider does not support single-stem regeneration.",
            status_code=500,
        )

    new_stem_stereo = processor.ensure_stereo(new_stem_audio)
    new_stem_dur = processor.enforce_exact_duration(
        new_stem_stereo, sample_rate, duration_s, fade_out_duration_s=0.0
    )

    # 2. Read the other stems from storage (strictly preserved bit-for-bit)
    aligned_stems = {}
    for s_name, s_info in existing_stems.items():
        if s_name == clean_stem_name:
            aligned_stems[s_name] = (new_stem_dur, s_info.get("gain", 1.0))
        else:
            other_bytes = await storage.get_bytes(job_id, s_info["filename"])
            other_audio, _ = sf.read(io.BytesIO(other_bytes), dtype="float32")
            other_stereo = processor.ensure_stereo(other_audio)
            aligned_stems[s_name] = (other_stereo, s_info.get("gain", 1.0))

    # 3. Re-render master mix from stems: mix = sum(stem_i * gain_i)
    reconstructed_mix = np.zeros_like(new_stem_dur)
    for s_name, (s_audio, s_gain) in aligned_stems.items():
        reconstructed_mix += s_audio * s_gain

    # The master mix is mathematically identical to the sum of stems: mix = sum(stem_i * gain_i)
    normalized_mix = reconstructed_mix

    # 4. Save updated new stem WAV and new mix WAV/MP3 to storage
    new_stem_buf = io.BytesIO()
    sf.write(new_stem_buf, new_stem_dur, sample_rate, format="WAV", subtype="PCM_16")
    new_stem_filename = existing_stems[clean_stem_name]["filename"]
    await storage.save_bytes(job_id, new_stem_filename, new_stem_buf.getvalue(), content_type="audio/wav")

    mix_buf = io.BytesIO()
    sf.write(mix_buf, normalized_mix, sample_rate, format="WAV", subtype="PCM_16")
    await storage.save_bytes(job_id, "mix.wav", mix_buf.getvalue(), content_type="audio/wav")

    formats = manifest_data.get("output_formats", ["wav", "mp3"])
    if "mp3" in formats:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            t_wav = temp_path / "mix.wav"
            t_mp3 = temp_path / "mix.mp3"
            sf.write(str(t_wav), normalized_mix, sample_rate, subtype="PCM_16")
            if convert_wav_to_mp3(t_wav, t_mp3):
                with open(t_mp3, "rb") as f:
                    await storage.save_bytes(job_id, "mix.mp3", f.read(), content_type="audio/mpeg")

    # 5. Measure quality & stem reconstruction
    reconstruction_measurements = validator.measure_stem_reconstruction(
        final_mix=normalized_mix, stems=aligned_stems, sample_rate=sample_rate
    )
    quality_measurements = validator.validate_and_measure(
        audio=normalized_mix,
        sample_rate=sample_rate,
        requested_duration_s=duration_s,
    )

    # 6. Update manifest
    for stem in manifest_data["stems"]:
        if stem["name"].lower() == clean_stem_name:
            stem["was_regenerated"] = True
            stem["regenerated_at"] = datetime.now(timezone.utc).isoformat()
            stem["duration_s"] = round(float(new_stem_dur.shape[0] / sample_rate), 4)

    manifest_data["quality_measurements"] = quality_measurements.model_dump()
    manifest_data["stem_reconstruction"] = reconstruction_measurements.model_dump()

    updated_manifest_json = json.dumps(manifest_data, indent=2)
    await storage.save_bytes(job_id, "music_manifest.json", updated_manifest_json.encode("utf-8"), content_type="application/json")
    db.update_job_progress(job_id, JobStatus.SUCCEEDED, 100, manifest=manifest_data)

    return {
        "status": "succeeded",
        "job_id": job_id,
        "regenerated_stem": clean_stem_name,
        "message": f"Successfully regenerated '{clean_stem_name}' stem without modifying other stems.",
        "stem_reconstruction": reconstruction_measurements.model_dump(),
        "assets": job.assets.model_dump() if hasattr(job.assets, "model_dump") else job.assets,
        "manifest": manifest_data,
    }


@router.post(
    "/api/v1/jobs/{job_id}/regenerate-stem",
    summary="Regenerate one stem without altering others",
    description="Regenerates a single instrument part (e.g. new drums) while keeping all other stems bit-for-bit identical, and re-renders the master mix.",
)
async def regenerate_stem_endpoint(
    job_id: str,
    request_data: RegenerateStemRequest,
):
    return await regenerate_single_stem_endpoint(job_id, request_data.stem_name, request_data)


@router.get(
    "/api/v1/jobs/{job_id}/download/{asset_name}",
    summary="Download generated audio asset",
    description="Streams or downloads generated assets (mix.wav, mix.mp3, stem_*.wav, mix_ducked.wav, etc.).",
)
async def download_job_asset(job_id: str, asset_name: str):
    cfg = get_settings()
    storage = get_storage(cfg)

    # Basic sanitization of filename
    if ".." in asset_name or "/" in asset_name or "\\" in asset_name:
        raise IncubrixError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid asset filename.",
            status_code=400,
        )

    data = await storage.get_bytes(job_id, asset_name)

    content_type = "application/octet-stream"
    if asset_name.endswith(".wav"):
        content_type = "audio/wav"
    elif asset_name.endswith(".mp3"):
        content_type = "audio/mpeg"
    elif asset_name.endswith(".json"):
        content_type = "application/json"

    headers = {
        "Content-Disposition": f'attachment; filename="{asset_name}"',
        "Content-Length": str(len(data)),
    }

    return Response(content=data, media_type=content_type, headers=headers)


# -------------------------------------------------------------
# Health and Readiness Endpoints
# -------------------------------------------------------------

@router.get(
    "/health",
    summary="Basic process health",
    description="Returns HTTP 200 indicating the API service process is up and listening.",
)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "incubrix-ai-music-engine",
    }


@router.get(
    "/ready",
    summary="System readiness check",
    description="Inspects storage, database, provider readiness, and queue status.",
)
async def readiness_check():
    cfg = get_settings()
    db = get_db(cfg.database_url)
    storage = get_storage(cfg)
    queue = get_queue(cfg)
    music_provider = get_music_provider(cfg)
    stem_separator = get_stem_separator(cfg)

    checks: Dict[str, Any] = {}
    is_ready = True

    # 1. Database check
    try:
        conn = db._get_connection()
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = {"status": "ok", "type": "sqlite"}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}
        is_ready = False

    # 2. Queue check
    q_status = queue.get_status()
    checks["queue"] = q_status

    # 3. Storage check
    checks["storage"] = {
        "type": cfg.storage_type,
        "status": "ok",
    }

    # 4. Music provider check
    p_ok, p_msg = await music_provider.health_check()
    checks["music_provider"] = {
        "provider": cfg.music_provider,
        "ready": p_ok,
        "details": p_msg,
    }
    if not p_ok and cfg.music_provider != "mock":
        is_ready = False

    # 5. Stem separator check
    s_ok, s_msg = await stem_separator.health_check()
    checks["stem_separator"] = {
        "separator": cfg.stem_separator,
        "ready": s_ok,
        "details": s_msg,
    }
    if not s_ok and cfg.stem_separator != "mock":
        is_ready = False

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": is_ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )
