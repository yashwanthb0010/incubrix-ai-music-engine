"""
API Route definitions for IncuBrix AI Background Music Engine.
All application endpoints are served under /api/v1/ except /health and /ready.
"""

from __future__ import annotations
import io
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, Header, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from incubrix.config import get_settings
from incubrix.schemas import (
    CreateMusicJobRequest,
    JobAcceptedResponse,
    JobStatusResponse,
    JobStatus,
)
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
    if "vocal" in p_lower or "singing" in p_lower or "lyrics" in p_lower:
        initial_warnings.append("Request asked for vocals. This engine is strictly instrumental; vocals were suppressed.")
    copyrighted_refs = ["taylor swift", "beatles", "drake", "queen", "michael jackson", "ed sheeran"]
    for cr in copyrighted_refs:
        if cr in p_lower:
            initial_warnings.append(
                f"Prompt referenced '{cr}'. To respect copyright, an original composition is generated instead."
            )

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
