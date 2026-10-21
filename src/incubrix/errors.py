"""
Standardized error handling and structured error definitions for IncuBrix AI Background Music Engine.
All errors follow the mandatory uppercase snake-case error codes and structured JSON response format.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_SECTION_TIMESTAMPS = "INVALID_SECTION_TIMESTAMPS"
    MISSING_IDEMPOTENCY_KEY = "MISSING_IDEMPOTENCY_KEY"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    MODEL_PROVIDER_UNAVAILABLE = "MODEL_PROVIDER_UNAVAILABLE"
    MISSING_MODEL_CREDENTIALS = "MISSING_MODEL_CREDENTIALS"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    STEM_SEPARATION_FAILED = "STEM_SEPARATION_FAILED"
    AUDIO_PROCESSING_ERROR = "AUDIO_PROCESSING_ERROR"
    STORAGE_FAILURE = "STORAGE_FAILURE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    SSRF_ATTEMPT_BLOCKED = "SSRF_ATTEMPT_BLOCKED"
    NARRATION_FETCH_FAILED = "NARRATION_FETCH_FAILED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    WORKER_FAILURE = "WORKER_FAILURE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class IncubrixError(Exception):
    """Base exception for all structured IncuBrix errors."""
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


def make_error_response(
    code: str,
    message: str,
    status_code: int = 400,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Helper to create a standard structured error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": details or {},
            }
        },
    )


async def incubrix_exception_handler(request: Request, exc: IncubrixError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
