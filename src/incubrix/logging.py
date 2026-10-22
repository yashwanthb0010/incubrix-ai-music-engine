"""
Structured JSON Logging for IncuBrix AI Background Music Engine.
Formats all log output as structured JSON containing:
timestamp, level, correlation_id, job_id, logger name, and message.
"""

from __future__ import annotations
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Context variables for request tracing
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id_ctx", default=None)
job_id_ctx: ContextVar[Optional[str]] = ContextVar("job_id_ctx", default=None)


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include tracing IDs if present in context or record
        corr_id = getattr(record, "correlation_id", None) or correlation_id_ctx.get()
        if corr_id:
            log_entry["correlation_id"] = corr_id

        j_id = getattr(record, "job_id", None) or job_id_ctx.get()
        if j_id:
            log_entry["job_id"] = j_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_structured_logging(level: int = logging.INFO) -> None:
    """Configures root and app loggers to use structured JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(stream_handler)


def get_logger(name: str = "incubrix") -> logging.Logger:
    return logging.getLogger(name)
