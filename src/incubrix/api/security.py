"""
Security utilities: SSRF protection, Idempotency hashing, and rate limiting.
"""

from __future__ import annotations
import hashlib
import ipaddress
import json
import socket
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import Request
from incubrix.config import Settings, get_settings
from incubrix.errors import IncubrixError, ErrorCode


def compute_request_hash(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of normalized request dictionary."""
    normalized_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def validate_url_against_ssrf(url: str, enabled: bool = True) -> None:
    """
    Checks URL against SSRF vulnerabilities:
    Blocks localhost, private IP ranges (10.x, 192.168.x, 172.16-31.x),
    link-local (169.254.x), loopback (127.x), and IPv6 private/loopback.
    """
    if not enabled or not url:
        return

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise IncubrixError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.",
                status_code=400,
            )

        hostname = parsed.hostname
        if not hostname:
            raise IncubrixError(
                code=ErrorCode.VALIDATION_ERROR,
                message="Invalid URL: Missing hostname.",
                status_code=400,
            )

        # Block literal localhost names
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
            raise IncubrixError(
                code=ErrorCode.SSRF_ATTEMPT_BLOCKED,
                message="Access to local or internal network hostnames is prohibited (SSRF prevention).",
                status_code=400,
            )

        # Resolve IP to detect internal routing
        try:
            ip_str = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip_str)

            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                raise IncubrixError(
                    code=ErrorCode.SSRF_ATTEMPT_BLOCKED,
                    message=f"Access to private or reserved IP address ({ip_str}) is prohibited.",
                    status_code=400,
                    details={"hostname": hostname, "resolved_ip": ip_str},
                )
        except socket.gaierror:
            # Cannot resolve hostname
            raise IncubrixError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Could not resolve narration URL hostname: '{hostname}'",
                status_code=400,
            )

    except IncubrixError:
        raise
    except Exception as e:
        raise IncubrixError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid narration URL: {str(e)}",
            status_code=400,
        )


class RateLimiter:
    """Simple in-memory token bucket rate limiter per client IP."""
    def __init__(self, limit_per_minute: int = 60):
        self.limit = limit_per_minute
        self.requests: Dict[str, list[float]] = {}

    def check_rate_limit(self, client_ip: str) -> None:
        import time
        now = time.time()
        window = 60.0

        if client_ip not in self.requests:
            self.requests[client_ip] = [now]
            return

        # Keep timestamps within window
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < window]

        if len(self.requests[client_ip]) >= self.limit:
            raise IncubrixError(
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=f"Rate limit exceeded. Maximum {self.limit} requests per minute.",
                status_code=429,
                retryable=True,
            )

        self.requests[client_ip].append(now)


_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter
