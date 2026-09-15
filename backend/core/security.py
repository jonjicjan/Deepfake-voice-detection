"""VoiceShield AI — API authentication, rate limiting, and security headers."""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings

# Paths that never require authentication
PUBLIC_PATHS = {"/health", "/"}

# Admin-only API paths (require admin key when auth is enabled)
ADMIN_PATHS = {
    "/api/calibrate",
    "/api/benchmark",
    "/api/benchmark/models",
    "/api/policy",
    "/api/enroll-speaker",
    "/api/performance",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "microphone=(self), camera=()"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = 120):
        super().__init__(app)
        self.limit = limit_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        settings = get_settings()
        now = time.time()
        ip = self._client_ip(request)
        window = self._hits[ip]
        window[:] = [t for t in window if now - t < 60]

        if len(window) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
            )

        window.append(now)
        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        settings = get_settings()
        path = request.url.path

        if path in PUBLIC_PATHS or not path.startswith("/api/"):
            return await call_next(request)

        if not settings.auth_enabled:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "").strip()
        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required. Provide X-API-Key header."},
            )

        is_admin_route = (
            path in ADMIN_PATHS
            or (path == "/api/policy" and request.method == "PUT")
        )

        if is_admin_route:
            if provided != settings.admin_key:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Administrator API key required."},
                )
        elif provided != settings.api_key and provided != settings.admin_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)


def verify_ws_api_key(api_key: str | None) -> bool:
    """Return True if WebSocket connection is authorized."""
    settings = get_settings()
    if not settings.auth_enabled:
        return True
    return bool(api_key and api_key in (settings.api_key, settings.admin_key))
