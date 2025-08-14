from __future__ import annotations
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.responses import Response

from backend.utils.logger import logging

log = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request id to request.state and response header."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request, call_next):
        rid = str(uuid.uuid4())
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers[self.header_name] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Cheap access log with latency and status."""

    async def dispatch(self, request, call_next):
        rid = getattr(request.state, "request_id", "-")
        start = time.perf_counter()
        response: Response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            f"{rid} {request.method} {request.url.path} -> {response.status_code} in {dur_ms:.1f}ms"
        )
        return response
