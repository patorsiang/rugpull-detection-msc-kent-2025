from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.utils.logger import logging
from backend.utils.etherscan_quota import QuotaExceeded

log = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(QuotaExceeded)
    async def _quota_exceeded_handler(request: Request, exc: QuotaExceeded):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=429,
            content={"status": "quota_exhausted", "message": str(exc), "request_id": rid},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content={
                "status": "invalid_request",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": rid,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail, "request_id": rid},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        # log full stack, return sanitized message
        log.exception(f"Unhandled error (request_id={rid}): {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "request_id": rid},
        )
