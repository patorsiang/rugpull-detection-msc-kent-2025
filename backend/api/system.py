from fastapi import APIRouter
from backend.utils.etherscan_quota import quota_guard
from backend.utils.logger import logging
from backend.core.meta_service import MetaService

router = APIRouter()
log = logging.getLogger(__name__)

@router.get("/system/health", summary="Liveness/health")
def health():
    return {"status": "ok"}

@router.get("/system/quota", summary="Etherscan quota snapshot")
def quota():
    try:
        s = quota_guard.snapshot()
    except Exception as e:
        log.warning(f"quota snapshot failed: {e}")
        s = {"error": "unavailable"}
    return {"status": "ok", "provider": "etherscan", **s}

@router.get("/system/versions", summary="Get current model status and backups")
def get_versions():
    return {
        "status": "Rug Pull Detection API is running.",
        **MetaService.get_status(),
    }
