from fastapi import APIRouter
from backend.core.meta_service import MetaService

router = APIRouter()

@router.get("/versions", summary="Get current model status and backups")
def get_versions():
    return {
        "status": "Rug Pull Detection API is running.",
        **MetaService.get_status(),
    }
