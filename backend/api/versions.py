from fastapi import APIRouter
from backend.core.meta_service import get_status
router = APIRouter()

@router.get("/versions", summary="Get all model versions and current status")
def get_versions():
    status = get_status()

    return {
        "status": "Rug Pull Detection API is running.",
        **status,
    }
