from fastapi import APIRouter
from backend.core.dataset_service import get_full_dataset

router = APIRouter()

@router.get("/dataset", summary="Get dataset features")
def get_dataset():
    dataset = get_full_dataset()
    return dataset

