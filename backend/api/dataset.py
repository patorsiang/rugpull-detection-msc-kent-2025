from fastapi import APIRouter, Query
from backend.core.dataset_service import get_full_dataset
from typing import List, Optional
from backend.utils.constants import UNLABELED_PATH
router = APIRouter()

@router.get("/dataset", summary="Get dataset features")
def get_dataset():
    dataset = get_full_dataset()
    return dataset


@router.post("/dataset/refresh", summary="Refresh dataset features")
def refresh_dataset(
    addresses: Optional[List[str]] = Query(default=None)
):
    dataset = get_full_dataset(refresh=True, addresses=addresses)
    return dataset

@router.get("/dataset-unlabeled", summary="Get dataset features")
def get_dataset():
    dataset = get_full_dataset(UNLABELED_PATH / 'groundtruth.csv')
    return dataset
