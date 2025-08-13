from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Literal, List
from backend.utils.constants import DATA_PATH
from backend.core.dataset_service import get_full_dataset

from backend.core.feature_service import extract_base_feature_from_address

router = APIRouter()

def _list_csv_files() -> List[str]:
    return sorted([p.name for p in DATA_PATH.glob("*.csv")])

# Build a Literal type from the files found at import time.
_csv_files = _list_csv_files()
ChoiceLiteral = Literal[tuple(_csv_files)] if _csv_files else Literal[""]

class DatasetRequest(BaseModel):
    filename: ChoiceLiteral  # type: ignore
    refresh: bool = True

@router.get("/dataset/files", summary="List available dataset CSV files")
def list_dataset_files():
    return {"files": _list_csv_files()}

@router.post("/dataset", summary="Get dataset features")
def get_dataset(request: DatasetRequest = Depends()):
    dataset = get_full_dataset(request.filename, request.refresh)
    return dataset

@router.get("/extract/{address}", summary="Extract features from a contract")
def extract_features(address: str):
    return extract_base_feature_from_address(address)
