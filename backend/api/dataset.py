from fastapi import APIRouter, Depends
from typing import Literal
from pydantic import BaseModel
from enum import Enum

from backend.utils.constants import DATA_PATH
from backend.core.dataset_service import get_full_dataset

router = APIRouter()

# Get CSV files at startup for dropdown
csv_files = [file.name for file in DATA_PATH.glob("*.csv")]

ChoiceLiteral = Literal[tuple(csv_files)] if csv_files else Literal[""]


class DatasetRequest(BaseModel):
    filename: ChoiceLiteral  # type: ignore # Dropdown in Swagger UI
    refresh: bool = True     # Default value

@router.post("/dataset", summary="Get dataset features")
def get_dataset(request: DatasetRequest = Depends()):
    dataset = get_full_dataset(request.filename, request.refresh)
    return dataset
