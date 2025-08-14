from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel, field_validator
from backend.utils.constants import DATA_PATH
from backend.core.dataset_service import get_full_dataset
from backend.utils.etherscan_quota import QuotaExceeded

router = APIRouter()


# ---------- helpers ----------

def _list_csv_files() -> List[str]:
    return sorted([p.name for p in DATA_PATH.glob("*.csv")])


# ---------- query model (kept as query params for Swagger dropdowns) ----------

class DatasetQuery(BaseModel):
    filename: str
    refresh: bool = True
    limit: Optional[int] = None
    offset: Optional[int] = None

    @field_validator("filename")
    @classmethod
    def must_exist(cls, v: str) -> str:
        files = _list_csv_files()
        if v not in files:
            raise ValueError(f"filename must be one of: {files}")
        return v


# ---------- endpoints ----------

@router.get("/dataset/files", summary="List available dataset CSV files")
def list_dataset_files():
    return {"files": _list_csv_files()}


@router.post(
    "/dataset",
    summary="Get dataset features (cached) for all/selected addresses",
    description=(
        "Query params: filename, refresh, limit, offset. "
        "Request body: optional JSON array of addresses. "
        "If addresses is omitted/empty, processes all rows (then applies offset/limit)."
    ),
)
def get_dataset(
    req: DatasetQuery = Depends(),
    # NOTE: embed=False makes the body be a plain JSON array (not an object wrapper)
    addresses: Optional[List[str]] = Body(default=None, embed=False, example=[
        "0x93023f1d3525e273f291b6f76d2f5027a39bf302",
        "0x2753dce37a7edb052a77832039bcc9aa49ad8b25",
    ]),
):
    try:
        dataset = get_full_dataset(
            filename=req.filename,
            refresh=req.refresh,
            addresses=addresses,
            limit=req.limit,
            offset=req.offset,
        )
        return dataset
    except QuotaExceeded as e:
        # Consistent shape with other endpoints
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
