# backend/api/expand_labels.py
from __future__ import annotations
from typing import Optional, Union, Dict, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ConfigDict

from backend.api.dataset import _list_csv_files
from backend.core.expand_labels_service import (
    ExpandLabelsService,
    SelfLearningExpandConfig,
)

router = APIRouter()
_service = ExpandLabelsService()

LowHigh = Union[float, Dict[str, float]]

def _must_exist(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    files = _list_csv_files()
    if v not in files:
        raise ValueError(f"filename must be one of: {files}")
    return v

def _enum_files() -> List[str]:
    try:
        return _list_csv_files()
    except Exception:
        return []

_FILES_ENUM = _enum_files()


class ExpandLabelsReq(BaseModel):
    # allow using either field name or alias in requests
    model_config = ConfigDict(populate_by_name=True)

    # NOTE: json_schema_extra "enum" -> Swagger shows dropdown
    filename: str = Field(
        ...,
        description="CSV under data/ containing Address column and any label columns",
        json_schema_extra={"enum": _FILES_ENUM},
    )

    round_name: str = Field(..., description="Round tag, e.g. r01 or 20250815")

    # thresholds: allow float or {label: float}
    low: LowHigh = Field(0.10, description="prob ≤ low → 0")
    high: LowHigh = Field(0.90, description="prob ≥ high → 1")

    # chunking for large files
    chunk_size: Optional[int] = Field(
        None,
        ge=1,
        description="If set, process file in chunks of this many rows",
    )
    chunk_index: int = Field(
        0,
        ge=0,
        description="0-based index of the chunk to process (ignored if chunk_size is None)",
    )
    save_confident_only: bool = Field(True, description="Emit a *_confident.csv where all system labels are not NaN")


@router.post(
    "/self-learning/expand-label",
    summary="Fulfil missing system labels via prediction and expand with any new labels found in the file (supports chunking)",
)
def expand_labels(req: ExpandLabelsReq = Depends()):
    # validate filename (like your training endpoints)
    fname = _must_exist(req.filename)

    # NOTE: do NOT pass 'overwrite' (dataclass no longer has this field)
    cfg = SelfLearningExpandConfig(
        filename=fname,
        round_name=req.round_name,
        low=req.low,
        high=req.high,
        chunk_size=req.chunk_size,
        chunk_index=req.chunk_index,
        save_confident_only=req.save_confident_only

    )
    return _service.run(cfg)
