# backend/api/self_learning.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, Dict

from backend.api.dataset import _list_csv_files
from backend.core.self_learning_service import SelfLearningService, SelfLearningConfig

router = APIRouter()
_service = SelfLearningService()

LowHigh = Union[float, Dict[str, float]]

def _must_exist(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    files = _list_csv_files()
    if v not in files:
        raise ValueError(f"filename must be one of: {files}")
    return v

def _enum_files():
    # compute once at startup for the Swagger dropdown
    try:
        return _list_csv_files()
    except Exception:
        return []

_FILES_ENUM = _enum_files()


class SelfLearnReq(BaseModel):
    # allow using either field name or alias in requests
    model_config = ConfigDict(populate_by_name=True)

    # NOTE: json_schema_extra "enum" -> Swagger shows dropdown
    source: str = Field(
        ...,
        description="Ground truth CSV",
        json_schema_extra={"enum": _FILES_ENUM},
    )
    target: str = Field(
        ...,
        description="Target CSV to pseudo-label",
        json_schema_extra={"enum": _FILES_ENUM},
    )
    eval_source: Optional[str] = Field(
        None,
        description="Evaluation dataset (defaults to source)",
        json_schema_extra={"enum": _FILES_ENUM},
    )

    round_name: str = Field(..., description="Round tag, e.g. r01")

    # thresholds: allow float or {label: float}
    low: LowHigh = 0.10
    high: LowHigh = 0.90

    test_size: float = Field(0.2, ge=0.0, le=0.9)

    # final (heavy) training trials; expose as N_TRIALS in the UI
    n_trials: int = Field(50, ge=1, le=2000, alias="N_TRIALS", description="Final training trials when accepted")

    # fast preview trials (cheap accept/reject)
    preview_trials: int = Field(8, ge=1, le=500, description="Cheap preview search trials")

    # acceptance & caching
    accept_min_delta: float = Field(0.0, ge=0.0, le=1.0, description="Minimum F1 gain required to accept")
    cache_baseline: bool = True
    do_train: bool = True

    # chunking
    chunk_size: Optional[int] = Field(None, description="If set, process target in chunks of this size")
    chunk_index: int = Field(0, ge=0, description="0-based chunk index when chunk_size is used")


@router.post("/self-learning/run", summary="Pseudo-label target and merge into source if accuracy improves")
def self_learning_run(req: SelfLearnReq = Depends()):
    # validate filenames (like your training endpoints)
    src = _must_exist(req.source)
    tgt = _must_exist(req.target)
    evl = _must_exist(req.eval_source) if req.eval_source is not None else None

    cfg = SelfLearningConfig(
        source=src,
        target=tgt,
        round_name=req.round_name,
        low=req.low,
        high=req.high,
        eval_source=evl or src,
        test_size=req.test_size,
        n_trials=req.n_trials ,                # heavy trials (only if accepted & do_train)
        do_train=req.do_train,
        chunk_size=req.chunk_size,
        chunk_index=req.chunk_index,
        preview_trials=req.preview_trials,    # cheap trials for acceptance
        accept_min_delta=req.accept_min_delta,
        cache_baseline=req.cache_baseline,
    )
    return _service.run(cfg)
