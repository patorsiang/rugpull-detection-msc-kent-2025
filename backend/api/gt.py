from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from backend.api.dataset import _list_csv_files
from backend.core.gt_service import GTService
from backend.utils.etherscan_quota import QuotaExceeded

router = APIRouter()
_service = GTService()


class PromoteReq(BaseModel):
    new_source: str = Field(..., description="Existing CSV in DATA_PATH to promote")
    output_name: Optional[str] = Field(
        None, description="Optional filename to write as new ground truth; default is timestamped"
    )
    retrain: bool = Field(True, description="Run training-pipeline after promotion")
    eval_source: Optional[str] = Field(
        None, description="Stable eval split; defaults to existing canonical groundtruth.csv"
    )
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    N_TRIALS: int = Field(50, ge=1, le=2000)


def _must_exist(name: Optional[str], files: List[str], field: str) -> Optional[str]:
    if name is None:
        return None
    if name not in files:
        raise HTTPException(status_code=400, detail=f"{field} must be one of: {files}")
    return name


@router.post("/gt/promote", summary="Promote a dataset to ground truth (optional auto-retrain)")
def promote_gt(req: PromoteReq = Depends()):
    files = _list_csv_files()
    req.new_source = _must_exist(req.new_source, files, "new_source")  # type: ignore
    req.eval_source = _must_exist(req.eval_source, files, "eval_source")  # type: ignore

    try:
        result = _service.promote(
            new_source=req.new_source,
            output_name=req.output_name,
            retrain=req.retrain,
            eval_source=req.eval_source,
            test_size=req.test_size,
            n_trials=req.N_TRIALS,
        )
        return result
    except QuotaExceeded as e:
        # Should already be handled in service, but keep a defensive catch.
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
