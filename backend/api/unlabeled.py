from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.core.unlabeled_service import UnlabeledService
from backend.utils.etherscan_quota import QuotaExceeded

router = APIRouter()
_service = UnlabeledService()


class UnlabeledUpsertReq(BaseModel):
    addresses: List[str] = Field(..., description="Contract addresses to predict and upsert")
    filename: str = Field("unlabeled.csv", description="Target CSV in DATA_PATH")
    low: float = Field(0.10, ge=0.0, le=0.5)
    high: float = Field(0.90, ge=0.5, le=1.0)
    write_probs: bool = True
    probs_filename: Optional[str] = None


@router.post("/unlabeled/upsert", summary="Predict & upsert into unlabeled.csv (fill only -1s)")
def upsert_unlabeled(req: UnlabeledUpsertReq = Depends(), # NOTE: embed=False makes the body be a plain JSON array (not an object wrapper)
    addresses: Optional[List[str]] = Body(default=None, embed=False, example=[
        "0x93023f1d3525e273f291b6f76d2f5027a39bf302",
        "0x2753dce37a7edb052a77832039bcc9aa49ad8b25",
    ])):
    try:
        result = _service.upsert_predictions(
            addresses=addresses,
            filename=req.filename,
            low=req.low,
            high=req.high,
            write_probs=req.write_probs,
            probs_filename=req.probs_filename,
        )
        return result
    except QuotaExceeded as e:
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
