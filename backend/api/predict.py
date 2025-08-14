from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from backend.core.predict_service import PredictService
from backend.utils.etherscan_quota import QuotaExceeded

router = APIRouter()


class PredictReq(BaseModel):
    addresses: List[str] = Field(..., description="Contract addresses to score")
    label_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Optional per-label thresholds in [0,1]. "
            "Only provide the labels you want to override; "
            "they will be merged with the model's saved thresholds."
        ),
        examples=[{"Mint": 0.65, "Leak": 0.55}],
    )
    anomaly_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional anomaly fusion threshold in [0,1]. Default is the saved model value.",
        examples=[0.6],
    )


@router.post("/predict", summary="Predict labels (with optional custom thresholds)")
def predict_endpoint(req: PredictReq):
    try:
        predictor = PredictService()  # load models once per import
        result = predictor.predict(
            req.addresses,
            label_thresholds=req.label_thresholds,
            anomaly_threshold=req.anomaly_threshold,
        )
        return {"status": "success", "results": result}
    except QuotaExceeded as e:
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
