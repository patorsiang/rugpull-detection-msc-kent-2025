from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from backend.core.predict_service import PredictService

router = APIRouter()

class PredictReq(BaseModel):
    addresses: List[str]

@router.post("/predict", summary="Predict labels")
def predict_endpoint(req: PredictReq):
    try:
        _predictor = PredictService()  # load models once at import
        result = _predictor.predict(req.addresses)
        return {"status": "success", "results": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
