from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.core.training.training_service import train_and_save_best_model, full_training

router = APIRouter()

class TuningTraining(BaseModel):
    test_size: Optional[float] = 0.2
    N_TRIALS: Optional[int] = 50

@router.post("/tuning-on-training", summary="Select Model and Train Model")
def tuning(params: TuningTraining):
    result = train_and_save_best_model(**params.dict())
    return {"status": "success", "metrics": result}

@router.post("/full-training", summary="Full Train Model")
def training():
    result = full_training()
    return {"status": "success", "metrics": result}
