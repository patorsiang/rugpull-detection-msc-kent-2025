from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from backend.core.training_service import train_and_save_best_model, full_training, train_pipeline
from backend.api.dataset import ChoiceLiteral

router = APIRouter()

class TuningTraining(BaseModel):
    test_size: Optional[float] = 0.2
    N_TRIALS: Optional[int] = 50
    source: ChoiceLiteral # type: ignore

@router.post("/tuning-on-training", summary="Select Model and Train Model")
def tuning(requests: TuningTraining = Depends()):
    result = train_and_save_best_model(**requests.dict())
    return {"status": "success", "metrics": result}

class FullTraining(BaseModel):
    source: ChoiceLiteral # type: ignore

@router.post("/finalized-training", summary="Full Train Model")
def training(requests: FullTraining = Depends()):
    result = full_training(requests.source)
    return {"status": "success", "metrics": result}

@router.post("/training-pipeline", summary="Full Train Model")
def training_pipe(requests: TuningTraining = Depends()):
    result = train_pipeline(**requests.dict())
    return {"status": "success", "metrics": result}
