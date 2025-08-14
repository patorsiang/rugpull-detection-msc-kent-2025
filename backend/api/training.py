from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from backend.api.dataset import _list_csv_files
from backend.core.training_service import TrainingPipeline
from backend.utils.etherscan_quota import QuotaExceeded

router = APIRouter()

class TuningTraining(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    N_TRIALS: int = Field(50, ge=1, le=2000)
    source: str

class FullTraining(BaseModel):
    N_TRIALS: int = Field(50, ge=1, le=2000)
    source: str

class PipelineTraining(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    N_TRIALS: int = Field(50, ge=1, le=2000)
    source: str
    eval_source: str

def _must_exist(v: str) -> str:
    files = _list_csv_files()
    if v not in files:
        raise ValueError(f"filename must be one of: {files}")
    return v

@router.post("/tuning-on-training", summary="Select & train (Optuna) on a split")
def tuning(requests: TuningTraining = Depends()):
    try:
        requests.source = _must_exist(requests.source)
        pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
        meta = pipeline.trainer.train_and_save(test_size=requests.test_size, source=requests.source)
        return {"status": "success", "metrics": meta}
    except QuotaExceeded as e:
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/finalized-training", summary="Retrain all on 100% and re-optimize fusions")
def training(requests: FullTraining = Depends()):
    try:
        requests.source = _must_exist(requests.source)
        pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
        meta = pipeline.full.run(source=requests.source)
        return {"status": "success", "metrics": meta}
    except QuotaExceeded as e:
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/training-pipeline", summary="Compare baseline vs trial; retrain if trial ≥ baseline")
def training_pipe(requests: PipelineTraining = Depends()):
    try:
        requests.source = _must_exist(requests.source)
        requests.eval_source = _must_exist(requests.eval_source)
        pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
        meta = pipeline.run(source=requests.source, eval_source=requests.eval_source, test_size=requests.test_size)
        return {"status": "success", "metrics": meta}
    except QuotaExceeded as e:
        return {"status": "quota_exhausted", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
