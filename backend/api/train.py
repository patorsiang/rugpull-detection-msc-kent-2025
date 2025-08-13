from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from backend.api.dataset import ChoiceLiteral
from backend.core.training_service import TrainingPipeline

router = APIRouter()

# ---------- Schemas ----------

class TuningTraining(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.9, description="Train/test split fraction")
    N_TRIALS: int = Field(50, ge=1, le=2000, description="Optuna trials")
    source: ChoiceLiteral  # type: ignore

class FullTraining(BaseModel):
    # Allow overriding trials (used by pipeline internals if needed)
    N_TRIALS: int = Field(50, ge=1, le=2000, description="Optuna trials for fusion re-optimization")
    source: ChoiceLiteral  # type: ignore

class PipelineTraining(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=0.9, description="Train/test split fraction")
    N_TRIALS: int = Field(50, ge=1, le=2000, description="Optuna trials")
    source: ChoiceLiteral  # type: ignore
    eval_source: ChoiceLiteral  # type: ignore

# ---------- Endpoints ----------

@router.post("/tuning-on-training", summary="Select Model and Train Model (Optuna on split)")
def tuning(requests: TuningTraining = Depends()):
    """
    Runs the tuning stage (Optuna on the split) and saves the current models.
    Does NOT retrain on 100% unless it's the very first run in your pipeline.
    """
    pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
    meta = pipeline.trainer.train_and_save(test_size=requests.test_size, source=requests.source)
    return {"status": "success", "metrics": meta}

@router.post("/finalized-training", summary="Full Train Model (retrain on 100%)")
def training(requests: FullTraining= Depends()):
    """
    Retrains saved models on 100% of the data and re-optimizes fusion.
    """
    pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
    meta = pipeline.full.run(source=requests.source)
    return {"status": "success", "metrics": meta}

@router.post("/training-pipeline", summary="Evaluate baseline vs trial, retrain if improved")
def training_pipe(requests: PipelineTraining = Depends()):
    """
    1) Evaluate current models on eval_source split (frozen).
    2) Evaluate refit-on-split trial on source.
    3) If trial >= baseline, run tuning + full training; else keep current.
    """
    pipeline = TrainingPipeline(n_trials=requests.N_TRIALS)
    meta = pipeline.run(
        source=requests.source,
        eval_source=requests.eval_source,
        test_size=requests.test_size,
    )
    return {"status": "success", "metrics": meta}
