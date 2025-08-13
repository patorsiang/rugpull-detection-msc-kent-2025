from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional

from backend.api.dataset import ChoiceLiteral
from backend.core.self_learning_service import SelfLearningService, SelfLearningConfig

router = APIRouter()
_service = SelfLearningService()  # allocate once like other services


class SelfLearnReq(BaseModel):
    source: ChoiceLiteral  # type: ignore
    target: ChoiceLiteral  # type: ignore
    round_name: str = Field(..., description="e.g. r01")
    low: float = Field(0.10, ge=0.0, le=0.5)
    high: float = Field(0.90, ge=0.5, le=1.0)
    eval_source: Optional[ChoiceLiteral] = None  # type: ignore
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    n_trials: int = Field(50, ge=1, le=2000)
    do_train: bool = True


@router.post("/self-learning/run", summary="Run pseudo-labeling + optional retrain")
def self_learning_run(req: SelfLearnReq = Depends()):
    cfg = SelfLearningConfig(
        source=req.source,
        target=req.target,
        round_name=req.round_name,
        low=req.low,
        high=req.high,
        eval_source=req.eval_source,
        test_size=req.test_size,
        n_trials=req.n_trials,
        do_train=req.do_train,
    )
    result = _service.run(cfg)
    return {"status": "success", **result}
