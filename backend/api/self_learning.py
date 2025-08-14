from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional
from backend.api.dataset import _list_csv_files
from backend.core.self_learning_service import SelfLearningService, SelfLearningConfig

router = APIRouter()
_service = SelfLearningService()

class SelfLearnReq(BaseModel):
    source: str
    target: str
    round_name: str = Field(..., description="e.g. r01")
    low: float = Field(0.10, ge=0.0, le=0.5)
    high: float = Field(0.90, ge=0.5, le=1.0)
    eval_source: Optional[str] = None
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    n_trials: int = Field(50, ge=1, le=2000)
    do_train: bool = True
    chunk_size: Optional[int] = Field(None, description="If set, process target in chunks of this size")
    chunk_index: int = Field(0, ge=0, description="0-based index of the chunk to process when chunk_size is given")

    @classmethod
    def validate_filename(cls, v: str) -> str:
        files = _list_csv_files()
        if v not in files:
            raise ValueError(f"filename must be one of: {files}")
        return v

    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs):
        # keep default behavior; validator used ad-hoc below
        return super().__get_pydantic_core_schema__(*args, **kwargs)

@router.post("/self-learning/run", summary="Pseudo-label new addresses (chunkable) and optionally retrain")
def self_learning_run(req: SelfLearnReq = Depends()):
    # runtime file checks
    req.source = SelfLearnReq.validate_filename(req.source)
    req.target = SelfLearnReq.validate_filename(req.target)

    cfg = SelfLearningConfig(
        source=req.source, target=req.target, round_name=req.round_name,
        low=req.low, high=req.high, eval_source=req.eval_source,
        test_size=req.test_size, n_trials=req.n_trials, do_train=req.do_train,
        chunk_size=req.chunk_size, chunk_index=req.chunk_index,
    )
    result = _service.run(cfg)
    return result
