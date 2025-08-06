from fastapi import APIRouter
from backend.core.training_service import train_pipeline

router = APIRouter()

@router.post("/train", summary="Train Model")
def extract_features():
    result = train_pipeline()
    return {"status": "success", "metrics": result}
