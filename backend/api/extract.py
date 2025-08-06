from fastapi import APIRouter
from backend.core.feature_service import extract_base_feature_from_address

router = APIRouter()

@router.get("/extract/{address}", summary="Extract features from a contract")
def extract_features(address: str):
    features = extract_base_feature_from_address(address)
    return features
