from fastapi import APIRouter, Depends
from typing import Literal
from pydantic import BaseModel
import zipfile
import io

from backend.core.feature_service import extract_base_feature_from_address
from fastapi.responses import StreamingResponse, JSONResponse
from backend.core.download_service import prepare_contract_download

from backend.utils.constants import DATA_PATH
from backend.core.dataset_service import get_full_dataset

router = APIRouter()

# Get CSV files at startup for dropdown
csv_files = [file.name for file in DATA_PATH.glob("*.csv")]

ChoiceLiteral = Literal[tuple(csv_files)] if csv_files else Literal[""]

class DatasetRequest(BaseModel):
    filename: ChoiceLiteral  # type: ignore # Dropdown in Swagger UI
    refresh: bool = True     # Default value

@router.post("/dataset", summary="Get dataset features")
def get_dataset(request: DatasetRequest = Depends()):
    dataset = get_full_dataset(request.filename, request.refresh)
    return dataset

@router.get("/download/{address}", summary="Download contract files as ZIP")
def download_files(address: str):
    files = prepare_contract_download(address)

    if not files:
        return JSONResponse(status_code=404, content={"message": "Contract not found on any chain."})

    # Create in-memory zip
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            if file.exists():
                zipf.write(file, arcname=file.name)

    zip_io.seek(0)  # rewind the buffer

    return StreamingResponse(
        zip_io,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={address}.zip"}
    )

@router.get("/extract/{address}", summary="Extract features from a contract")
def extract_features(address: str):
    features = extract_base_feature_from_address(address)
    return features
