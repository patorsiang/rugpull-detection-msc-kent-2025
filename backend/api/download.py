# backend/api/download.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from backend.core.download_service import prepare_contract_download
import zipfile
import io

router = APIRouter()

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
