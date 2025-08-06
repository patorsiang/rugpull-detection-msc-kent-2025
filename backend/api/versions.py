# backend/api/version.py
from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter()

@router.get("/versions", summary="Get all model versions and current status")
def get_versions():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    current_path = os.path.join(base_dir, "current")
    backup_path = os.path.join(base_dir, "backup")

    # Current version
    version_file = os.path.join(current_path, "version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            current_version = f.read().strip()
    else:
        current_version = "Not trained yet"

    # List backup versions
    backup_versions = []
    if os.path.exists(backup_path):
        backup_versions = sorted(
            [name for name in os.listdir(backup_path) if os.path.isdir(os.path.join(backup_path, name))],
            reverse=True
        )

    return {
        "status": "Rug Pull Detection API is running.",
        "current_version": current_version,
        "backup_versions": backup_versions,
        "timestamp": datetime.now().isoformat()
    }
