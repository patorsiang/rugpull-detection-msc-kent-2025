import os
import json
from datetime import datetime
from typing import Dict, List
from backend.utils.constants import CURRENT_MODEL_PATH, BACKUP_MODEL_PATH

class MetaService:
    """Read/write lightweight model metadata (version, backups, label names)."""

    @staticmethod
    def get_status() -> Dict:
        version_file = CURRENT_MODEL_PATH / "version.json"
        if os.path.exists(version_file):
            try:
                current_version = json.load(open(version_file, "r")).get("version", "Not trained yet")
            except Exception:
                current_version = "Not trained yet"
        else:
            current_version = "Not trained yet"

        backups: List[str] = []
        if os.path.exists(BACKUP_MODEL_PATH):
            backups = sorted(
                [d for d in os.listdir(BACKUP_MODEL_PATH) if os.path.isdir(os.path.join(BACKUP_MODEL_PATH, d))],
                reverse=True,
            )

        return {
            "current_version": current_version,
            "backup_versions": backups,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def read_version() -> Dict:
        path = CURRENT_MODEL_PATH / "version.json"
        return json.load(open(path, "r")) if os.path.exists(path) else {}

    @staticmethod
    def write_version(meta: Dict) -> None:
        path = CURRENT_MODEL_PATH / "version.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(meta, open(path, "w"), indent=2)
