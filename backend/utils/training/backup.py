import os
import json
import shutil
from typing import List
from backend.utils.constants import (
    CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH,
    BACKUP_MODEL_PATH, BACKUP_TRAINING_LOG_PATH
)

class BackupManager:
    """
    Back up current models/logs by version and prune old ones.
    Strategy:
      • Call before overwriting (start of training/eval that mutates).
      • Call after success (to capture the new version).
      • Keep latest K (time‑ordered by version string).
    """

    @staticmethod
    def _list_dirs(path) -> List[str]:
        if not os.path.exists(path):
            return []
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    @staticmethod
    def backup_current(max_backups: int = 5) -> None:
        meta_path = CURRENT_MODEL_PATH / "version.json"
        if not meta_path.exists():
            return
        meta = json.load(open(meta_path, "r"))
        version = meta.get("version")
        if not version:
            return

        dst_model = BACKUP_MODEL_PATH / version
        dst_logs  = BACKUP_TRAINING_LOG_PATH / version
        dst_model.mkdir(parents=True, exist_ok=True)
        dst_logs.mkdir(parents=True, exist_ok=True)

        for f in CURRENT_MODEL_PATH.glob("*"):
            if f.is_file():
                shutil.copy(f, dst_model / f.name)

        for f in CURRENT_TRAINING_LOG_PATH.glob("*"):
            if f.is_file():
                shutil.copy(f, dst_logs / f.name)

        BackupManager._prune(BACKUP_MODEL_PATH, max_backups)
        BackupManager._prune(BACKUP_TRAINING_LOG_PATH, max_backups)

    @staticmethod
    def _prune(root, keep: int):
        versions = BackupManager._list_dirs(root)
        if len(versions) <= keep:
            return
        for v in versions[0 : len(versions) - keep]:
            shutil.rmtree(root / v, ignore_errors=True)
