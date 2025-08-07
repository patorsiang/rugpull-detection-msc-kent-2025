import os
import shutil
import json

from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, BACKUP_MODEL_PATH, BACKUP_TRAINING_LOG_PATH  # Adjust if needed
from backend.utils.logger import get_logger

logger = get_logger("backup")

def backup_model_and_logs():
    # Load version
    version_meta_path = CURRENT_MODEL_PATH / "version.json"

    if version_meta_path.exists():
        with open(version_meta_path, "r") as f:
            version_meta = json.load(f)
        version = version_meta.get("version")
        if not version:
            logger.error("Missing 'version' key in version.json")
            raise ValueError("Missing 'version' key in version.json")

        # Paths
        model_backup_dir = BACKUP_MODEL_PATH / version
        log_backup_dir = BACKUP_TRAINING_LOG_PATH / version

        # Create backup dirs
        os.makedirs(model_backup_dir, exist_ok=True)
        os.makedirs(log_backup_dir, exist_ok=True)

        # Copy all from model/current to model/backup/{version}
        for file in CURRENT_MODEL_PATH.glob("*"):
            if file.is_file():
                shutil.copy(file, model_backup_dir / file.name)

        # Copy all from logs/training/current to logs/training/backup/{version}
        for file in CURRENT_TRAINING_LOG_PATH.glob("*"):
            if file.is_file():
                shutil.copy(file, log_backup_dir / file.name)

        logger.info(f"✅ Backup completed: {version}")
