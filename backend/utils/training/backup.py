import shutil
from datetime import datetime
from backend.utils.constants import (
    CURRENT_MODEL_PATH, BACKUP_MODEL_PATH,
    CURRENT_TRAINING_LOG_PATH, BACKUP_TRAINING_LOG_PATH
)
from backend.utils.logger import logging

logger = logging.getLogger(__name__)

class BackupManager:
    @staticmethod
    def backup_current() -> str:
        """Backup current model & logs only if there is something to back up."""
        model_files = list(CURRENT_MODEL_PATH.glob("*")) if CURRENT_MODEL_PATH.exists() else []
        log_files = list(CURRENT_TRAINING_LOG_PATH.glob("*")) if CURRENT_TRAINING_LOG_PATH.exists() else []

        # Nothing to back up
        if not model_files and not log_files:
            logger.info("No model or logs found to back up.")
            return ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Backup models
        if model_files:
            model_dst = BACKUP_MODEL_PATH / ts
            model_dst.mkdir(parents=True, exist_ok=True)
            for item in model_files:
                try:
                    if item.is_file():
                        shutil.copy2(item, model_dst / item.name)
                except Exception as e:
                    logger.warning(f"Failed backing up model {item}: {e}")

        # Backup logs
        if log_files:
            logs_dst = BACKUP_TRAINING_LOG_PATH / ts
            logs_dst.mkdir(parents=True, exist_ok=True)
            for item in log_files:
                try:
                    if item.is_file():
                        shutil.move(item, logs_dst / item.name)
                except Exception as e:
                    logger.warning(f"Failed backing up log {item}: {e}")

        logger.info(f"Backup completed: {ts}")
        return ts
