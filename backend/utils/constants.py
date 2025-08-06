from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
FEATURE_PATH = DATA_PATH / "features"
LABELED_PATH = DATA_PATH / "labeled"
UNLABELED_PATH = DATA_PATH / "unlabeled"
MODELS_PATH = PROJECT_ROOT / "backend" / "models"
CURRENT_MODEL_PATH = MODELS_PATH / "current"
BACKUP_MODEL_PATH = MODELS_PATH / "backup"
LOGS_PATH = PROJECT_ROOT / "logs"
TRAINING_LOG_PATH = LOGS_PATH / "training"
