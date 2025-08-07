import os
import json
from datetime import datetime
from backend.utils.constants import CURRENT_MODEL_PATH, BACKUP_MODEL_PATH

def get_status():
    # Current version
    version_file = CURRENT_MODEL_PATH / "version.json"

    if os.path.exists(version_file):
        current_version = json.load(open(version_file, 'r'))['version']
    else:
        current_version = "Not trained yet"

    # List backup versions
    backup_versions = []
    if os.path.exists(BACKUP_MODEL_PATH):
        backup_versions = sorted(
            [name for name in os.listdir(BACKUP_MODEL_PATH) if os.path.isdir(os.path.join(BACKUP_MODEL_PATH, name))],
            reverse=True
        )

    return {
        "current_version": current_version,
        "backup_versions": backup_versions,
        "timestamp": datetime.now().isoformat()
    }
