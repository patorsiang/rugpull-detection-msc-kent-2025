import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from backend.utils.constants import CURRENT_MODEL_PATH, BACKUP_MODEL_PATH
from backend.utils.logger import logging

logger = logging.getLogger(__name__)

class MetaService:
    """Read/write lightweight model metadata (version, backups, label names)."""

    @staticmethod
    def get_status() -> Dict:
        """
        Return a rich snapshot of the current model state:
          - is_trained/current_version
          - label_names and counts
          - train/test sizes and notes
          - model files (exists/size/mtime) pulled from version.json filenames
          - fusion thresholds/weights
          - anomaly fusion/AE threshold
          - backups list
          - when version.json was last modified
        """
        version_path = CURRENT_MODEL_PATH / "version.json"
        meta: Dict = {}
        current_version = "Not trained yet"
        is_trained = False
        version_last_modified: Optional[str] = None

        if version_path.exists():
            try:
                meta = json.load(open(version_path, "r"))
                current_version = meta.get("version", "Not trained yet")
                is_trained = current_version != "Not trained yet"
                try:
                    mtime = datetime.fromtimestamp(version_path.stat().st_mtime).isoformat()
                    version_last_modified = mtime
                except Exception:
                    version_last_modified = None
            except Exception as e:
                logger.warning(f"cannot read version.json: {e}")

        # Backups
        backups: List[str] = []
        if os.path.exists(BACKUP_MODEL_PATH):
            try:
                backups = sorted(
                    [d for d in os.listdir(BACKUP_MODEL_PATH) if os.path.isdir(os.path.join(BACKUP_MODEL_PATH, d))],
                    reverse=True,
                )
            except Exception as e:
                logger.warning(f"cannot list backups: {e}")
                backups = []

        # Labels / sizes / notes
        label_names: List[str] = list(meta.get("label_names", []))
        n_labels = len(label_names)
        train_size = meta.get("train_size")
        test_size = meta.get("test_size")
        notes = meta.get("notes")

        # Model filenames from meta
        clf_summary = meta.get("clf_model_summary", {}) or {}
        anm_summary = meta.get("anomaly_model_summary", {}) or {}

        def _file_stat(fname: Optional[str]) -> Dict:
            if not fname:
                return {"exists": False, "size": 0, "mtime": None, "path": None}
            p = CURRENT_MODEL_PATH / fname
            try:
                st = p.stat()
                return {
                    "exists": True,
                    "size": int(st.st_size),
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "path": str(p),
                }
            except Exception:
                return {"exists": False, "size": 0, "mtime": None, "path": str(p)}

        # Collect file stats for each model listed in meta
        model_files = {}
        for section_name, section in [
            ("general_model", clf_summary.get("general_model", {})),
            ("sol_model",     clf_summary.get("sol_model", {})),
            ("opcode_model",  clf_summary.get("opcode_model", {})),
            ("gru_model",     clf_summary.get("gru_model", {})),
            ("if_general_model", anm_summary.get("general_model", {})),
            ("if_sol_model",     anm_summary.get("sol_model", {})),
            ("if_opcode_model",  anm_summary.get("opcode_model", {})),
            ("gru_ae_model",     anm_summary.get("gru_model", {})),
        ]:
            fname = section.get("filename")
            model_files[section_name] = _file_stat(fname)

        # Fusion configs (echo back for quick inspection)
        clf_fusion = (clf_summary.get("fusion_model") or {})
        anm_fusion = (anm_summary.get("fusion_model") or {})
        ae_threshold = (anm_summary.get("gru_model") or {}).get("threshold")

        return {
            "is_trained": is_trained,
            "current_version": current_version,
            "timestamp": datetime.now().isoformat(),
            "version_last_modified": version_last_modified,

            "label_names": label_names,
            "n_labels": n_labels,
            "train_size": train_size,
            "test_size": test_size,
            "notes": notes,

            "model_files": model_files,  # per-file exists/size/mtime/path

            "clf_fusion": {
                "weights": clf_fusion.get("weights"),
                "thresholds": clf_fusion.get("thresholds"),
                "f1_score": clf_fusion.get("f1_score"),
            },
            "anomaly_fusion": {
                "weights": anm_fusion.get("weights"),
                "threshold": anm_fusion.get("threshold"),
                "f1_score": anm_fusion.get("f1_score"),
                "ae_threshold": ae_threshold,
            },

            "backup_versions": backups,
        }

    @staticmethod
    def read_version() -> Dict:
        path = CURRENT_MODEL_PATH / "version.json"
        try:
            return json.load(open(path, "r")) if os.path.exists(path) else {}
        except Exception as e:
            logger.warning(f"cannot read version.json: {e}")
            return {}

    @staticmethod
    def write_version(meta: Dict) -> None:
        path = CURRENT_MODEL_PATH / "version.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
