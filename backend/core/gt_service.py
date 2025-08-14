from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from backend.utils.constants import DATA_PATH, GROUND_TRUTH_FILE
from backend.core.training_service import TrainingPipeline
from backend.utils.logger import logging
from backend.utils.etherscan_quota import QuotaExceeded

logger = logging.getLogger(__name__)


def _validate_labels(df: pd.DataFrame) -> None:
    """
    Light sanity check: allow numeric label columns with values in {-1,0,1}.
    Non-numeric columns are ignored (treated as non-labels/features).
    """
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            bad = ~df[c].isin([-1, 0, 1])
            if bad.any():
                sample = df.index[bad][:5].tolist()
                raise ValueError(f"Column '{c}' contains values outside -1/0/1 at rows: {sample} ...")


def _write_csv(df: pd.DataFrame, name: str) -> str:
    out = DATA_PATH / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return name


class GTService:
    """
    Promote a dataset to a canonical ground truth file and optionally kick off training.
    Quota is only relevant if/when training hits feature extraction; we catch QuotaExceeded and surface it.
    """

    def promote(
        self,
        new_source: str,
        output_name: Optional[str] = None,
        retrain: bool = True,
        eval_source: Optional[str] = None,
        test_size: float = 0.2,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        src_path = DATA_PATH / new_source
        if not src_path.exists():
            return {"status": "error", "message": f"File not found: {new_source}"}

        try:
            df = pd.read_csv(src_path, index_col=0)
            df.index = df.index.str.lower()
            _validate_labels(df)
        except Exception as e:
            logger.exception(f"Validation failed for {new_source}: {e}")
            return {"status": "error", "message": f"Validation failed: {e}"}

        out_name = output_name or f"groundtruth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            _write_csv(df, out_name)
        except Exception as e:
            logger.exception(f"Could not write {out_name}: {e}")
            return {"status": "error", "message": f"Could not write {out_name}: {e}"}

        result: Dict[str, Any] = {
            "status": "success",
            "message": "Promoted dataset written.",
            "new_groundtruth_file": out_name,
            "retrained": False,
            "metrics": None,
        }

        if not retrain:
            return result

        # Optional retrain: compare against a stable eval by default.
        try:
            pipeline = TrainingPipeline(n_trials=n_trials)
            meta = pipeline.run(
                source=out_name,
                eval_source=eval_source or GROUND_TRUTH_FILE,
                test_size=test_size,
            )
            result["retrained"] = True
            result["metrics"] = meta
            return result
        except QuotaExceeded as e:
            # Feature building required network but Etherscan quota is exhausted.
            logger.warning(f"Training aborted due to Etherscan quota: {e}")
            return {
                "status": "quota_exhausted",
                "message": f"Promotion succeeded but training aborted: {e}",
                "new_groundtruth_file": out_name,
                "retrained": False,
                "metrics": None,
            }
        except Exception as e:
            logger.exception(f"Training pipeline failed on {out_name}: {e}")
            return {
                "status": "error",
                "message": f"Promotion succeeded, but training failed: {e}",
                "new_groundtruth_file": out_name,
                "retrained": False,
                "metrics": None,
            }
