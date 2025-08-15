from __future__ import annotations

from typing import Dict, List, Optional, Any
import pandas as pd

from backend.utils.constants import DATA_PATH
from backend.core.predict_service import PredictService
from backend.utils.logger import logging
from backend.utils.etherscan_quota import QuotaExceeded

logger = logging.getLogger(__name__)


class UnlabeledService:
    """
    Upsert predictions into an 'unlabeled' CSV:
      - Creates the CSV if missing
      - Ensures all model label columns exist
      - Fills only positions currently == -1 using thresholds (<= low -> 0, >= high -> 1)
      - Never overwrites existing 0/1 labels
      - Optionally writes a parallel *_probs.csv with probabilities
    """

    def _read_or_create_unlabeled(self, filename: str, label_names: List[str]) -> pd.DataFrame:
        path = DATA_PATH / filename
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.str.lower()
        else:
            df = pd.DataFrame(columns=label_names)
            df.index.name = "Address"

        # ensure columns
        for c in label_names:
            if c not in df.columns:
                df[c] = -1

        # coerce label dtypes numeric and normalize to {-1,0,1} where possible
        for c in label_names:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
            except Exception:
                df[c] = -1

        # keep only label columns + preserve any extra columns as-is
        # label columns ordering
        cols = list(df.columns)
        # move labels to front in label_names order
        rest = [c for c in cols if c not in label_names]
        df = df[label_names + rest]
        return df

    def upsert_predictions(
        self,
        addresses: List[str],
        filename: str = "unlabeled.csv",
        low: float = 0.10,
        high: float = 0.90,
        write_probs: bool = True,
        probs_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not addresses:
            return {"status": "noop", "message": "No addresses supplied.", "file": filename}

        addresses = [a.lower() for a in addresses]

        try:
            predictor = PredictService()
            pred_map = predictor.predict(addresses)  # {addr: {"labels":..., "label_probs":...}}
            label_names = predictor.label_names
        except QuotaExceeded as e:
            return {"status": "quota_exhausted", "message": str(e)}
        except Exception as e:
            logger.exception(f"prediction failed: {e}")
            return {"status": "error", "message": str(e)}

        df = self._read_or_create_unlabeled(filename, label_names)

        # Prepare probs DF if requested
        probs_df = None
        if write_probs:
            pfname = probs_filename or filename.replace(".csv", "_probs.csv")
            # start from existing if any
            if (DATA_PATH / pfname).exists():
                probs_df = pd.read_csv(DATA_PATH / pfname, index_col=0)
                probs_df.index = probs_df.index.str.lower()
            else:
                probs_df = pd.DataFrame(columns=label_names)
                probs_df.index.name = "Address"
            # ensure all label columns exist
            for c in label_names:
                if c not in probs_df.columns:
                    probs_df[c] = pd.Series(dtype=float)

        # Upsert rows & fill only -1 positions
        updated_rows = 0
        created_rows = 0
        for addr in addresses:
            # ensure row exists
            if addr not in df.index:
                # create a new row with -1 for all labels
                df.loc[addr, label_names] = -1
                created_rows += 1

            # fill using thresholds
            out_row = df.loc[addr, label_names].astype(int).copy()
            lp = pred_map.get(addr, {}).get("label_probs", {})
            any_change = False
            for lbl in label_names:
                p = float(lp.get(lbl, 0.0))
                if out_row[lbl] == -1:
                    if p <= low:
                        out_row[lbl] = 0
                        any_change = True
                    elif p >= high:
                        out_row[lbl] = 1
                        any_change = True
                    # else keep -1

            df.loc[addr, label_names] = out_row.values
            if any_change:
                updated_rows += 1

            if write_probs and probs_df is not None:
                # always record probabilities
                row_probs = [float(pred_map.get(addr, {}).get("label_probs", {}).get(lbl, 0.0)) for lbl in label_names]
                probs_df.loc[addr, label_names] = row_probs

        # Write files
        out_path = DATA_PATH / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)

        out_probs = None
        if write_probs and probs_df is not None:
            pfname = probs_filename or filename.replace(".csv", "_probs.csv")
            probs_df.to_csv(DATA_PATH / pfname)
            out_probs = pfname

        return {
            "status": "ok",
            "file": filename,
            "probs_file": out_probs,
            "created_rows": int(created_rows),
            "updated_rows": int(updated_rows),
            "labels": label_names,
        }
