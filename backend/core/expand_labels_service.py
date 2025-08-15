# backend/core/expand_labels_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd

from backend.utils.constants import DATA_PATH, CURRENT_TRAINING_LOG_PATH
from backend.core.meta_service import MetaService
from backend.core.predict_service import PredictService
from backend.utils.logger import logging
from backend.utils.etherscan_quota import QuotaExceeded

logger = logging.getLogger(__name__)

LowHigh = Union[float, Dict[str, float]]


@dataclass
class SelfLearningExpandConfig:
    filename: str
    round_name: str
    low: LowHigh = 0.10
    high: LowHigh = 0.90
    # chunking
    chunk_size: Optional[int] = None
    chunk_index: int = 0
    # post-processing
    save_confident_only: bool = True       # also emit *_confident.csv (all system labels strictly 0 or 1)


class ExpandLabelsService:
    """Fulfil missing system labels with predictions and expand with new labels found in file (chunk-friendly)."""

    class QuotaExhausted(Exception):
        pass

    # ------------------------------ PUBLIC ENTRYPOINT ------------------------------

    def run(self, cfg: SelfLearningExpandConfig) -> Dict:
        src_path = DATA_PATH / cfg.filename
        if not src_path.exists():
            raise FileNotFoundError(str(src_path))

        # 1) System labels (from meta; fallback to 3 canonical labels)
        sys_labels = self._get_system_labels()
        logger.info(f"System labels: {sys_labels}")

        # 2) Read CSV (optionally chunked)
        if cfg.chunk_size and cfg.chunk_size > 0:
            df, chunk_meta = self._read_chunk(src_path, cfg.chunk_size, cfg.chunk_index)
        else:
            df = pd.read_csv(src_path)
            chunk_meta = {
                "is_chunked": False,
                "chunk_index": 0,
                "total_chunks": 1,
                "start_row": 0,
                "end_row": len(df) - 1 if len(df) else -1,
                "rows_in_chunk": len(df),
            }

        # 3) Ensure Address and index by it
        if "Address" not in df.columns:
            candidates = [c for c in df.columns if c.lower() in ("contract", "contractaddress", "addr")]
            if candidates:
                df = df.rename(columns={candidates[0]: "Address"})
            else:
                raise ValueError("CSV must contain an 'Address' column.")

        df["Address"] = df["Address"].astype(str).str.strip()
        df = df.set_index("Address", drop=True)
        df = df[~df.index.duplicated(keep="last")]

        # 4) Identify label columns in the file
        present_label_cols = [c for c in df.columns if self._is_label_column(c)]
        present_sys = [c for c in present_label_cols if c in sys_labels]
        present_new = [c for c in present_label_cols if c not in sys_labels]

        # Ensure all system label columns exist (unknowns start as NaN)
        for lbl in sys_labels:
            if lbl not in df.columns:
                df[lbl] = np.nan

        addresses = list(df.index)

        # 5) Predict probabilities
        used_thresholds = self._expand_thresholds(cfg.low, cfg.high, sys_labels)
        try:
            pred_svc = PredictService()
            pred = pred_svc.predict(
                addresses=addresses,
                label_thresholds=used_thresholds,  # service may or may not use this; we threshold here anyway
                anomaly_threshold=None,
            )
        except QuotaExceeded as e:
            raise ExpandLabelsService.QuotaExhausted(str(e))

        status = str(pred.get("status"))
        if status not in ("ok", "success"):
            # Pass through error / quota message
            return pred

        results = pred.get("results") or pred.get("result", {}) or {}
        # Map: address -> {label: prob}
        label_probs_by_addr: Dict[str, Dict[str, float]] = {
            a: (results.get(a, {}).get("label_probs", {}) or {})
            for a in addresses
        }

        # 6) Fill logic:
        #    - ONLY replace when current value is NaN or -1
        #    - NEVER overwrite existing 0/1
        added_counts = {lbl: 0 for lbl in sys_labels}
        changed_counts = {lbl: 0 for lbl in sys_labels}  # stays 0 by design

        for addr in addresses:
            lp = label_probs_by_addr.get(addr, {})
            for lbl in sys_labels:
                p = lp.get(lbl)
                if p is None or not np.isfinite(p):
                    continue  # no prediction for this label

                lo = used_thresholds["low"][lbl]
                hi = used_thresholds["high"][lbl]
                new_val = self._apply_threshold(p, lo, hi)  # 1 / 0 / None
                if new_val is None:
                    continue  # uncertain → keep as-is (NaN or -1)

                cur_val = df.at[addr, lbl]
                if pd.isna(cur_val) or cur_val == -1:
                    df.at[addr, lbl] = int(new_val)
                    added_counts[lbl] += 1
                # else: existing 0/1 → keep as-is

        # 7) Save outputs (filtering & confident)
        base_name = Path(cfg.filename).stem
        chunk_tag = ""
        if chunk_meta["is_chunked"]:
            total = chunk_meta["total_chunks"]
            idx = chunk_meta["chunk_index"] + 1
            chunk_tag = f"_chunk{idx}of{total}"

        df_out = df.copy()
        if cfg.filtered:
            # Drop rows that still have any -1 in system labels
            mask_has_minus1 = df_out[sys_labels].eq(-1).any(axis=1)
            df_out = df_out.loc[~mask_has_minus1].copy()

        expanded_file = DATA_PATH / f"expanded_{base_name}_{cfg.round_name}{chunk_tag}.csv"
        df_out.reset_index(names="Address").to_csv(expanded_file, index=False)

        confident_file = None
        if cfg.save_confident_only:
            # Ensure all system label cols exist (already ensured above, but guard anyway)
            for lbl in sys_labels:
                if lbl not in df_out.columns:
                    df_out[lbl] = np.nan

            # Confident rows: every system label strictly 0 or 1 (no NaN, no -1)
            confident_mask = df_out[sys_labels].applymap(lambda v: v in (0, 1)).all(axis=1)
            df_conf = df_out.loc[confident_mask].copy()
            confident_file = DATA_PATH / f"expanded_{base_name}_{cfg.round_name}{chunk_tag}_confident.csv"
            df_conf.reset_index(names="Address").to_csv(confident_file, index=False)

        # 8) Log & return
        log = {
            "action": "expand_label",
            "input_file": str(src_path),
            "output_file": str(expanded_file),
            "confident_file": str(confident_file) if confident_file else None,
            "system_labels": sys_labels,
            "present_system_in_input": present_sys,
            "present_new_in_input": present_new,
            "added_counts": added_counts,
            "changed_counts": changed_counts,
            "used_thresholds": used_thresholds,
            "num_addresses_in_chunk": int(len(df)),
            "chunk_meta": chunk_meta,
            "filtered": cfg.filtered,
        }
        self._write_log(cfg, log, chunk_tag)

        return {
            "status": "ok",
            "message": "Labels expanded/fulfilled successfully (0/1 preserved; NaN/-1 replaced when confident).",
            "expanded_file": str(expanded_file),
            "confident_file": str(confident_file) if confident_file else None,
            "counts": {
                "rows_in_chunk_raw": int(len(df)),
                "rows_saved": int(len(df_out)),
                "added": added_counts,
                "changed": changed_counts,
            },
            "present_new_labels": present_new,
            "used_thresholds": used_thresholds,
            "chunk": chunk_meta,
        }

    # ------------------------------ HELPERS ------------------------------

    def _read_chunk(self, path: Path, chunk_size: int, chunk_index: int) -> Tuple[pd.DataFrame, Dict]:
        """Read only the requested chunk using pandas chunksize; also compute total_chunks."""
        total_chunks = 0
        for _ in pd.read_csv(path, chunksize=chunk_size):
            total_chunks += 1

        if total_chunks == 0:
            return pd.DataFrame(columns=["Address"]), {
                "is_chunked": True,
                "chunk_index": 0,
                "total_chunks": 0,
                "start_row": 0,
                "end_row": -1,
                "rows_in_chunk": 0,
            }

        if chunk_index >= total_chunks:
            raise ValueError(f"chunk_index {chunk_index} is out of range (total_chunks={total_chunks})")

        start_row = chunk_index * chunk_size
        end_row = start_row + chunk_size - 1
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            if i == chunk_index:
                rows_in_chunk = len(chunk)
                end_row = start_row + rows_in_chunk - 1
                meta = {
                    "is_chunked": True,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "start_row": int(start_row),
                    "end_row": int(end_row),
                    "rows_in_chunk": int(rows_in_chunk),
                }
                return chunk, meta

        raise RuntimeError("Failed to read requested chunk")

    def _get_system_labels(self) -> List[str]:
        """Ask MetaService; fallback to default 3 labels."""
        sys_labels: Optional[List[str]] = None
        try:
            meta = MetaService().get_meta()
            sys_labels = list(meta.get("labels", [])) or None
        except Exception:
            pass
        if not sys_labels:
            sys_labels = ["Mint", "Leak", "Limit"]
        return sys_labels

    def _is_label_column(self, c: str) -> bool:
        return c != "Address"

    def _expand_thresholds(self, low: LowHigh, high: LowHigh, labels: List[str]) -> Dict[str, Dict[str, float]]:
        """Return per-label low/high; broadcast scalar or fill missing from defaults."""
        def _mk_map(v: LowHigh, default: float) -> Dict[str, float]:
            if isinstance(v, dict):
                return {lbl: float(v.get(lbl, default)) for lbl in labels}
            return {lbl: float(v) for lbl in labels}

        low_map = _mk_map(low, 0.10)
        high_map = _mk_map(high, 0.90)
        return {"low": low_map, "high": high_map}

    def _apply_threshold(self, p: float, lo: float, hi: float) -> Optional[int]:
        if p >= hi:
            return 1
        if p <= lo:
            return 0
        return None

    def _write_log(self, cfg: SelfLearningExpandConfig, payload: Dict, chunk_tag: str):
        CURRENT_TRAINING_LOG_PATH.mkdir(parents=True, exist_ok=True)
        out = CURRENT_TRAINING_LOG_PATH / f"expand_label_{cfg.round_name}{chunk_tag}.json"
        with out.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
