from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.utils.constants import DATA_PATH, CURRENT_TRAINING_LOG_PATH, CURRENT_MODEL_PATH
from backend.core.meta_service import MetaService
from backend.core.predict_service import PredictService
from backend.core.training_service import TrainingPipeline
from backend.utils.logger import logging
from backend.utils.etherscan_quota import quota_guard, QuotaExceeded

logger = logging.getLogger(__name__)

# ------------------------- Config -------------------------

@dataclass
class SelfLearningConfig:
    # Required
    source: str                    # ground truth CSV (addresses index or Address column)
    target: str                    # file to pseudo-label
    round_name: str                # suffix for outputs

    # Pseudo-label thresholds (scalar or per-label map)
    low: Union[float, Dict[str, float]] = 0.10
    high: Union[float, Dict[str, float]] = 0.90

    # Evaluation / training
    eval_source: Optional[str] = None    # defaults to source
    test_size: float = 0.2
    n_trials: int = 50                   # heavy finalize trials (only if accepted & do_train)
    do_train: bool = True

    # Chunking
    chunk_size: Optional[int] = None
    chunk_index: int = 0

    # Speed/acceptance controls
    preview_trials: int = 8        # cheap preview trials for accept/reject
    accept_min_delta: float = 0.0  # require at least this F1 improvement
    cache_baseline: bool = True    # reuse baseline F1 across runs when possible


# ------------------------- Service -------------------------

class SelfLearningService:
    """
    Self-learning pipeline:

      1) Predict on target addresses not in source.
      2) Pseudo-label using thresholds for EXISTING source labels ("old labels").
      3) Build a candidate ground truth (merge only new rows).
      4) Accept the candidate iff preview F1 >= baseline F1 + accept_min_delta.
      5) If accepted AND do_train=True, run one heavy train_and_save + full pipeline.
    """

    def __init__(self):
        self.data_path = DATA_PATH
        self.logs_path = CURRENT_TRAINING_LOG_PATH
        self.model_path = CURRENT_MODEL_PATH
        self.logs_path.mkdir(parents=True, exist_ok=True)
        # Cache: (eval_source, test_size, current_version) -> baseline_f1
        self._baseline_cache: Dict[tuple, Optional[float]] = {}

    # ---------- Public API ----------

    def run(self, cfg: SelfLearningConfig) -> Dict:
        # Load source/target
        src_df = self._read_csv(cfg.source)
        tgt_df = self._read_csv(cfg.target)

        # Old labels = numeric columns in source
        old_labels = sorted(self._numeric_label_columns(src_df))
        if not old_labels:
            raise ValueError("Source has no numeric label columns to learn on.")

        # Normalize source labels upfront
        src_df = self._ensure_int_labels(src_df[old_labels])

        # Ensure target has at least old labels
        for col in old_labels:
            if col not in tgt_df.columns:
                tgt_df[col] = -1
        tgt_df[old_labels] = self._ensure_int_labels(tgt_df[old_labels])

        # New addresses to predict
        new_addresses_all = sorted(list(set(tgt_df.index) - set(src_df.index)))
        if not new_addresses_all:
            note = "No new addresses to process; target is already contained in source."
            log = self._write_log(cfg, {"status": "no_change", "note": note})
            return {
                "status": "no_change",
                "note": note,
                "log_file": log,
                "source_file": cfg.source,
                "target_file": cfg.target,
            }

        # Optional chunking
        if cfg.chunk_size and cfg.chunk_size > 0:
            start = cfg.chunk_index * cfg.chunk_size
            end = start + cfg.chunk_size
            new_addresses = new_addresses_all[start:end]
            remaining_after_chunk = new_addresses_all[end:]
        else:
            new_addresses = new_addresses_all
            remaining_after_chunk = []

        # Quota guard (pre)
        if quota_guard.is_exhausted(1):
            note = "Etherscan daily quota exhausted; cannot extract features for predictions."
            log = self._write_log(cfg, {"status": "quota_exhausted_precheck", "note": note})
            return {"status": "quota_exhausted", "note": note, "log_file": log}

        # Predict (batch)
        predictor = PredictService()
        pred_map: Dict[str, Dict] = {}
        skipped: List[Dict[str, str]] = []
        quota_hit_midrun = False

        try:
            batch_out = predictor.predict(new_addresses)
            raw = batch_out.get("results", batch_out) if isinstance(batch_out, dict) else batch_out
            for a in new_addresses:
                v = raw.get(a) or raw.get(a.lower())
                if v is None:
                    skipped.append({"address": a, "reason": "no_output"})
                else:
                    pred_map[a.lower()] = v
        except QuotaExceeded as qe:
            skipped.extend({"address": a, "reason": f"quota_exceeded: {qe}"} for a in new_addresses)
            quota_hit_midrun = True
        except Exception as e:
            skipped.extend({"address": a, "reason": f"batch_predict_error: {e}"} for a in new_addresses)

        if not pred_map:
            note = "No predictions produced (skipped or quota hit)."
            rem_file = None
            if remaining_after_chunk or new_addresses:
                rem_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
                exist_idx = [i for i in (remaining_after_chunk or new_addresses) if i in tgt_df.index]
                if exist_idx:
                    # save full rows to preserve context for next pass
                    self._write_csv(tgt_df.loc[exist_idx], rem_name)
                    rem_file = rem_name
            log = self._write_log(cfg, {
                "status": "no_predictions",
                "note": note,
                "quota_hit_midrun": quota_hit_midrun,
                "skipped": skipped,
                "files": {"remaining_file": rem_file},
            })
            return {
                "status": "no_predictions",
                "note": note,
                "log_file": log,
                "remaining_file": rem_file,
                "skipped": skipped,
            }

        # Build prob/hard for OLD labels only (what we will actually threshold)
        prob_df, _ = self._make_prob_and_hard(pred_map, old_labels, nan_for_missing=True)

        # Keep only addresses actually predicted
        available = sorted(list(set(new_addresses) & set(prob_df.index)))
        missing_pred = sorted(list(set(new_addresses) - set(available)))
        if missing_pred:
            skipped.extend({"address": a, "reason": "not_in_predictions"} for a in missing_pred)

        if not available:
            note = "All selected addresses failed prediction."
            rem_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
            # save the full remaining rows to retry later
            self._write_csv(tgt_df.loc[new_addresses], rem_name)
            log = self._write_log(cfg, {
                "status": "no_available",
                "note": note,
                "skipped": skipped,
                "files": {"remaining_file": rem_name},
            })
            return {"status": "no_available", "log_file": log, "remaining_file": rem_name, "skipped": skipped}

        # Thresholding (fill only where -1)
        prob_df = prob_df.reindex(available)
        target_slice = tgt_df.loc[available, old_labels].copy()
        filled_df = self._fill_with_thresholds(target_slice, prob_df, cfg.low, cfg.high)

        # Pseudo rows: require all OLD labels decided
        pseudo_mask = (filled_df[old_labels] != -1).all(axis=1)
        pseudo_df = self._ensure_int_labels(filled_df.loc[pseudo_mask, old_labels])

        # Uncertain rows: some old label still -1 (export with p_<label> for context)
        uncertain_mask = ~pseudo_mask
        uncertain_export = None
        if uncertain_mask.any():
            prob_old = prob_df[old_labels].add_prefix("p_")
            uncertain_export = filled_df.loc[uncertain_mask, old_labels].join(prob_old, how="left")

        artifacts = {
            "pseudo_file": None,
            "uncertain_file": None,
            "skipped_addresses_file": None,
            "candidate_file": None,
            "accepted": False,
            "remaining_file": None,
        }

        # Write uncertain/skipped
        if uncertain_export is not None and not uncertain_export.empty:
            hard_name = self._suffix_file(cfg.target, f"uncertain-{cfg.round_name}")
            self._write_csv(uncertain_export, hard_name)
            artifacts["uncertain_file"] = hard_name

        if skipped:
            skipped_name = self._suffix_file(cfg.target, f"skipped-{cfg.round_name}")
            pd.DataFrame(skipped).to_csv(self.data_path / skipped_name, index=False)
            artifacts["skipped_addresses_file"] = skipped_name

        if pseudo_df.empty:
            note = "No high-confidence pseudo labels for OLD labels."
            log = self._write_log(cfg, {
                "status": "no_pseudo",
                "note": note,
                "counts": {"new_addresses": len(new_addresses),
                           "pseudo_rows": 0,
                           "uncertain_rows": int(uncertain_mask.sum())},
                "labels": old_labels,
            })
            return {"status": "no_pseudo", "note": note, "log_file": log, **artifacts}

        # Save pseudo (only old labels)
        pseudo_name = self._suffix_file(cfg.target, f"pseudo-{cfg.round_name}")
        self._write_csv(pseudo_df[old_labels], pseudo_name)
        artifacts["pseudo_file"] = pseudo_name

        # Build candidate GT: append only NEW rows (addresses), columns = old_labels
        add_rows = pseudo_df.loc[~pseudo_df.index.isin(src_df.index), old_labels]
        candidate = pd.concat([src_df[old_labels], self._ensure_int_labels(add_rows)], axis=0)
        candidate = self._ensure_int_labels(candidate.reindex(columns=sorted(candidate.columns)))
        cand_name = self._suffix_file(cfg.source, f"merged-{cfg.round_name}")
        self._write_csv(candidate, cand_name)
        artifacts["candidate_file"] = cand_name

        # Fast acceptance & optional finalize
        eval_src = cfg.eval_source or cfg.source
        try:
            metrics = self._evaluate_and_maybe_retrain(
                source=cand_name,
                eval_source=eval_src,
                test_size=cfg.test_size,
                n_trials=cfg.n_trials,
                do_train=cfg.do_train,
                preview_trials=cfg.preview_trials,
                accept_min_delta=cfg.accept_min_delta,
                cache_baseline=cfg.cache_baseline,
            )
            accepted = bool(metrics.get("accepted"))
        except Exception as e:
            logger.exception(f"evaluation/retrain failed: {e}")
            metrics = {"error": str(e)}
            accepted = False

        artifacts["accepted"] = accepted

        # Remaining (if chunked)
        if remaining_after_chunk:
            rem_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
            # save full rows so next pass has context
            self._write_csv(tgt_df.loc[remaining_after_chunk], rem_name)
            artifacts["remaining_file"] = rem_name

        # Log & return
        log = self._write_log(cfg, {
            "status": "ok",
            "counts": {"new_addresses": len(new_addresses),
                       "pseudo_rows": int(len(pseudo_df)),
                       "uncertain_rows": int(uncertain_mask.sum())},
            "labels": {"old": old_labels},
            "files": artifacts,
            "quota_hit_midrun": quota_hit_midrun,
            "metrics": metrics,
        })
        return {"status": "ok", "log_file": log, **artifacts, "metrics": metrics}

    # ---------- Helpers ----------

    def _numeric_label_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _looks_like_eth_address_col(self, series: pd.Series) -> bool:
        s = series.dropna().astype(str).str.strip().str.lower()
        if s.empty:
            return False
        mask = s.str.startswith("0x") & (s.str.len() >= 40)
        return (mask.sum() / len(s)) >= 0.8

    def _read_csv(self, name: str) -> pd.DataFrame:
        p = self.data_path / name
        df = pd.read_csv(p)

        # Prefer address-like column if present
        idx_col = None
        for cand in ["address", "contract", "addr", "Address"]:
            if cand in df.columns:
                idx_col = cand
                break

        if idx_col is None and len(df.columns) > 0:
            first_col = df.columns[0]
            if first_col.lower() in {"unnamed: 0", "index"}:
                df = df.set_index(first_col)
            elif self._looks_like_eth_address_col(df[first_col]):
                idx_col = first_col

        if idx_col is not None:
            df[idx_col] = df[idx_col].astype(str).str.lower().str.strip()
            df = df.drop_duplicates(subset=[idx_col]).set_index(idx_col)
        else:
            df.index = df.index.astype(str).str.lower().str.strip()
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _ensure_int_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric columns, fill NaNs with -1, cast to int64."""
        if df.empty:
            return df
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.fillna(-1)
        try:
            out = out.astype("int64")
        except Exception:
            for c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-1).astype("int64")
        return out

    def _write_csv(self, df: pd.DataFrame, name: str) -> str:
        (self.data_path / name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path / name)
        return name

    def _suffix_file(self, fname: str, suffix: str) -> str:
        p = Path(fname)
        return f"{p.stem}-{suffix}{p.suffix}"

    def _make_prob_and_hard(
        self,
        pred_map: Dict[str, Dict],
        label_names: List[str],
        nan_for_missing: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build probability & hard-label frames.
        If nan_for_missing=True, any label not present in prediction for an address becomes NaN
        (so we won't accidentally treat "missing" as 0-probability).
        """
        probs, hards = {}, {}
        for addr, out in pred_map.items():
            lp = out.get("label_probs", {}) or {}
            lb = out.get("labels", {}) or {}
            p_row, h_row = [], []
            for l in label_names:
                if l in lp:
                    p_row.append(float(lp[l]))
                else:
                    p_row.append(float("nan") if nan_for_missing else 0.0)
                h_row.append(int(lb.get(l, 0)))
            probs[addr] = p_row
            hards[addr] = h_row
        prob_df = pd.DataFrame.from_dict(probs, orient="index", columns=label_names)
        hard_df = pd.DataFrame.from_dict(hards, orient="index", columns=label_names).astype(int)
        prob_df.index = prob_df.index.str.lower()
        hard_df.index = hard_df.index.str.lower()
        return prob_df, hard_df

    def _fill_with_thresholds(
        self,
        original: pd.DataFrame,
        probs: pd.DataFrame,
        low: Union[float, Dict[str, float]],
        high: Union[float, Dict[str, float]],
    ) -> pd.DataFrame:
        out = original.copy()
        for col in out.columns:
            lo = low.get(col, low) if isinstance(low, dict) else low
            hi = high.get(col, high) if isinstance(high, dict) else high
            mask_unknown = (out[col] == -1)
            if not mask_unknown.any():
                continue
            # p may contain NaN (we leave -1 for unknowns if prob is missing)
            p = probs.loc[mask_unknown.index[mask_unknown], col]
            out.loc[p.index[p <= lo], col] = 0
            out.loc[p.index[p >= hi], col] = 1
        return self._ensure_int_labels(out)

    # ---------- Fast evaluate/train ----------

    def _evaluate_and_maybe_retrain(
        self,
        source: str,
        eval_source: str,
        test_size: float,
        n_trials: int,
        do_train: bool,
        preview_trials: int,
        accept_min_delta: float,
        cache_baseline: bool,
    ) -> Dict:
        # Baseline (cached)
        version = (MetaService.get_status() or {}).get("current_version")
        cache_key = (eval_source, float(test_size), str(version))
        b = None
        if cache_baseline and cache_key in self._baseline_cache:
            b = self._baseline_cache[cache_key]
        else:
            pipeline_eval = TrainingPipeline(n_trials=1)  # should not tune
            baseline = pipeline_eval.eval.evaluate(
                test_size=test_size,
                source=eval_source,
                freeze_gru=True,
                freeze_sklearn=True,
            )
            b = baseline.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")
            if cache_baseline:
                self._baseline_cache[cache_key] = b

        # Preview (fast)
        pipeline_preview = TrainingPipeline(n_trials=int(preview_trials))
        preview = pipeline_preview.trainer.train_preview(
            test_size=test_size,
            train_source=source,
            eval_source=eval_source
        )
        n = preview.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

        # Accept if improved by at least accept_min_delta
        accepted = (b is None) or (n is not None and n >= (b if b is not None else -1.0) + float(accept_min_delta))
        metrics = {
            "baseline_f1": float(b) if b is not None else None,
            "trial_f1": float(n) if n is not None else None,
            "accepted": bool(accepted),
        }

        # Finalize (heavy) only when accepted & requested
        if accepted and do_train:
            pipeline_final = TrainingPipeline(n_trials=int(n_trials))
            pipeline_final.trainer.train_and_save(test_size=test_size, source=source)
            pipeline_final.full.run(source=source)

        return metrics

    # ---------- Logging ----------

    def _write_log(self, cfg: SelfLearningConfig, payload: dict) -> str:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"self_learning_{cfg.round_name}_{ts}.json"
        log_path = self.logs_path / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(log_path)
