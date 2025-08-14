from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backend.utils.constants import DATA_PATH, CURRENT_TRAINING_LOG_PATH, CURRENT_MODEL_PATH
from backend.core.meta_service import MetaService
from backend.core.predict_service import PredictService
from backend.core.training_service import TrainingPipeline
from backend.utils.logger import logging
from backend.utils.etherscan_quota import quota_guard, QuotaExceeded

logger = logging.getLogger(__name__)

@dataclass
class SelfLearningConfig:
    source: str
    target: str
    round_name: str
    low: float = 0.10
    high: float = 0.90
    eval_source: Optional[str] = None
    test_size: float = 0.2
    n_trials: int = 50
    do_train: bool = True
    chunk_size: Optional[int] = None
    chunk_index: int = 0

class SelfLearningService:
    """Pseudo-label target (vs source); merge or create new GT; optional retrain."""

    def __init__(self):
        self.data_path = DATA_PATH
        self.logs_path = CURRENT_TRAINING_LOG_PATH
        self.model_path = CURRENT_MODEL_PATH
        self.logs_path.mkdir(parents=True, exist_ok=True)

    # ---------- Public API ----------

    def run(self, cfg: SelfLearningConfig) -> Dict:
        src_df = self._read_csv(cfg.source)
        tgt_df = self._read_csv(cfg.target)

        # Normalize indices (addresses) to lower-case
        src_df.index = src_df.index.str.lower()
        tgt_df.index = tgt_df.index.str.lower()

        # Fast-path: nothing to do if target ⊆ source
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

        # Ensure label columns are aligned between source and target
        src_labels = set(src_df.columns)
        tgt_labels = set(tgt_df.columns)
        all_labels = sorted(list(src_labels | tgt_labels))

        # If ground truth (source) has a new label absent in target:
        # add that column to target filled with -1 and save back to itself
        new_labels_in_src = sorted(list(src_labels - tgt_labels))
        if new_labels_in_src:
            for col in new_labels_in_src:
                tgt_df[col] = -1
            tgt_df = tgt_df[sorted(tgt_df.columns)]
            self._write_csv(tgt_df, cfg.target)

        # ---------- Quota guard (pre-check) ----------
        if quota_guard.is_exhausted(1):
            note = "Etherscan daily quota exhausted; cannot extract features for predictions."
            log = self._write_log(cfg, {"status": "quota_exhausted_precheck", "note": note})
            return {"status": "quota_exhausted", "note": note, "log_file": log}

        # ---------- Predict only for truly new addresses ----------
        predictor = PredictService()
        pred_map: Dict[str, Dict] = {}
        skipped_addresses: List[Dict[str, str]] = []
        quota_hit_midrun = False

        for a in new_addresses:
            # Mid-run quota check to avoid surprises
            if quota_guard.is_exhausted(1):
                quota_hit_midrun = True
                logger.warning("Quota exhausted mid-run; stopping predictions.")
                break
            try:
                out = predictor.predict([a])  # predict expects list
                if a in out:
                    pred_map[a] = out[a]
                else:
                    skipped_addresses.append({"address": a, "reason": "no_output"})
            except QuotaExceeded as qe:
                skipped_addresses.append({"address": a, "reason": f"quota_exceeded: {qe}"})
                quota_hit_midrun = True
                break
            except Exception as e:
                # e.g., feature extraction failure, network, etc.
                skipped_addresses.append({"address": a, "reason": str(e)})

        if not pred_map:
            note = "No predictions were produced for the selected addresses (all skipped or quota hit)."
            remaining_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
            # Remaining includes: all not-attempted (chunk remainder) + attempted but failed (skipped)
            attempted = set(addr for addr in new_addresses if addr in pred_map or any(x["address"] == addr for x in skipped_addresses))
            not_attempted = sorted(list(set(new_addresses) - attempted))
            remainder = sorted(set(remaining_after_chunk) | set(not_attempted))
            if remainder:
                self._write_csv(tgt_df.loc[remainder], remaining_name)
            else:
                remaining_name = None
            log = self._write_log(cfg, {
                "status": "no_predictions",
                "note": note,
                "quota_hit_midrun": quota_hit_midrun,
                "skipped": skipped_addresses,
                "files": {"remaining_file": remaining_name},
            })
            return {"status": "no_predictions", "note": note, "log_file": log, "remaining_file": remaining_name, "skipped": skipped_addresses}

        # Label names: prefer meta; else keys from first result; else union
        label_names = self._resolve_label_names(pred_map, all_labels)

        # Build prob/label frames from predictions
        prob_df, hard_df = self._make_prob_and_hard(pred_map, label_names)

        # Only operate on addresses we actually predicted
        available = sorted(list(set(new_addresses) & set(prob_df.index)))
        missing = sorted(list(set(new_addresses) - set(available)))
        if missing:
            skipped_addresses.extend({"address": a, "reason": "not_in_predictions"} for a in missing)

        # Guard: if nothing available, bail with remaining file
        if not available:
            remaining_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
            self._write_csv(tgt_df.loc[new_addresses], remaining_name)
            log = self._write_log(cfg, {
                "status": "no_available",
                "note": "All selected addresses failed prediction.",
                "skipped": skipped_addresses,
                "files": {"remaining_file": remaining_name},
            })
            return {"status": "no_available", "log_file": log, "remaining_file": remaining_name, "skipped": skipped_addresses}

        # Slice frames safely
        prob_df = prob_df.reindex(available)
        # Original target rows for these addresses (labels subset)
        # If eval_source/target do not have some meta labels yet, fill with -1 to allow thresholding
        missing_cols = [c for c in label_names if c not in tgt_df.columns]
        if missing_cols:
            for c in missing_cols:
                tgt_df[c] = -1
        target_slice = tgt_df.loc[available, label_names].copy()

        # Apply thresholds only where target currently == -1
        filled_df = self._fill_with_thresholds(
            original=target_slice,
            probs=prob_df,
            low=cfg.low,
            high=cfg.high,
        )

        # ---------- Derive artifacts ----------
        # 1) Pseudo-labeled (rows that got all labels decided: no -1 left)
        pseudo_df = filled_df[(filled_df != -1).all(axis=1)].copy()

        # 2) Uncertain / hard cases (any label still -1 because prob in (low, high))
        # Save for manual review; include probabilities for context
        uncertain_mask = (filled_df == -1).any(axis=1)
        uncertain_df = filled_df[uncertain_mask].copy()
        if not uncertain_df.empty:
            # merge back probs for visibility: col -> p_<col>
            prob_renamed = prob_df.add_prefix("p_")
            uncertain_export = uncertain_df.join(prob_renamed, how="left")
        else:
            uncertain_export = pd.DataFrame(index=[])

        # 3) Remaining addresses to retry later:
        #    (chunk remainder) U (attempted-but-missing) U (skipped mid-run)
        remainder = sorted(set(remaining_after_chunk) | set(missing) |
                           set([s["address"] for s in skipped_addresses if s.get("reason","").startswith("quota_exceeded")]))
        artifacts = {
            "pseudo_file": None,
            "merged_file": None,
            "new_groundtruth_file": None,
            "target_augmented": cfg.target if new_labels_in_src else None,
            "skipped_addresses_file": None,
            "uncertain_file": None,
            "remaining_file": None,
        }

        # Save uncertain/hard file (if any)
        if not uncertain_export.empty:
            hard_name = self._suffix_file(cfg.target, f"uncertain-{cfg.round_name}")
            self._write_csv(uncertain_export, hard_name)
            artifacts["uncertain_file"] = hard_name

        # Save skipped addresses for traceability (if any)
        if skipped_addresses:
            skipped_name = self._suffix_file(cfg.target, f"skipped-{cfg.round_name}")
            pd.DataFrame(skipped_addresses).to_csv(self.data_path / skipped_name, index=False)
            artifacts["skipped_addresses_file"] = skipped_name

        # Save remainder file (if any)
        if remainder:
            remaining_name = self._suffix_file(cfg.target, f"remaining-{cfg.round_name}")
            self._write_csv(tgt_df.loc[remainder], remaining_name)
            artifacts["remaining_file"] = remaining_name

        # If no pseudo rows, finish early
        if pseudo_df.empty:
            note = "No high-confidence pseudo labels (all rows still contain -1)."
            log = self._write_log(cfg, {
                "status": "no_pseudo",
                "counts": {"new_addresses": len(new_addresses), "pseudo_rows": 0, "uncertain_rows": int(uncertain_mask.sum())},
                "labels": label_names,
                "thresholds": {"low": cfg.low, "high": cfg.high},
                "files": artifacts,
            })
            return {"status": "no_pseudo", "note": note, "log_file": log, **artifacts}

        # Save pseudo-labeled file (only complete rows)
        pseudo_name = self._suffix_file(cfg.target, f"pseudo-{cfg.round_name}")
        self._write_csv(pseudo_df, pseudo_name)
        artifacts["pseudo_file"] = pseudo_name

        # Detect brand-new labels that were neither in source nor target but appear in predictions
        pred_only_labels = sorted(list(set(label_names) - set(all_labels)))
        if pred_only_labels:
            # Build a new ground truth for this round: combine source + pseudo (expand with new labels)
            ngt = self._expand_with_new_labels(pd.concat([src_df, pseudo_df], axis=0), pred_only_labels)
            # Drop rows where new labels are -1 to keep GT clean
            ngt_clean = ngt.copy()
            for col in pred_only_labels:
                ngt_clean = ngt_clean[ngt_clean[col] != -1]
            new_gt_name = self._suffix_file(cfg.source, f"new-gt-{cfg.round_name}")
            self._write_csv(ngt_clean, new_gt_name)
            artifacts["new_groundtruth_file"] = new_gt_name
            chosen_source = new_gt_name  # use newly created GT
            eval_source = cfg.eval_source or cfg.source
        else:
            # Merge source + pseudo (only addresses not in source)
            merge_df = self._safe_merge_source_pseudo(src_df, pseudo_df)
            merged_name = self._suffix_file(cfg.source, f"merged-{cfg.round_name}")
            self._write_csv(merge_df, merged_name)
            artifacts["merged_file"] = merged_name
            chosen_source = merged_name
            eval_source = cfg.eval_source or cfg.source

        # Optional: evaluate → retrain if improved
        metrics = None
        if cfg.do_train:
            try:
                metrics = self._evaluate_and_maybe_retrain(
                    source=chosen_source,
                    eval_source=eval_source,
                    test_size=cfg.test_size,
                    n_trials=cfg.n_trials
                )
            except Exception as e:
                logger.exception(f"evaluation/retrain failed: {e}")
                metrics = {"error": str(e)}

        # Log summary
        log = self._write_log(cfg, {
            "status": "ok",
            "counts": {
                "new_addresses": len(new_addresses),
                "pseudo_rows": int(len(pseudo_df)),
                "uncertain_rows": int(uncertain_mask.sum()),
                "skipped": int(len(skipped_addresses)),
                "remainder": int(len(remainder)),
            },
            "files": artifacts,
            "labels": label_names,
            "thresholds": {"low": cfg.low, "high": cfg.high},
            "quota_hit_midrun": quota_hit_midrun,
            "metrics": metrics,
        })

        return {"status": "ok", "log_file": log, **artifacts, "metrics": metrics}

    # ---------- helpers ----------

    def _numeric_label_columns(self, df: pd.DataFrame) -> List[str]:
        """Return columns that look like label columns (numeric)."""
        cols = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    def _read_csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.data_path / name, index_col=0)

    def _write_csv(self, df: pd.DataFrame, name: str) -> str:
        (self.data_path / name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path / name)
        return name

    def _suffix_file(self, fname: str, suffix: str) -> str:
        p = Path(fname)
        return f"{p.stem}-{suffix}{p.suffix}"

    def _resolve_label_names(self, pred_map: Dict[str, Dict], fallback: List[str]) -> List[str]:
        meta = MetaService.read_version()
        meta_labels = meta.get("label_names", [])
        if meta_labels:
            return meta_labels
        for v in pred_map.values():
            if "labels" in v:
                return list(v["labels"].keys())
        return fallback

    def _make_prob_and_hard(self, pred_map: Dict[str, Dict], label_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        probs, hards = {}, {}
        for addr, out in pred_map.items():
            lp = out.get("label_probs", {})
            lb = out.get("labels", {})
            probs[addr] = [float(lp.get(l, 0.0)) for l in label_names]
            hards[addr] = [int(lb.get(l, 0)) for l in label_names]
        prob_df = pd.DataFrame.from_dict(probs, orient="index", columns=label_names)
        hard_df = pd.DataFrame.from_dict(hards, orient="index", columns=label_names).astype(int)
        return prob_df, hard_df

    def _fill_with_thresholds(self, original: pd.DataFrame, probs: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
        out = original.copy()
        for col in out.columns:
            mask_unknown = (out[col] == -1)
            p = probs.loc[mask_unknown, col]
            out.loc[p.index[p <= low], col] = 0
            out.loc[p.index[p >= high], col] = 1
        return out.astype(int)

    def _expand_with_new_labels(self, df: pd.DataFrame, new_labels: List[str]) -> pd.DataFrame:
        for col in new_labels:
            if col not in df.columns:
                df[col] = -1
        return df[df.columns.sort_values()]

    def _safe_merge_source_pseudo(self, src: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
        # Keep source rows; add only truly new rows from pseudo
        add_rows = pseudo.loc[~pseudo.index.isin(src.index)]
        merged = pd.concat([src, add_rows], axis=0)
        # Align columns (fill missing with -1)
        all_cols = sorted(list(set(src.columns) | set(pseudo.columns)))
        merged = merged.reindex(columns=all_cols, fill_value=-1).astype(int)
        return merged

    def _evaluate_and_maybe_retrain(self, source: str, eval_source: str, test_size: float, n_trials: int) -> Dict:
        pipeline = TrainingPipeline(n_trials=n_trials)
        baseline = pipeline.eval.evaluate(test_size=test_size, source=eval_source, freeze_gru=True, freeze_sklearn=True)
        b = baseline.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")
        preview  = pipeline.trainer.train_preview(test_size=test_size, train_source=source, eval_source=eval_source)
        n = preview.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")
        improved = (b is None) or (n is not None and n >= b)
        if improved:
            pipeline.trainer.train_and_save(test_size=test_size, source=source)
            pipeline.full.run(source=source)
        return {"baseline_f1": float(b) if b is not None else None,
                "trial_f1": float(n) if n is not None else None,
                "improved": bool(improved),
                "final_version": MetaService.get_status().get("current_version")}

    def _write_log(self, cfg: SelfLearningConfig, payload: dict) -> str:
        """Write a JSON log entry for this self-learning run."""
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"self_learning_{cfg.round_name}_{ts}.json"
        log_path = self.logs_path / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            json.dump(payload, f, indent=2)

        return str(log_path)
