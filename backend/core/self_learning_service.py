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


@dataclass
class SelfLearningConfig:
    source: str                  # e.g. "groundtruth.csv"
    target: str                  # e.g. "unlabeled.csv"
    round_name: str              # e.g. "r01"
    low: float = 0.10            # <= low -> 0
    high: float = 0.90           # >= high -> 1
    eval_source: Optional[str] = None  # baseline eval source (defaults to source)
    test_size: float = 0.2       # for evaluation
    n_trials: int = 50           # for (re)training when improved
    do_train: bool = True        # whether to run the eval/compare/retrain step


class SelfLearningService:
    """Run pseudo-labeling using current models and manage merges/new GT."""

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
        new_addresses = sorted(list(set(tgt_df.index) - set(src_df.index)))
        if not new_addresses:
            note = "No new addresses to process; target is already contained in source."
            log = self._write_log(cfg, {"status": "no_change", "note": note})
            return {
                "status": "no_change",
                "note": note,
                "log_file": log,
                "source_file": cfg.source,
                "target_file": cfg.target,
            }

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

        # Predict only for the truly new addresses
        predictor = PredictService()
        pred_map = predictor.predict(new_addresses)  # {address: {...}}

        # Build prob/label frames from predictions
        label_names = self._resolve_label_names(pred_map, all_labels)
        prob_df, hard_df = self._make_prob_and_hard(pred_map, label_names)

        # Only fill positions where target currently == -1
        # And apply 10%/90% gating: <= low -> 0, >= high -> 1, else keep -1
        filled_df = self._fill_with_thresholds(
            original=tgt_df.loc[new_addresses, label_names].copy(),
            probs=prob_df.loc[new_addresses, label_names],
            low=cfg.low,
            high=cfg.high,
        )

        # Pseudo-labeled (keep only rows where no -1 remains)
        pseudo_df = filled_df[(filled_df != -1).all(axis=1)].copy()

        artifacts = {
            "pseudo_file": None,
            "merged_file": None,
            "new_groundtruth_file": None,
            "target_augmented": cfg.target if new_labels_in_src else None,
        }

        if pseudo_df.empty:
            note = "No high-confidence pseudo labels (all rows still contain -1)."
            log = self._write_log(cfg, {
                "status": "no_pseudo",
                "counts": {"new_addresses": len(new_addresses), "pseudo_rows": 0},
                "labels": label_names,
                "note": note,
            })
            return {"status": "no_pseudo", "note": note, "log_file": log, **artifacts}

        # Save pseudo-labeled file (only complete rows)
        pseudo_name = self._suffix_file(cfg.target, f"pseudo-{cfg.round_name}")
        self._write_csv(pseudo_df, pseudo_name)
        artifacts["pseudo_file"] = pseudo_name

        # Detect brand-new labels that were neither in source nor target but appear in predictions
        # (e.g., model now emits "Trapdoor" but CSVs don’t have it)
        pred_only_labels = sorted(list(set(label_names) - set(all_labels)))
        if pred_only_labels:
            # Build a new ground truth for this round: combine source + pseudo
            ngt = self._expand_with_new_labels(pd.concat([src_df, pseudo_df], axis=0), pred_only_labels)
            # Drop -1 rows/values for new labels to keep GT clean
            ngt_clean = ngt.copy()
            for col in pred_only_labels:
                ngt_clean = ngt_clean[ngt_clean[col] != -1]
            new_gt_name = self._suffix_file(cfg.source, f"new-gt-{cfg.round_name}")
            self._write_csv(ngt_clean, new_gt_name)
            artifacts["new_groundtruth_file"] = new_gt_name

            # If new GT is created, spec says: "no need to merge"
            eval_source = cfg.eval_source or cfg.source
            chosen_source = new_gt_name  # use the newly created ground truth
        else:
            # Merge source + pseudo (only addresses not in source)
            merge_df = self._safe_merge_source_pseudo(src_df, pseudo_df)
            merged_name = self._suffix_file(cfg.source, f"merged-{cfg.round_name}")
            self._write_csv(merge_df, merged_name)
            artifacts["merged_file"] = merged_name
            eval_source = cfg.eval_source or cfg.source
            chosen_source = merged_name

        # Optional: evaluate → retrain if improved
        metrics = None
        if cfg.do_train:
            metrics = self._evaluate_and_maybe_retrain(
                source=chosen_source,
                eval_source=eval_source,
                test_size=cfg.test_size,
                n_trials=cfg.n_trials
            )

        # Log summary
        log = self._write_log(cfg, {
            "status": "ok",
            "counts": {
                "new_addresses": len(new_addresses),
                "pseudo_rows": int(len(pseudo_df)),
            },
            "files": artifacts,
            "labels": label_names,
            "thresholds": {"low": cfg.low, "high": cfg.high},
            "metrics": metrics,
        })

        return {"status": "ok", "log_file": log, **artifacts}

    # ---------- Internals ----------

    def _read_csv(self, name: str) -> pd.DataFrame:
        df = pd.read_csv(self.data_path / name, index_col=0)
        # Ensure int labels if possible, preserve -1
        return df.astype(int, errors="ignore")

    def _write_csv(self, df: pd.DataFrame, name: str) -> str:
        df.to_csv(self.data_path / name)
        return name

    def _suffix_file(self, fname: str, suffix: str) -> str:
        p = Path(fname)
        return f"{p.stem}-{suffix}{p.suffix}"

    def _resolve_label_names(self, pred_map: Dict[str, Dict], fallback: List[str]) -> List[str]:
        # Use meta label names if available, else keys from first result, else fallback
        meta = MetaService.read_version()
        meta_labels = meta.get("label_names", [])
        if meta_labels:
            return meta_labels
        for v in pred_map.values():
            if "labels" in v:
                return list(v["labels"].keys())
        return fallback

    def _make_prob_and_hard(self, pred_map: Dict[str, Dict], label_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        probs = {}
        hards = {}
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
            # <= low => 0 ; >= high => 1 ; else keep -1
            set_zero = p.index[p <= low]
            set_one  = p.index[p >= high]
            out.loc[set_zero, col] = 0
            out.loc[set_one, col]  = 1
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

        # 1) Frozen baseline on eval_source using CURRENT saved models
        baseline = pipeline.eval.evaluate(test_size=test_size, source=eval_source, freeze_gru=True, freeze_sklearn=True)
        b = baseline.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

        # 2) Tuned "preview" on source (Optuna), NO SAVING
        preview  = pipeline.trainer.train_preview(test_size=test_size, train_source=source, eval_source=eval_source)
        n = preview.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

        improved = (b is None) or (n is not None and n >= b)

        if improved:
            # 3) Commit: run the real save + full retrain on 100%
            pipeline.trainer.train_and_save(test_size=test_size, source=source)
            pipeline.full.run(source=source)

        return {
            "baseline_f1": float(b) if b is not None else None,
            "trial_f1": float(n) if n is not None else None,
            "improved": bool(improved),
            "final_version": MetaService.get_status().get("current_version"),
        }


    def _write_log(self, cfg: SelfLearningConfig, payload: Dict) -> str:
        log_name = f"selflearn_{cfg.round_name}.json"
        data = {
            "config": {
                "source": cfg.source, "target": cfg.target, "round": cfg.round_name,
                "low": cfg.low, "high": cfg.high,
                "eval_source": cfg.eval_source or cfg.source,
                "test_size": cfg.test_size, "n_trials": cfg.n_trials, "do_train": cfg.do_train,
            },
            **payload,
        }
        with open(self.logs_path / log_name, "w") as f:
            json.dump(data, f, indent=2)
        return log_name
