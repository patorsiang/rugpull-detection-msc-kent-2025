import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend.core.meta_service import MetaService
from backend.utils.constants import CURRENT_MODEL_PATH
from backend.utils.predict.transform import FeatureAligner
from backend.utils.predict.fusion import Fusion
from backend.utils.predict.anomaly_fusion import AnomalyFusion
from backend.core.feature_service import extract_base_feature_from_address
from backend.utils.etherscan_quota import QuotaExceeded

class PredictService:
    """Single-request prediction across 4 models + anomaly fusion (thresholds overridable per call)."""

    def __init__(self):
        status = MetaService.get_status()
        if status.get("current_version") in (None, "Not trained yet"):
            raise Exception("you have to train model first.")

        meta = MetaService.read_version()
        self.label_names = meta.get("label_names", [])

        # fusion params
        clf = meta.get("clf_model_summary", {})
        anm = meta.get("anomaly_model_summary", {})
        self.clf_weights = clf.get("fusion_model", {}).get("weights", {}) or {}
        self.clf_thresholds = clf.get("fusion_model", {}).get("thresholds", {}) or {}
        self.anom_weights = anm.get("fusion_model", {}).get("weights", {}) or {}
        self.anom_threshold = float(anm.get("fusion_model", {}).get("threshold", 0.5))
        self.ae_threshold = anm.get("gru_model", {}).get("threshold", None)

        # load models
        self.general_clf = joblib.load(CURRENT_MODEL_PATH / "general_model.pkl")
        self.if_general  = joblib.load(CURRENT_MODEL_PATH / "if_general_model.pkl")
        self.sol_clf     = joblib.load(CURRENT_MODEL_PATH / "sol_model.pkl")
        self.if_sol      = joblib.load(CURRENT_MODEL_PATH / "if_sol_model.pkl")
        self.opcode_clf  = joblib.load(CURRENT_MODEL_PATH / "opcode_model.pkl")
        self.if_opcode   = joblib.load(CURRENT_MODEL_PATH / "if_opcode_model.pkl")
        self.gru_clf     = load_model(CURRENT_MODEL_PATH / "gru_model.keras")
        self.gru_ae      = load_model(CURRENT_MODEL_PATH / "gru_ae_model.keras")
        self.n_features  = int(self.gru_clf.input_shape[-1])

    @staticmethod
    def _clip01(x: float) -> float:
        try:
            return float(min(max(x, 0.0), 1.0))
        except Exception:
            return 0.5

    def _merge_label_thresholds(
        self, overrides: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Merge caller overrides with saved thresholds; clip to [0,1]; ignore unknown labels."""
        merged = dict(self.clf_thresholds or {})
        if overrides:
            for k, v in overrides.items():
                if k in self.label_names:
                    merged[k] = self._clip01(v)
        # Ensure all labels have a threshold (fallback 0.5)
        for lbl in self.label_names:
            if lbl not in merged:
                merged[lbl] = 0.5
        return merged

    def predict(
        self,
        addresses: List[str],
        label_thresholds: Optional[Dict[str, float]] = None,
        anomaly_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-head predictions and fuse with (optionally) custom thresholds.

        Returns (always the same envelope on success or error):
        {
            "status": "ok" | "error" | "quota_exhausted",
            "message": str (optional),
            "results": {
            address: {
                "labels": {label: 0/1},
                "label_probs": {label: float},
                "anomaly": 0/1,
                "anomaly_score": float
            },
            ...
            },
            "used_thresholds": {label: float},
            "used_anomaly_threshold": float
        }
        """
        # ---- thresholds (per call)
        used_thresholds = self._merge_label_thresholds(label_thresholds)
        used_anom_thr = (
            self._clip01(anomaly_threshold)
            if anomaly_threshold is not None
            else float(self.anom_threshold)
        )

        results: Dict[str, Any] = {}

        # Small helpers to keep shapes predictable
        def _stack_probas_from_multioutput(clf, X) -> np.ndarray:
            """
            clf is a MultiOutputClassifier or compatible; returns shape (n_samples, n_labels)
            with class-1 probabilities for each label.
            """
            probas_list = clf.predict_proba(X)  # list of [n_samples x 2] per label
            # If model returned a single array (binary one-vs-rest wrapper off), normalize to list
            if not isinstance(probas_list, (list, tuple)):
                # Expect shape (n_samples, n_classes) – take p(class=1)
                return np.asarray(probas_list)[:, 1].reshape(-1, 1)
            # Ensure each element has at least 2 columns; take column 1
            cols = []
            for p in probas_list:
                p = np.asarray(p)
                if p.ndim == 1:  # rare, but be defensive
                    # Treat as logit/score; squash via sigmoid to [0,1]
                    p = 1 / (1 + np.exp(-p))
                    cols.append(p.reshape(-1, 1))
                else:
                    if p.shape[1] == 1:
                        # Single-column probability; assume it's p(class=1)
                        cols.append(p[:, 0].reshape(-1, 1))
                    else:
                        cols.append(p[:, 1].reshape(-1, 1))
            return np.hstack(cols)  # (n_samples, n_labels)

        # AE / timeline defaults
        seq_len = int(getattr(self, "SEQ_LEN", 500))  # fallback if constant is not in scope
        n_feats = int(getattr(self, "n_features", 12))  # channels per timestep for the AE/GRU

        try:
            for addr in addresses:
                feature = extract_base_feature_from_address(addr)  # may raise QuotaExceeded

                # -------- General (tabular) head
                exclude = {"Label", "opcode_sequence", "timeline_sequence", "sourcecode"}
                Xf = pd.DataFrame([{k: v for k, v in feature.items() if k not in exclude}])

                Xf_clf = FeatureAligner.align_dataframe(Xf.copy(), self.general_clf)
                p_general = _stack_probas_from_multioutput(self.general_clf, Xf_clf)

                # IsolationForest for tabular anomaly (align to its fitted features)
                Xf_if = FeatureAligner.align_dataframe(Xf.copy(), self.if_general)
                an_gen = (self.if_general.predict(Xf_if) == -1).astype(int)  # shape (n_samples,)

                # -------- Source head
                xcode = [feature.get("sourcecode", "")]
                p_src = _stack_probas_from_multioutput(self.sol_clf, xcode)
                an_src = (self.if_sol.predict(xcode) == -1).astype(int)

                # -------- Opcode head
                xop = [feature.get("opcode_sequence", "")]
                p_opc = _stack_probas_from_multioutput(self.opcode_clf, xop)
                an_opc = (self.if_opcode.predict(xop) == -1).astype(int)

                # -------- Timeline head (GRU classifier + AE anomaly)
                raw_tl = feature.get("timeline_sequence", [])
                if not raw_tl:
                    # Ensure at least one timestep of zeros with correct channel count
                    raw_tl = [[0.0] * n_feats]
                # pad_sequences expects list of sequences (timesteps x features)
                Xtl = pad_sequences([raw_tl], maxlen=seq_len, padding="post", dtype="float32")

                # If GRU classifier outputs per-label probs with sigmoid
                p_time = np.asarray(self.gru_clf.predict(Xtl, verbose=0)).reshape(1, -1)
                if p_time.shape[1] != len(self.label_names):
                    # Defensive: if the GRU outputs a single score, broadcast or adjust
                    p_time = np.repeat(p_time, len(self.label_names)).reshape(1, -1)

                # AE reconstruction error -> anomaly
                recon = np.asarray(self.gru_ae.predict(Xtl, verbose=0))
                err = np.mean((Xtl - recon) ** 2, axis=(1, 2))  # shape (1,)
                ae_thr = (
                    float(self.ae_threshold)
                    if getattr(self, "ae_threshold", None) is not None
                    else float(np.percentile(err, 95))
                )
                an_time = (err > ae_thr).astype(int)  # shape (1,)

                # -------- Fusion
                # Classifier fusion (per label)
                pred, prob = Fusion.fuse(
                    {"general": p_general, "sol": p_src, "opcode": p_opc, "gru": p_time},
                    self.label_names,
                    self.clf_weights,
                    used_thresholds,
                )  # pred, prob each shape (1, n_labels)

                # Anomaly fusion (heads are binary anomaly flags; score depends on your implementation)
                an_flag, an_score = AnomalyFusion.fuse(
                    {
                        "if_general": an_gen.reshape(-1, 1),   # ensure 2D for weight broadcast
                        "if_sol": an_src.reshape(-1, 1),
                        "if_opcode": an_opc.reshape(-1, 1),
                        "ae_timeline": an_time.reshape(-1, 1),
                    },
                    self.anom_weights,
                    used_anom_thr,
                )  # shapes (1, 1)

                results[addr] = {
                    "labels": {lbl: int(v) for lbl, v in zip(self.label_names, pred[0].tolist())},
                    "label_probs": {lbl: float(v) for lbl, v in zip(self.label_names, prob[0].tolist())},
                    "anomaly": int(np.asarray(an_flag).ravel()[0]),
                    "anomaly_score": float(np.asarray(an_score).ravel()[0]),
                }

            return {
                "status": "success",
                "results": results,
                "used_thresholds": {lbl: float(used_thresholds.get(lbl, 0.5)) for lbl in self.label_names},
                "used_anomaly_threshold": float(used_anom_thr),
            }

        except QuotaExceeded as e:
            # Keep partial results so callers can decide what to do
            return {
                "status": "quota_exhausted",
                "message": str(e),
                "result": results,
                "used_thresholds": {lbl: float(used_thresholds.get(lbl, 0.5)) for lbl in self.label_names},
                "used_anomaly_threshold": float(used_anom_thr),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "result": results,
                "used_thresholds": {lbl: float(used_thresholds.get(lbl, 0.5)) for lbl in self.label_names},
                "used_anomaly_threshold": float(used_anom_thr),
            }
