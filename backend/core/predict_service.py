import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend.core.meta_service import MetaService
from backend.utils.constants import CURRENT_MODEL_PATH, SEQ_LEN
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

        Args:
            addresses: list of contract addresses to score
            label_thresholds: partial/complete per-label thresholds to override saved values
            anomaly_threshold: optional override for anomaly fusion threshold

        Returns:
            Dict[address] -> {
                labels: {label: 0/1},
                label_probs: {label: float},
                anomaly: 0/1,
                anomaly_score: float,
                used_thresholds: {label: float},     # echo back for traceability
                used_anomaly_threshold: float
            }
        """
        # Prepare thresholds (per call)
        used_thresholds = self._merge_label_thresholds(label_thresholds)
        used_anom_thr = self._clip01(anomaly_threshold) if anomaly_threshold is not None else float(self.anom_threshold)

        results: Dict[str, Any] = {}
        try:
            for addr in addresses:
                feature = extract_base_feature_from_address(addr)  # may raise QuotaExceeded

                # static
                exclude = {"Label","opcode_sequence","timeline_sequence","sourcecode"}
                Xf = pd.DataFrame([{k:v for k,v in feature.items() if k not in exclude}])
                Xf_clf = FeatureAligner.align_dataframe(Xf.copy(), self.general_clf)
                p_general = np.array([p[:,1] for p in self.general_clf.predict_proba(Xf_clf)]).T
                an_gen    = (self.if_general.predict(FeatureAligner.align_dataframe(Xf.copy(), self.if_general)) == -1).astype(int).reshape(-1)

                # source
                xcode = [feature.get("sourcecode","")]
                p_src = np.array([p[:,1] for p in self.sol_clf.predict_proba(xcode)]).T
                an_src= (self.if_sol.predict(xcode) == -1).astype(int).reshape(-1)

                # opcode
                xop   = [feature.get("opcode_sequence","")]
                p_opc = np.array([p[:,1] for p in self.opcode_clf.predict_proba(xop)]).T
                an_opc= (self.if_opcode.predict(xop) == -1).astype(int).reshape(-1)

                # timeline
                raw_tl = feature.get("timeline_sequence", [])
                if not raw_tl:
                    raw_tl = [[0.0]*self.n_features]
                Xtl = pad_sequences([raw_tl], maxlen=SEQ_LEN, padding="post", dtype="float32")
                p_time = self.gru_clf.predict(Xtl, verbose=0)
                recon  = self.gru_ae.predict(Xtl, verbose=0)
                err = np.mean((Xtl - recon)**2, axis=(1,2))
                thr = float(self.ae_threshold) if self.ae_threshold is not None else float(np.percentile(err,95))
                an_time = (err > thr).astype(int).reshape(-1)

                # fusion with custom thresholds
                pred, prob = Fusion.fuse(
                    {"general":p_general, "sol":p_src, "opcode":p_opc, "gru":p_time},
                    self.label_names, self.clf_weights, used_thresholds
                )
                an_flag, an_score = AnomalyFusion.fuse(
                    {"if_general":an_gen, "if_sol":an_src, "if_opcode":an_opc, "ae_timeline":an_time},
                    self.anom_weights, used_anom_thr
                )

                results[addr] = {
                    "labels": {lbl:int(v) for lbl, v in zip(self.label_names, pred[0].tolist())},
                    "label_probs": {lbl:float(v) for lbl, v in zip(self.label_names, prob[0].tolist())},
                    "anomaly": int(an_flag[0]),
                    "anomaly_score": float(an_score[0]),
                    "used_thresholds": {lbl: float(used_thresholds.get(lbl, 0.5)) for lbl in self.label_names},
                    "used_anomaly_threshold": float(used_anom_thr),
                }
                return results
        except QuotaExceeded as e:
        # Should already be handled in service, but keep a defensive catch.
            return {"status": "quota_exhausted", "message": str(e), "result": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}
