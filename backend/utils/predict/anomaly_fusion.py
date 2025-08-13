from typing import Dict, Tuple, Optional
import numpy as np

class AnomalyFusion:
    """Dict‑driven late fusion for anomaly flags (0/1 vectors)."""

    @staticmethod
    def fuse(
        preds: Dict[str, np.ndarray],                # {"if_general": (N,), "if_sol": (N,), "if_opcode": (N,), "ae_timeline": (N,)}
        weights: Optional[Dict[str, float]] = None,  # dict per source
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not preds:
            raise ValueError("preds is empty")

        N = None
        mats = {}
        for k, v in preds.items():
            a = np.asarray(v).reshape(-1).astype(float)
            if N is None:
                N = a.shape[0]
            if a.shape[0] != N:
                raise ValueError(f"Anomaly length mismatch for '{k}': {a.shape[0]} vs {N}")
            mats[k] = a

        if not weights:
            weights = {k: 1.0 for k in mats.keys()}
        s = sum(float(x) for x in weights.values()) or 1.0
        weights = {k: float(v)/s for k, v in weights.items()}

        fused = np.zeros((N,), dtype=float)
        for name, vec in mats.items():
            fused += vec * float(weights.get(name, 0.0))

        flag = (fused > float(threshold)).astype(int)
        return flag, fused
