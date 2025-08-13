from typing import Dict, List, Tuple, Optional
import numpy as np

class Fusion:
    """Dict‑driven fusion for multilabel classification."""

    @staticmethod
    def fuse(
        model_probs: Dict[str, np.ndarray],   # {"general": (N,L), "sol": (N,L), "opcode": (N,L), "gru": (N,L)}
        label_names: List[str],
        weights: Optional[Dict[str, float]] = None,   # {"general":0.3, ...}
        thresholds: Optional[Dict[str, float]] = None # {"Mint":0.5, "Leak":0.55, ...}
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not model_probs:
            raise ValueError("model_probs is empty")

        N, L = next(iter(model_probs.values())).shape
        for k, arr in model_probs.items():
            if arr.shape != (N, L):
                raise ValueError(f"shape mismatch for '{k}': {arr.shape} != {(N, L)}")

        # weights
        if not weights:
            weights = {k: 1.0 for k in model_probs.keys()}
        s = sum(float(v) for v in weights.values()) or 1.0
        weights = {k: float(v)/s for k, v in weights.items()}

        fused = np.zeros((N, L), dtype=float)
        for name, probs in model_probs.items():
            fused += probs * float(weights.get(name, 0.0))

        # thresholds per label
        if not thresholds:
            thr_vec = np.full((L,), 0.5, dtype=float)
        else:
            thr_vec = np.array([float(thresholds.get(lbl, 0.5)) for lbl in label_names], dtype=float)

        pred = (fused >= thr_vec[None, :]).astype(int)
        return pred, fused
