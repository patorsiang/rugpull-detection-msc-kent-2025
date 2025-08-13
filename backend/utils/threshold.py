from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score

class ThresholdTuner:
    """Independent per‑label threshold scan, returns dicts keyed by label."""

    @staticmethod
    def tune(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        L = y_true.shape[1]
        best_t, best_s = {}, {}
        for i in range(L):
            yy = y_true[:, i]; pp = y_prob[:, i]
            grid = np.linspace(0, 1, 101)
            scores = [f1_score(yy, (pp >= t).astype(int), zero_division=0) for t in grid]
            j = int(np.argmax(scores))
            best_t[label_names[i]] = float(grid[j])
            best_s[label_names[i]] = float(scores[j])
        return best_t, best_s
