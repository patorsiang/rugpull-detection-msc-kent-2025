import numpy as np
from sklearn.metrics import f1_score

def get_best_thresholds(y_pred_prob, y_test):
    best_thresholds = []

    for i in range(y_pred_prob.shape[1]):
        best_f1 = 0
        best_thresh = 0.5  # default
        for t in np.arange(0.0, 1.01, 0.01):
            y_pred = (y_pred_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_test.iloc[:, i], y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds.append(best_thresh)

    best_thresholds = np.array(best_thresholds)
    return best_thresholds
