import numpy as np

def anomaly_fuse_predictions(preds: list, weights: list = None, threshold: float = 0.9):
    """
    Args:
        model_preds (List[np.ndarray]): List of shape (N, L) model output probabilities
        weights (List[float], optional): Fusion weights. Defaults to uniform.
        thresholds (List[float], optional): Per-label threshold. Defaults to 0.5.

    Returns:
        pred (np.ndarray): Binary (0/1) prediction of shape (N, L)
        confidence (np.ndarray): Fused confidence (probabilities) of shape (N, L)
    """
    fused_score = np.average(np.stack(preds, axis=1), axis=1, weights=weights)
    fused_flag = (fused_score > threshold).astype(int)

    return fused_flag, fused_score
