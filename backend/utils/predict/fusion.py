import numpy as np

def fuse_predictions(model_preds: list, weights: list = None, thresholds: list = None):
    """
    Args:
        model_preds (List[np.ndarray]): List of shape (N, L) model output probabilities
        weights (List[float], optional): Fusion weights. Defaults to uniform.
        thresholds (List[float], optional): Per-label threshold. Defaults to 0.5.

    Returns:
        pred (np.ndarray): Binary (0/1) prediction of shape (N, L)
        confidence (np.ndarray): Fused confidence (probabilities) of shape (N, L)
    """
    if weights is None:
        weights = [1.0 / len(model_preds)] * len(model_preds)

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Weighted average of predictions
    fused = np.zeros_like(model_preds[0])
    for pred, w in zip(model_preds, weights):
        fused += pred * w

    # Thresholding
    if thresholds is None:
        thresholds = [0.5] * fused.shape[1]

    pred = (fused >= thresholds).astype(int)
    return pred, fused
