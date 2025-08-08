# utils/training/evaluate_without_optuna.py

import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import load_model

from backend.utils.training.tuning import get_train_test_group
from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, GROUND_TRUTH_FILE
from backend.utils.training.training_objectives import training_gru
from backend.utils.predict.fusion import fuse_predictions
from backend.utils.logger import get_logger

logger = get_logger("eval_no_optuna")

def evaluate_with_current_models(test_size=0.2, source=GROUND_TRUTH_FILE):
    """
    Load CURRENT models, refit them on the new 80%, evaluate on 20%, and
    fuse using the *saved* fusion weights/thresholds. No Optuna anywhere.
    Does NOT overwrite saved model files.
    Returns a meta dict shaped like version.json with only eval metrics updated.
    """
    data = get_train_test_group(source=source, test_size=test_size)
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # Load current version metadata (for filenames + saved fusion weights/thresholds)
    with open(CURRENT_MODEL_PATH / "version.json", "r") as f:
        version_meta = json.load(f)

    # --- Collect per-model probabilities on X_test ---
    model_preds = {}
    clf_summary = version_meta.get("clf_model_summary", {})

    for key, meta in clf_summary.items():
        if "filename" not in meta:
            continue

        field = meta["field"]  # e.g., X_feature, X_code, X_opcode_seq, X_timeline_seq
        X_train = data.get(f"{field}_train")
        X_test  = data.get(f"{field}_test")
        file_path = CURRENT_MODEL_PATH / meta["filename"]

        if meta["filename"].endswith(".keras"):
            # Keras model (GRU classifier)
            model = load_model(file_path)
            # retrain on 80% with saved params (epochs/batch_size)
            params = meta.get("params", {"epochs": 10, "batch_size": 64})
            model, _ = training_gru(model, X_train, X_test, y_train.values, y_test.values, params)
            probas = model.predict(X_test)  # shape: (n_samples, n_labels)
            y_pred_bin = (probas > 0.5).astype(int)
            meta["f1_score"] = f1_score(y_test.values, y_pred_bin, average='macro', zero_division=0)
            model_preds[key] = probas
        else:
            # sklearn pipeline
            model = joblib.load(file_path)
            model.fit(X_train, y_train)  # fit on 80%
            # proba list per label → stack into (n_samples, n_labels)
            probas_list = model.predict_proba(X_test)
            probas = np.array([p[:, 1] for p in probas_list]).T
            preds = model.predict(X_test)
            meta["f1_score"] = f1_score(y_test, preds, average='macro', zero_division=0)
            model_preds[key] = probas

    # --- Fuse using saved weights/thresholds (no Optuna) ---
    fusion_saved = clf_summary.get("fusion_model", {})
    weights = fusion_saved.get("weights")
    thresholds = fusion_saved.get("thresholds")

    # Fallbacks if not present
    if weights is None:
        weights = [1.0] * len([k for k in model_preds.keys() if k != "fusion_model"])
    if thresholds is None:
        thresholds = [0.5] * y_test.shape[1]

    # Preserve model order: all non-fusion keys in insertion order
    pred_stack = [model_preds[k] for k in clf_summary.keys() if k != "fusion_model" and k in model_preds]
    y_pred, _ = fuse_predictions(pred_stack, weights=weights, thresholds=thresholds)

    # --- Reports (to logs current) ---
    report = classification_report(y_test.values, y_pred, output_dict=True, zero_division=0)
    with open(CURRENT_TRAINING_LOG_PATH / "classification_report_80_20.json", "w") as f:
        json.dump(report, f, indent=2)

    fusion_f1 = f1_score(y_test.values, y_pred, average='macro', zero_division=0)

    # Return a “trial” meta that looks like version.json enough for comparison
    return {
        "clf_model_summary": {
            **clf_summary,
            "fusion_model": {
                **fusion_saved,
                "f1_score": fusion_f1
            }
        },
        "label_names": y_train.columns.tolist(),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "notes": "80/20 evaluation using current saved models/params; no Optuna.",
    }
