from backend.core.meta_service import get_status
from backend.utils.training.tuning import train_and_save_best_model
from backend.utils.training.full_training import full_training
from backend.utils.training.evaluation import evaluate_with_current_models
from backend.utils.constants import CURRENT_MODEL_PATH, GROUND_TRUTH_FILE, N_TRIALS
from backend.utils.logger import logging

import json

logger = logging.getLogger(__name__)

def train_pipeline(source=GROUND_TRUTH_FILE, eval_source=GROUND_TRUTH_FILE, test_size=0.2, N_TRIALS=N_TRIALS):
    status = get_status()

    # First-time training
    if status.get("current_version") in [None, "Not trained yet"]:
        # Step1: run Optuna model selection on 80/20 and log results
        meta = train_and_save_best_model(test_size=test_size, source=source, N_TRIALS=N_TRIALS)
        # Step2: finalise (optional): retrain on 100% using saved best params/weights
        full_training(source=source)
        return meta

    baseline_meta = evaluate_with_current_models(
        test_size=test_size,
        source=eval_source,
        freeze_gru=True,
        freeze_sklearn=True,
    )

    baseline_f1 = baseline_meta.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

    logger.debug(f"[PIPELINE] Baseline fusion F1 (80/20, frozen): {baseline_f1}")

    # Step1: try new 80/20
    trial_meta = evaluate_with_current_models(
        test_size=test_size,
        source=source,
        freeze_gru=False,             # or False if you want to refit GRU on the 80%
        freeze_sklearn=False,        # refit sklearn on 80%
    )

    new_f1 = trial_meta.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

    logger.debug(f"[PIPELINE] Trial fusion F1 (80/20, refit sklearn): {new_f1}")

    # --- Decide ---
    improved = (baseline_f1 is None) or (new_f1 is not None and new_f1 >= baseline_f1)
    logger.debug(f"[PIPELINE] Improved vs baseline on same split? {improved}")

    if improved:
        # Promote the new config and re-train on all data
        meta = train_and_save_best_model(test_size=test_size, source=source, N_TRIALS=N_TRIALS)

        full_training(source=source)

    with open(CURRENT_MODEL_PATH / "version.json", "r") as f:
        current_meta = json.load(f)
    return current_meta
