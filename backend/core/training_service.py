from backend.core.meta_service import get_status
from backend.utils.training.tuning import train_and_save_best_model
from backend.utils.training.full_training import full_training
from backend.utils.training.evaluation import evaluate_with_current_models
from backend.utils.constants import CURRENT_MODEL_PATH, GROUND_TRUTH_FILE, N_TRIALS
import json

def train_pipeline(source=GROUND_TRUTH_FILE, test_size=0.2, N_TRIALS=N_TRIALS):
    status = get_status()

    # First-time training
    if status.get("current_version") in [None, "Not trained yet"]:
        # Step1: run Optuna model selection on 80/20 and log results
        meta = train_and_save_best_model(test_size=test_size, source=source, N_TRIALS=N_TRIALS)
        # Step2: finalise (optional): retrain on 100% using saved best params/weights
        full_training()
        return meta

    # Incremental: compare new 80/20 against current
    # NOTE: assumes CURRENT_MODEL_PATH/version.json exists
    with open(CURRENT_MODEL_PATH / "version.json", "r") as f:
        current_meta = json.load(f)
    current_f1 = current_meta.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

    # Step1: try new 80/20
    trial_meta = evaluate_with_current_models(test_size=test_size, source=source)

    new_f1 = trial_meta.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")

    # Step2: if improved → finalise on 100% and keep; else keep current
    if current_f1 is None or (new_f1 is not None and new_f1 >= current_f1):
        # Promote the new config and re-train on all data
        meta = train_and_save_best_model(test_size=test_size, source=source, N_TRIALS=N_TRIALS)

        full_training()
        return trial_meta
    else:
        # No improvement; keep current, return it
        return current_meta
