from backend.core.meta_service import get_status
from backend.utils.training.tuning import train_and_save_best_model
from backend.utils.training.full_training import full_training

def train_pipeline():
    status = get_status()

    if status["current_version"] == "Not trained yet":
        # Step1: Find the best model from optuna and spilt-train and test 80 / 20, 20 is old ground truth, and save evaluation metrics to logs/training
        res = train_and_save_best_model()
        # Step2: retrain 100% dataset
        # Step4: save the model back to CURRENT_MODEL_PATH
        # Last Step: return the meta info of the model
        return res
    else:
        # Step1 check f1-score of the current model
        # - train 80% of the new dataset and 20% is old ground truth, and save evaluation metrics to logs/training
        # Step2: if f1-score is not good,
        # - move the current model to BACKUP_MODEL_PATH
        # - Find the best model from optuna and spilt-train and test 80 / 20, 20 is old ground truth, and save evaluation metrics to logs/training
        # - retrain 100% dataset
        # - save the model back to CURRENT_MODEL_PATH
        # Step3: if it is good, return the meta info of the model
        pass

    return {}
