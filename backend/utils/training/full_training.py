import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, f1_score
import optuna

from backend.utils.training.tuning import get_train_test_group
from backend.utils.constants import CURRENT_MODEL_PATH
from backend.utils.training.training_objectives import training_gru, fusion_objective, anomaly_fusion_objective
# from backend.utils.logger import get_logger
from backend.utils.predict.fusion import fuse_predictions
from backend.utils.constants import N_TRIALS, CURRENT_TRAINING_LOG_PATH, GROUND_TRUTH_FILE
from backend.utils.training.tuning import plot_multilabel_confusion_matrix, plot_anomaly_distribution
from backend.utils.training.backup import backup_model_and_logs

#logger = get_logger('train_pipeline')

def full_training(source=GROUND_TRUTH_FILE):
    backup_model_and_logs()
    dataset = get_train_test_group(source, 0.0)

    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    version_meta = json.load(open(CURRENT_MODEL_PATH / 'version.json', 'r'))

    model_preds = []

    for model_meta in list(version_meta['clf_model_summary'].values()):
        if 'filename' in model_meta:
            X_train = dataset[f"{model_meta['field']}_train"]
            X_test = dataset[f"{model_meta['field']}_test"]
            file_path = CURRENT_MODEL_PATH / model_meta['filename']
            if '.keras' in model_meta['filename']:
                model =  load_model(file_path)
                model.summary()
                model, _ = training_gru(model, X_train, X_test, y_train, y_test, model_meta["params"])
                model.save(file_path)
                probas = model.predict(X_train)
                model_preds.append(probas)
                y_pred_bin = (probas > 0.5).astype(int)
                model_meta["f1_score"] = f1_score(y_test.values, y_pred_bin, average='macro', zero_division=0)
            else:
                model = joblib.load(file_path)
                model.fit(X_train, y_train)
                probas = model.predict_proba(X_test)
                probas = np.array([p[:, 1] for p in probas]).T

                model_preds.append(probas)
                joblib.dump(model, file_path)
                preds = model.predict(X_test)
                model_meta["f1_score"] = f1_score(y_test, preds, average='macro', zero_division=0)

    isolation_preds = []
    y_anomaly_test = (y_test.sum(axis=1) > 0).astype(int).values

    for model_meta in list(version_meta['anomaly_model_summary'].values()):
        if 'filename' in model_meta:
            X_train = dataset[f"{model_meta['field']}_train"]
            X_test = dataset[f"{model_meta['field']}_test"]
            file_path = CURRENT_MODEL_PATH / model_meta['filename']
            if '.keras' in model_meta['filename']:
                model =  load_model(file_path)
                model.summary()
                model, _ = training_gru(model, X_train, X_test, X_train, X_test, model_meta["params"])
                model.save(file_path)
                # Calculate reconstruction errors
                recon = model.predict(X_test)
                reconstruction_errors = np.mean((X_test - recon)**2, axis=(1, 2))

                reconstruction_errors = np.nan_to_num(
                    reconstruction_errors, nan=0.0, posinf=1e12, neginf=-1e12
                )

                # Optional: Find a threshold (e.g. 95th percentile) to mark anomaly
                threshold = np.percentile(reconstruction_errors, 95)
                anomaly_flags = (reconstruction_errors > threshold).astype(int)
                isolation_preds.append(anomaly_flags)  # Now len = 4
                model_meta['MSE'] = float(np.mean(reconstruction_errors))
                model_meta['threshold'] = threshold

            else:
            # if '.pkl' in model_meta['filename']:
                model = joblib.load(file_path)
                model.fit(X_train)
                joblib.dump(model, file_path)
                pred = model.predict(X_test)
                preds_bin = (pred == -1).astype(int)

                isolation_preds.append(preds_bin)  # anomaly = 1

                y_anomaly = (y_test.sum(axis=1) > 0).astype(int)

                model_meta['f1_score'] = f1_score(y_anomaly, preds_bin, zero_division=0)

    #### 🔗 Fusion #####
    # logger.info("🔗 Running Optuna for Fusion...")
    study_fusion = optuna.create_study(direction="maximize")
    study_fusion.optimize(lambda trial: fusion_objective(trial, model_preds, y_test.values), n_trials=N_TRIALS)

    best_weights = [study_fusion.best_params[f"w{i}"] for i in range(len(model_preds))]
    best_thresholds = [study_fusion.best_params[f"t{i}"] for i in range(y_test.shape[1])]
    y_pred, _ = fuse_predictions(model_preds, weights=best_weights, thresholds=best_thresholds)

    version_meta['clf_model_summary']['fusion_model'] = {
        "f1_score": study_fusion.best_value,
        "weights": best_weights,
        "thresholds": best_thresholds,
    }

    #### 📊 Logging #####
    # logger.info("📊 Saving classification report and confusion matrix...")
    report = classification_report(y_test.values, y_pred, output_dict=True, zero_division=0)
    report_path = CURRENT_TRAINING_LOG_PATH / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    cm_path = CURRENT_TRAINING_LOG_PATH / "confusion_matrix.png"
    plot_multilabel_confusion_matrix(y_test.values, y_pred, labels=y_train.columns.tolist(), save_path=cm_path)

    # logger.info("🔗 Running Optuna for Anomaly Fusion...")

    study_anomaly = optuna.create_study(direction="maximize")
    study_anomaly.optimize(lambda trial: anomaly_fusion_objective(trial, isolation_preds, y_anomaly_test), n_trials=N_TRIALS)

    # Save results
    anomaly_weights = [study_anomaly.best_params[f"w{i}"] for i in range(len(isolation_preds))]
    anomaly_threshold = study_anomaly.best_params["threshold"]


    version_meta['anomaly_model_summary']['fusion_model'] = {
        "f1_score": study_anomaly.best_value,
        "weights": anomaly_weights,
        "threshold": anomaly_threshold
    }

    report_anomaly = classification_report(y_anomaly_test, (np.average(np.stack(isolation_preds, axis=1), axis=1, weights=anomaly_weights) > anomaly_threshold).astype(int), output_dict=True, zero_division=0)
    with open(CURRENT_TRAINING_LOG_PATH / "anomaly_report.json", "w") as f:
        json.dump(report_anomaly, f, indent=2)

    fused_score = np.average(np.stack(isolation_preds, axis=1), axis=1, weights=anomaly_weights)

    plot_anomaly_distribution(fused_score, anomaly_threshold, CURRENT_TRAINING_LOG_PATH / "anomaly_fusion_distribution.png")

    version_meta["current_version"] = version_meta.get("current_version", "") + '_1'
    version_meta['label_names'] = y_train.columns.tolist()
    version_meta['train_size'] = len(y_train)
    version_meta['test_size'] = len(y_test)
    version_meta['notes'] = f"Trained with whole dataset"

    # Save version timestamp
    version_path = CURRENT_MODEL_PATH / "version.json"
    version_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    with open(version_path, "w") as f:
        json.dump(version_meta, f, indent=4)

    return version_meta
