import pandas as pd
from sklearn.model_selection import train_test_split

import os
import json
import optuna
import joblib
import numpy as np
import matplotlib
import seaborn as sns
from datetime import datetime
matplotlib.use('Agg')  # Must come before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend.utils.training.training_objectives import (general_objective,
                                                        gru_objective,
                                                        fusion_objective,
                                                        build_model_by_name,
                                                        build_gru_model,
                                                        if_objective,
                                                        build_if_model_by_mode,
                                                        ae_objective,
                                                        build_ae_gru_model,
                                                        anomaly_fusion_objective,
                                                        training_gru)
from backend.utils.predict.fusion import fuse_predictions

from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, N_TRIALS, SEQ_LEN, GROUND_TRUTH_FILE

from backend.core.dataset_service import get_full_dataset

# from backend.utils.logger import get_logger

from backend.utils.training.backup import backup_model_and_logs

def get_train_test_group(source=GROUND_TRUTH_FILE, test_size=0.2):
    dataset = get_full_dataset(filename=source)
    addresses = list(dataset.keys())

    # Split addresses
    if test_size and test_size > 0:
        addresses_train, addresses_test = train_test_split(addresses, test_size=test_size, random_state=42)
    else:
        # use ALL data both sides (for "train on 100%" stage)
        addresses_train = addresses
        addresses_test  = addresses

    # Prepare Y (multi-label DataFrame)
    y_train_df = pd.DataFrame([dataset[a].get("Label", {}) for a in addresses_train], index=addresses_train)
    y_test_df = pd.DataFrame([dataset[a].get("Label", {}) for a in addresses_test], index=addresses_test)

    # Prepare X - Text sequences
    X_opcode_seq_train = [dataset[a].get("opcode_sequence", "") for a in addresses_train]
    X_opcode_seq_test = [dataset[a].get("opcode_sequence", "") for a in addresses_test]

    X_timeline_seq_train = [dataset[a].get("timeline_sequence", []) for a in addresses_train]
    X_timeline_seq_test = [dataset[a].get("timeline_sequence", []) for a in addresses_test]

    X_code_train = [dataset[a].get("sourcecode", "") for a in addresses_train]
    X_code_test = [dataset[a].get("sourcecode", "") for a in addresses_test]

    # Prepare X - Tabular features
    exclude_keys = ["Label", "opcode_sequence", "timeline_sequence", "sourcecode"]
    X_feature_train_dicts = [
        {k: v for k, v in dataset[a].items() if k not in exclude_keys}
        for a in addresses_train
    ]
    X_feature_test_dicts = [
        {k: v for k, v in dataset[a].items() if k not in exclude_keys}
        for a in addresses_test
    ]

    X_feature_train_df = pd.DataFrame(X_feature_train_dicts, index=addresses_train)
    X_feature_test_df = pd.DataFrame(X_feature_test_dicts, index=addresses_test)

    # After creating both train and test feature DataFrames
    X_feature_train_df, X_feature_test_df = X_feature_train_df.align(X_feature_test_df, join="outer", axis=1, fill_value=0)


    # ✅ Ready to train with:
    # - X_feature_train_scaled, y_train_df (for ML)
    # - X_opcode_seq_train, X_code_train (for NLP)
    # - X_timeline_seq_train (for GRU)

    # Save or return if needed
    return {
        "X_opcode_seq_train": X_opcode_seq_train,
        "X_opcode_seq_test": X_opcode_seq_test,
        "X_timeline_seq_train": pad_sequences(X_timeline_seq_train, maxlen=SEQ_LEN, padding='post', dtype='float32'),
        "X_timeline_seq_test": pad_sequences(X_timeline_seq_test, maxlen=SEQ_LEN, padding='post', dtype='float32'),
        "X_code_train": X_code_train,
        "X_code_test": X_code_test,
        "X_feature_train": X_feature_train_df,
        "X_feature_test": X_feature_test_df,
        "y_train": y_train_df,
        "y_test": y_test_df,
    }


def plot_multilabel_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm_list = multilabel_confusion_matrix(y_true, y_pred)

    n = len(labels)
    _, axs = plt.subplots(1, n, figsize=(6 * n, 5))

    for i in range(n):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_list[i], display_labels=[f"Not {labels[i]}", labels[i]])
        disp.plot(cmap=plt.cm.Blues, ax=axs[i] if n > 1 else axs, values_format="d", colorbar=False)
        axs[i].set_title(labels[i])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_anomaly_distribution(fused_score, anomaly_threshold, save_path=None):
    plt.figure(figsize=(8, 4))
    sns.histplot(fused_score, kde=True, bins=50)
    plt.axvline(anomaly_threshold, color='red', linestyle='--', label=f'Threshold: {anomaly_threshold:.2f}')
    plt.title("Anomaly Fusion Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#logger = get_logger("train_and_save")

def train_and_save_best_model(test_size=0.2, N_TRIALS=N_TRIALS, source=GROUND_TRUTH_FILE):
    backup_model_and_logs()
    # logger.info("📦 Loading data and splitting train/test...")
    # Get train/test split
    data = get_train_test_group(source=source, test_size=test_size)

    y_train = data["y_train"]
    y_test = data["y_test"]

    model_preds = []
    version_metadata = {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "clf_model_summary": {},
        "anomaly_model_summary": {},
    }

    ##### 🎯 General Model  #####
    for mode, field in [('general', "X_feature"), ('sol', "X_code"), ('opcode', "X_opcode_seq")]:
        X_train, X_test = data[f'{field}_train'], data[f'{field}_test']
        # logger.info(f"🧠 Running Optuna for {mode} Model...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: general_objective(trial, X_train, X_test, y_train, y_test, mode),
            n_trials=N_TRIALS
        )

        model = build_model_by_name(
            study.best_trial.user_attrs["model_name"],
            mode=mode,  # or sol/opcode
            param_source=study.best_trial.params,
            is_trial=False
        )
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        preds = np.array([p[:, 1] for p in probas]).T

        model_preds.append(preds)

        joblib.dump(model, CURRENT_MODEL_PATH /  f"{mode}_model.pkl")

        version_metadata['clf_model_summary'][f'{mode}_model'] = {
            "filename": f"{mode}_model.pkl",
            "f1_score": study.best_value,
            "params": study.best_trial.params,
            "field": field,
        }

    ##### 🔁 GRU Model #####
    # logger.info("🧠 Running Optuna for GRU (timeline)...")

    X_timeline_seq_train = data["X_timeline_seq_train"]
    X_timeline_seq_test = data["X_timeline_seq_test"]

    study_gru = optuna.create_study(direction="maximize")
    study_gru.optimize(
        lambda trial: gru_objective(trial, X_timeline_seq_train, X_timeline_seq_test, y_train.values, y_test.values),
        n_trials=N_TRIALS
    )
    model_gru = build_gru_model(
            input_shape=(X_timeline_seq_train.shape[1], X_timeline_seq_train.shape[2]),
            units=study_gru.best_trial.params['units'], lr=study_gru.best_trial.params['lr'],
            output=y_train.shape[1]
    )

    model_gru, _ = training_gru(model_gru, X_timeline_seq_train, X_timeline_seq_test, y_train.values, y_test.values, study_gru.best_trial.params)

    probas_gru = model_gru.predict(X_timeline_seq_test)

    save_model(model_gru, CURRENT_MODEL_PATH / "gru_model.keras")

    version_metadata['clf_model_summary']['gru_model'] = {
        "filename": "gru_model.keras",
        "f1_score": study_gru.best_value,
        "params": study_gru.best_trial.params,
        "field": "X_timeline_seq"
    }

    ##### 🔗 Fusion #####
    # logger.info("🔗 Running Optuna for Fusion...")
    model_preds.append(probas_gru)
    study_fusion = optuna.create_study(direction="maximize")
    study_fusion.optimize(lambda trial: fusion_objective(trial, model_preds, y_test.values), n_trials=N_TRIALS)

    best_weights = [study_fusion.best_params[f"w{i}"] for i in range(len(model_preds))]
    best_thresholds = [study_fusion.best_params[f"t{i}"] for i in range(y_test.shape[1])]
    y_pred, _ = fuse_predictions(model_preds, weights=best_weights, thresholds=best_thresholds)

    version_metadata['clf_model_summary']['fusion_model'] = {
        "f1_score": study_fusion.best_value,
        "weights": best_weights,
        "thresholds": best_thresholds,
    }


    ##### 📊 Logging #####
    # logger.info("📊 Saving classification report and confusion matrix...")
    report = classification_report(y_test.values, y_pred, output_dict=True, zero_division=0)
    report_path = CURRENT_TRAINING_LOG_PATH / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    cm_path = CURRENT_TRAINING_LOG_PATH / "confusion_matrix.png"
    plot_multilabel_confusion_matrix(y_test.values, y_pred, labels=y_train.columns.tolist(), save_path=cm_path)

    ##### 🧩 Isolation Forest Anomaly Detection #####
    isolation_preds = []
    y_anomaly = (data["y_train"].sum(axis=1) > 0).astype(int)

    for mode, field in [('general', 'X_feature'), ('sol', 'X_code'), ('opcode', 'X_opcode_seq')]:
        train, test = data[f'{field}_train'], data[f'{field}_test']
        # logger.info(f"🌲 Running Optuna for IsolationForest on {mode}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: if_objective(trial, mode, train, y_anomaly), n_trials=N_TRIALS)
        best_model = build_if_model_by_mode(mode, study.best_trial.params)
        best_model.fit(train)
        joblib.dump(best_model, CURRENT_MODEL_PATH / f"if_{mode}_model.pkl")

        pred = best_model.predict(test)
        isolation_preds.append((pred == -1).astype(int))  # anomaly = 1
        version_metadata['anomaly_model_summary'][f'{mode}_model'] = {
            "filename": f"if_{mode}_model.pkl",
            "f1_score": study.best_value,
            "params": study.best_trial.params,
            "field": field
        }

    ##### 🔁 GRU AE Model #####
    # logger.info("🧠 Running Optuna for GRU Autoencoder (Timeline)...")

    study_ae = optuna.create_study(direction="minimize")
    study_ae.optimize(lambda trial: ae_objective(trial, X_timeline_seq_train, X_timeline_seq_test), n_trials=N_TRIALS)

    best_trial_ae = study_ae.best_trial

    # Rebuild and train the best model
    best_model_ae = build_ae_gru_model(
        input_shape=X_timeline_seq_train.shape[1:],  # (SEQ_LEN, FEATURE_DIM)
        units=best_trial_ae.params["units"],
        lr=best_trial_ae.params["lr"]
    )

    best_model_ae, _ = training_gru(best_model_ae, X_timeline_seq_train, X_timeline_seq_test, X_timeline_seq_train, X_timeline_seq_test, best_trial_ae.params)

    # Calculate reconstruction errors
    recon = best_model_ae.predict(X_timeline_seq_test)
    reconstruction_errors = np.mean((X_timeline_seq_test - recon)**2, axis=(1, 2))

    # Optional: Find a threshold (e.g. 95th percentile) to mark anomaly
    threshold = np.percentile(reconstruction_errors, 95)
    anomaly_flags = (reconstruction_errors > threshold).astype(int)
    isolation_preds.append(anomaly_flags)  # Now len = 4

    save_model(best_model_ae, CURRENT_MODEL_PATH / "gru_ae_model.keras")

    # logger.info(f"📉 GRU AE MSE: {study_ae.best_value:.6f} | Threshold (95%): {threshold:.6f}")

    y_anomaly_test = (y_test.sum(axis=1) > 0).astype(int).values

    version_metadata['anomaly_model_summary']['gru_model'] = {
        "filename": "gru_ae_model.keras",
        "MSE": study_ae.best_value,
        "params": study_ae.best_trial.params,
        "threshold": threshold,
        "field": "X_timeline_seq"
    }

    # logger.info("🔗 Running Optuna for Anomaly Fusion...")

    study_anomaly = optuna.create_study(direction="maximize")
    study_anomaly.optimize(lambda trial: anomaly_fusion_objective(trial, isolation_preds, y_anomaly_test), n_trials=N_TRIALS)

    # Save results
    anomaly_weights = [study_anomaly.best_params[f"w{i}"] for i in range(len(isolation_preds))]
    anomaly_threshold = study_anomaly.best_params["threshold"]

    version_metadata['anomaly_model_summary']['fusion_model'] = {
        "f1_score": study_anomaly.best_value,
        "weights": anomaly_weights,
        "threshold": anomaly_threshold
    }

    report_anomaly = classification_report(y_anomaly_test, (np.average(np.stack(isolation_preds, axis=1), axis=1, weights=anomaly_weights) > anomaly_threshold).astype(int), output_dict=True, zero_division=0)
    with open(CURRENT_TRAINING_LOG_PATH / "anomaly_report.json", "w") as f:
        json.dump(report_anomaly, f, indent=2)

    fused_score = np.average(np.stack(isolation_preds, axis=1), axis=1, weights=anomaly_weights)

    plot_anomaly_distribution(fused_score, anomaly_threshold, CURRENT_TRAINING_LOG_PATH / "anomaly_fusion_distribution.png")

    version_metadata['label_names'] = y_train.columns.tolist()
    version_metadata['train_size'] = len(y_train)
    version_metadata['test_size'] = len(y_test)
    version_metadata['notes'] = f"Trained with Optuna {N_TRIALS} trials per model. All models evaluated on {test_size} test split."

    # Save version timestamp
    version_path = CURRENT_MODEL_PATH / "version.json"
    version_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    with open(version_path, "w") as f:
        json.dump(version_metadata, f, indent=4)

    ##### ✅ Done #####
    return version_metadata
