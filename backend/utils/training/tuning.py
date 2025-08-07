import pandas as pd
from sklearn.model_selection import train_test_split

import os
import json
import optuna
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend.utils.training.training_objectives import general_objective, gru_objective, fusion_objective, build_model_by_name, build_gru_model
from backend.utils.predict.fusion import fuse_predictions

from backend.utils.constants import MODELS_PATH, TRAINING_LOG_PATH, N_TRIALS, SEQ_LEN

from backend.core.dataset_service import get_full_dataset

from backend.utils.logger import get_logger

logger = get_logger("train_and_save")

def get_train_test_group(test_size=0.2):
    dataset = get_full_dataset()
    addresses = list(dataset.keys())

    # Split addresses
    addresses_train, addresses_test = train_test_split(addresses, test_size=test_size, random_state=42)

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
        "X_timeline_seq_train": X_timeline_seq_train,
        "X_timeline_seq_test": X_timeline_seq_test,
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

def train_and_save_best_model(test_size=0.2):
    logger.info("📦 Loading data and splitting train/test...")
    # Get train/test split
    data = get_train_test_group(test_size=test_size)

    y_train = data["y_train"]
    y_test = data["y_test"]

    model_preds = []
    result = dict()

    ##### 🎯 General Model  #####
    for mode, X_train, X_test in [('general', data["X_feature_train"], data["X_feature_test"]), ('sol', data["X_code_train"], data["X_code_test"]), ('opcode', data["X_opcode_seq_train"], data["X_opcode_seq_test"])]:
        logger.info(f"🧠 Running Optuna for {mode} Model...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: general_objective(trial, X_train, X_test, y_train, y_test, mode),
            n_trials=N_TRIALS
        )

        result[f'{mode}_model_f1'] = study.best_value

        model_general = build_model_by_name(
            study.best_trial.user_attrs["model_name"],
            mode=mode,  # or sol/opcode
            param_source=study.best_trial.params,
            is_trial=False
        )
        model_general.fit(X_train, y_train)

        probas = model_general.predict_proba(X_test)
        preds = np.array([p[:, 1] for p in probas]).T

        model_preds.append(preds)

        joblib.dump(model_general, MODELS_PATH /  f"{mode}_model.pkl")

    ##### 🔁 GRU Model #####
    logger.info("🧠 Running Optuna for GRU (timeline)...")

    X_timeline_seq_train = pad_sequences(data["X_timeline_seq_train"], maxlen=SEQ_LEN, padding='post', dtype='float32')
    X_timeline_seq_test = pad_sequences(data["X_timeline_seq_test"], maxlen=SEQ_LEN, padding='post', dtype='float32')

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
    probas_gru = model_gru.predict(X_timeline_seq_test)

    save_model(model_gru, MODELS_PATH / "gru_model.keras")

    ##### 🔗 Fusion #####
    logger.info("🔗 Running Optuna for Fusion...")
    model_preds.append(probas_gru)
    study_fusion = optuna.create_study(direction="maximize")
    study_fusion.optimize(lambda trial: fusion_objective(trial, model_preds, y_test.values), n_trials=N_TRIALS)

    best_weights = [study_fusion.best_params[f"w{i}"] for i in range(len(model_preds))]
    best_thresholds = [study_fusion.best_params[f"t{i}"] for i in range(y_test.shape[1])]
    y_pred, _ = fuse_predictions(model_preds, weights=best_weights, thresholds=best_thresholds)

    ##### 📊 Logging #####
    logger.info("📊 Saving classification report and confusion matrix...")
    report = classification_report(y_test.values, y_pred, labels=y_train.columns, output_dict=True, zero_division=0)
    report_path = TRAINING_LOG_PATH / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    cm_path = TRAINING_LOG_PATH / "confusion_matrix.png"
    plot_multilabel_confusion_matrix(y_test.values, y_pred, labels=y_train.columns, save_path=cm_path)

    ##### ✅ Done #####
    return {
        "status": "Training complete",
        **result,
        "gru_model_f1": study_gru.best_value,
        "fusion_model_f1": study_fusion.best_value,
        "weights": best_weights,
        "thresholds": best_thresholds,
        "report_path": report_path
    }
