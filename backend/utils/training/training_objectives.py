import gc

import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import f1_score

import keras
import tensorflow as tf
import tensorflow.keras.backend as tfkb
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from backend.utils.threshold import tune_thresholds
from backend.utils.logger import get_logger
from backend.utils.predict.fusion import fuse_predictions

logger = get_logger("auto_ml")

def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * 0.9

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
lr_scheduler = LearningRateScheduler(scheduler)

def build_gru_model(input_shape, units, lr, output):
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        GRU(units),
        Dense(output, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model



def build_model_by_name(name, mode, param_source, is_trial=False, random_state=42):
    if is_trial:
        trial = param_source
        def get_float(param_name, *args, **kwargs): return trial.suggest_float(param_name, *args, **kwargs)
        def get_int(param_name, *args, **kwargs): return trial.suggest_int(param_name, *args, **kwargs)
        def get_cat(param_name, *args, **kwargs): return trial.suggest_categorical(param_name, *args, **kwargs)
    else:
        params = param_source
        def get_float(param_name, *_, **__): return params[param_name]
        def get_int(param_name, *_, **__): return params[param_name]
        def get_cat(param_name, *_, **__): return params[param_name]

    match name:
        case "LogisticRegression":
            base_model = LogisticRegression(
                C=get_float("C", 0.01, 10.0),
                class_weight='balanced',
                solver="liblinear",
                random_state=random_state
            )

        case "RidgeClassifier":
            base_model = RidgeClassifier(
                alpha=get_float("alpha", 0.01, 10.0),
                random_state=random_state
            )

        case "RandomForest":
            base_model = RandomForestClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 3, 20),
                class_weight='balanced',
                random_state=random_state
            )

        case "BaggingClassifier":
            base_model = BaggingClassifier(
                n_estimators=get_int("n_estimators", 10, 100),
                random_state=random_state
            )

        case "HistGradientBoosting":
            base_model = HistGradientBoostingClassifier(
                max_iter=get_int("max_iter", 100, 500),
                max_depth=get_int("max_depth", 3, 20),
                random_state=random_state
            )

        case "XGBoost":
            base_model = XGBClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 3, 20),
                learning_rate=get_float("learning_rate", 0.01, 0.3),
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                random_state=random_state
            )

        case "LightGBM":
            base_model = LGBMClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 3, 20),
                learning_rate=get_float("learning_rate", 0.01, 0.3),
                random_state=random_state,
                verbose=-1
            )

        case "MLP":
            hidden_options = {
                "32": (32,),
                "64": (64,),
                "128": (128,),
                "256": (256,),
                "32_64": (32, 64),
                "64_32": (64, 32),
                "64_64": (64, 64),
                "64_128": (64, 128),
                "128_128": (128, 128),
                "32_64_128": (32, 64, 128)
            }
            key = get_cat("hidden_layer_sizes", list(hidden_options.keys()))
            base_model = MLPClassifier(
                hidden_layer_sizes=hidden_options[key],
                learning_rate_init=get_float("learning_rate_init", 1e-4, 1e-1),
                max_iter=300,
                random_state=random_state
            )

        case _:
            raise ValueError(f"Unknown model name: {name}")

    base_model = MultiOutputClassifier(base_model)

    match mode:
        case 'general':
            return Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", base_model)
            ])
        case 'sol':
            return Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    analyzer='word',
                    token_pattern=r'\b\w+\b',
                    max_features=get_int("n_max_features", 10, 20000),
                    min_df=get_int("n_min_df", 1, 5)
                )),
                ("clf", base_model)
            ])
        case 'opcode':
            ngram_range_options = {
                "1": (1, 1),
                "2": (1, 2),
                "3": (1, 3),
            }
            key = get_cat("ngram_range", list(ngram_range_options.keys()))
            return Pipeline([
              ("count", CountVectorizer(
                    analyzer='word',
                    ngram_range=ngram_range_options[key],
                    max_features=get_int("n_max_features", 10, 20000),
                    min_df=get_int("n_min_df", 1, 5)
              )),
              ("clf", base_model)
            ])
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def general_objective(trial, X_train, X_test, y_train, y_test, mode='general', random_state=42):
    try:
        model_name = trial.suggest_categorical("model", [
            "RandomForest",         # Strong general-purpose baseline
            "XGBoost",              # Powerful gradient boosting
            "LightGBM",             # Fast, efficient boosting (esp. large data)
            "LogisticRegression",   # Good linear baseline
            "MLP",                  # Deep learning baseline for general
            "RidgeClassifier",      # Linear model with L2 regularization
            "BaggingClassifier",    # Bootstrap ensemble method
            "HistGradientBoosting"  # sklearn's efficient GBDT implementation
        ])

        trial.set_user_attr("model_name", model_name)

        model = build_model_by_name(model_name, mode, trial, is_trial=True, random_state=random_state)

        logger.info(f"[Trial {trial.number}] {model}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        trial.set_user_attr("final_model", model)

        return f1_score(y_test, y_pred, average='macro', zero_division=0)
    except Exception as e:
        logger.error(f"[Trial failed] {e}")
        return float('-inf')  # or np.nan
    finally:
        del_vars = ['model_name', 'model', 'X_train', 'X_test', 'y_train', 'y_test']
        for var in del_vars:
            obj = locals().get(var, None)
            if obj is not None:
                del obj
        gc.collect()


def gru_objective(trial, X_train, X_test, y_train, y_test):
    try:
        tfkb.clear_session()
        units = trial.suggest_int("units", 32, 516)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
        epochs = trial.suggest_int("epochs", 8, 100)

        logger.info(f"[Trial {trial.number}] {{'units': {units}, 'lr': {lr}, 'batch_size': {batch_size}, 'epochs': {epochs},}}")

        model = build_gru_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=units, lr=lr,
            output=y_train.shape[1]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, lr_scheduler],
            verbose=1
        )

        trial.set_user_attr("train_loss", history.history['loss'])
        trial.set_user_attr("val_loss", history.history['val_loss'])

        y_pred = model.predict(X_test)
        thresholds, _ = tune_thresholds(y_test, y_pred)
        trial.set_user_attr("thresholds", thresholds)
        y_pred_bin = (y_pred > thresholds).astype(int)

        return f1_score(y_test, y_pred_bin, average='macro')
    except Exception as e:
        logger.error(f"[Trial failed] {e}")
        return float('-inf')  # or np.nan
    finally:
        tfkb.clear_session()
        tf.compat.v1.reset_default_graph()
        keras.backend.clear_session()
        obj = locals().get('model', None)
        if obj is not None:
            del obj
        gc.collect()

def fusion_objective(trial, model_preds: list, y_true: np.ndarray):
    weights = [
        trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(model_preds))
    ]

    thresholds = [
        trial.suggest_float(f"t{i}", 0.3, 0.7) for i in range(y_true.shape[1])
    ]

    pred, fused = fuse_predictions(model_preds, weights=weights, thresholds=thresholds)

    trial.set_user_attr("weights", weights)
    trial.set_user_attr("thresholds", thresholds)
    trial.set_user_attr("confidence", fused)

    return f1_score(y_true, pred, average='macro')
