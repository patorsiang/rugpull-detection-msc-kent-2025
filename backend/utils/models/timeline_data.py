from tqdm import tqdm
from pathlib import Path
import os
import gc
import json
from functools import partial
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import keras
import tensorflow as tf
import tensorflow.keras.backend as tfkb
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from optuna_integration.tfkeras import TFKerasPruningCallback

from backend.utils.threshold import tune_thresholds
from backend.utils.comparing import split_train_n_test

SEQ_LEN = 500
FEATURE_DIM = 12

def extract_tx_sequence(txns):
    return [[
        int(tx.get("blockNumber", 0)),
        int(tx.get("timeStamp", 0)),
        int(tx.get("nonce", 0)),
        int(tx.get("transactionIndex", 0)),
        int(tx.get("value", 0)),
        int(tx.get("gas", 0)),
        int(tx.get("gasPrice", 0)),
        int(tx.get("isError", 0)),
        int(tx.get("txreceipt_status", 0)),
        int(tx.get("cumulativeGasUsed", 0)),
        int(tx.get("gasUsed", 0)),
        int(tx.get("confirmations", 0)),
    ] for tx in txns]

def pad_and_scale(seq):
    if len(seq) < SEQ_LEN:
        seq = seq + [[0]*FEATURE_DIM] * (SEQ_LEN - len(seq))
    else:
        seq = seq[:SEQ_LEN]

    arr = np.asarray(seq, dtype=float)

    # Fit scaler on non-padded rows only (any nonzero feature)
    nonpad_mask = np.any(arr != 0.0, axis=1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    if np.any(nonpad_mask):
        arr[nonpad_mask] = scaler.fit_transform(arr[nonpad_mask])
        # transform padded rows with same scaler center/scale
        if np.any(~nonpad_mask):
            arr[~nonpad_mask] = (arr[~nonpad_mask] - scaler.mean_) / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    else:
        # all padded; just return zeros
        arr[:] = 0.0

    # Replace any residual non-finite values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def extract_timeline_feature(src_path, address=None):
    ts = dict()
    txn_dir = Path(os.path.join(src_path, 'txn'))
    pattern = f"{address}.json" if address else "*.json"
    txn_files = list(txn_dir.glob(pattern))

    for path in tqdm(txn_files, desc="Extracting timeline features"):
        addr = path.stem
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            txns = sorted(data.get("transaction", []), key=lambda x: int(x.get("timeStamp", 0)))
            seq = extract_tx_sequence(txns)
            ts[addr] = pad_and_scale(seq)
        except Exception as e:
            print(f"Skipping {addr}: {e}")

    return ts  # Dict[str, np.ndarray shape=(500,12)]

def load_data(src_path, ground_df):
    X, y = [], []
    txn_files = list(Path(os.path.join(src_path, 'txn')).glob('*.json'))

    for path in tqdm(txn_files, desc="Loading labeled timeline data"):
        addr = path.stem
        try:
            if addr not in ground_df.index:
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            txns = sorted(data.get("transaction", []), key=lambda x: int(x.get("timeStamp", 0)))
            seq = extract_tx_sequence(txns)
            padded = pad_and_scale(seq)
            label = ground_df.loc[addr].tolist()

            if padded.shape == (SEQ_LEN, FEATURE_DIM) and len(label) > 0:
                X.append(padded)
                y.append(label)
        except Exception as e:
            print(f"Skipping {addr}: {e}")

    if len(X) != len(y):
        raise ValueError(f"[ERROR] X and y length mismatch: X={len(X)}, y={len(y)}")
    return np.array(X), np.array(y)


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

def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * 0.9

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
lr_scheduler = LearningRateScheduler(scheduler)

def objective(trial, X, y, test_size):
    try:
        tfkb.clear_session()
        units = trial.suggest_int("units", 32, 516, step=8)
        lr = trial.suggest_float("lr", 1e-10, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
        epochs = trial.suggest_int("epochs", 100, 200, step=10)

        print(f"[Trial {trial.number}] {{'units': {units}, 'lr': {lr}, 'batch_size': {batch_size}, 'epochs': {epochs}}}")

        model = build_gru_model(
            input_shape=(X.shape[1], X.shape[2]),
            units=units, lr=lr,
            output=y.shape[1]
        )

        X_train, X_test, y_train, y_test = split_train_n_test(X, y, test_size)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, lr_scheduler, TFKerasPruningCallback(trial, "val_loss")],
            verbose=1
        )

        y_pred = model.predict(X_test, verbose=0)
        thresholds, _ = tune_thresholds(y_test, y_pred)
        thresholds = np.asarray(thresholds).tolist()  # ensure JSON-safe later
        y_pred_bin = (y_pred > np.asarray(thresholds)).astype(int)

        # (Optional) extra prune gate post-training – safe to keep
        if trial.should_prune():
            raise optuna.TrialPruned()

        return f1_score(y_test, y_pred_bin, average='macro')
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"[Trial failed] {e}")
        return float('-inf')
    finally:
        tfkb.clear_session()
        keras.backend.clear_session()
        if 'model' in locals():
            del model
        gc.collect()

def get_trained_gru_model(labeled_path, model_path, n_trials=100, test_size=0, n_jobs=1):
    ground_df = pd.read_csv(os.path.join(labeled_path, 'groundtruth.csv'), index_col=0)
    X, y = load_data(labeled_path, ground_df)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize",
                                storage=optuna.storages.InMemoryStorage(),
                                load_if_exists=False,
                                pruner=pruner)

    study.optimize(partial(objective, X=X, y=y, test_size=test_size),
                   n_trials=n_trials, n_jobs=n_jobs)

    print("✅ Best Params:", study.best_params)
    print("🥇 Best Score:", study.best_value)

    tuned_gru_model = build_gru_model(
        input_shape=(X.shape[1], X.shape[2]),
        units=study.best_params['units'],
        lr=study.best_params['lr'],
        output=y.shape[1]
    )

    X_train, X_test, y_train, y_test = split_train_n_test(X, y, test_size)

    tuned_gru_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=study.best_params['epochs'],
        batch_size=study.best_params['batch_size'],
        callbacks=[early_stop, reduce_lr, lr_scheduler],
        verbose=2
    )

    y_pred = tuned_gru_model.predict(X_test, verbose=0)
    thresholds, _ = tune_thresholds(y_test, y_pred)
    thresholds = np.asarray(thresholds).tolist()
    y_pred_bin = (y_pred > np.asarray(thresholds)).astype(int)
    weights = f1_score(y_test, y_pred_bin, average=None)

    save_data = {
        'thresholds': thresholds,
        'weights': weights.tolist(),
    }

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'gru_txn_extension.json'), 'w') as f:
        json.dump(save_data, f, indent=4)
    tuned_gru_model.save(os.path.join(model_path, 'gru_txn_model.keras'))

    return tuned_gru_model, ground_df, X_train, X_test, y_train, y_test, thresholds
