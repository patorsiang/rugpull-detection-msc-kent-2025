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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

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
        seq += [[0]*FEATURE_DIM] * (SEQ_LEN - len(seq))
    else:
        seq = seq[:SEQ_LEN]
    return StandardScaler().fit_transform(seq)

def extract_timeline_feature(src_path):
    ts = {}
    txn_files = list(Path(os.path.join(src_path, 'txn')).glob('*.json'))

    for path in tqdm(txn_files, desc="Extracting timeline features"):
        addr = path.stem
        try:
            data = json.load(open(path, 'r'))
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
            data = json.load(open(path, 'r'))
            txns = sorted(data.get("transaction", []), key=lambda x: int(x.get("timeStamp", 0)))
            seq = extract_tx_sequence(txns)
            X.append(pad_and_scale(seq))
            y.append(ground_df.loc[addr].tolist())
        except Exception as e:
            print(f"Skipping {addr}: {e}")

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

def objective(trial, epochs, X, y, test_size):
    try:
        units = trial.suggest_int("units", 32, 516)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 256, log=True)

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
            callbacks=[early_stop, reduce_lr, lr_scheduler],
            verbose=1
        )

        y_pred = model.predict(X_test)
        thresholds, _ = tune_thresholds(y_test, y_pred)
        y_pred_bin = (y_pred > thresholds).astype(int)

        return f1_score(y, y_pred_bin, average='macro')
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

def get_trained_gru_model(labeled_path, model_path, epochs=100, n_trials=100, test_size=0):
    ground_df = pd.read_csv(os.path.join(labeled_path, 'groundtruth.csv'), index_col=0)

    X, y = load_data(labeled_path, ground_df)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence debug spam
    study = optuna.create_study(direction="maximize", study_name="my_study", storage=None, load_if_exists=False)
    study.optimize(partial(objective, epochs=epochs, X=X, y=y, test_size=test_size), n_trials=n_trials, n_jobs=1)

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
        epochs=epochs,
        batch_size=study.best_params['batch_size'],
        callbacks=[early_stop, reduce_lr, lr_scheduler],
        verbose=2
    )

    y_pred = tuned_gru_model.predict(X_test)
    thresholds, _ = tune_thresholds(y_test, y_pred)
    y_pred = (y_pred > thresholds).astype(int)
    weights = f1_score(y_test, y_pred, average=None)

    save_data = {
        'thresholds': thresholds,
        'weights': weights.tolist(),
    }

    # Save thresholds to JSON
    filename = 'gru_txn_extension.json'
    extension_save_path = os.path.join(model_path, filename)
    with open(extension_save_path, 'w') as f:
        json.dump(save_data, f, indent=4)

    print(f"Saved thresholds to {filename}")

    # Save the model
    filename = 'gru_txn_model.keras'
    model_save_path = os.path.join(model_path, filename)
    tuned_gru_model.save(model_save_path)
    print(f"Saved model to {filename}")

    return tuned_gru_model, ground_df, X_train, X_test, y_train, y_test, thresholds
