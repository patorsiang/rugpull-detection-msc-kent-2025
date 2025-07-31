from tqdm import tqdm
from pathlib import Path
import os
import json
from functools import partial
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from backend.utils.threshold import tune_thresholds
from backend.utils.comparing import split_train_n_test

def load_data(src_path, ground_df):
    seq_len = 500
    feature_dim = 12

    X, y  = [], []

    for path in tqdm(list(Path(os.path.join(src_path, 'txn')).glob('*.json'))):
        addr = path.stem
        data = json.load(open(path, 'r'))
        txns = sorted(data.get("transaction", []), key=lambda x: int(x.get("timeStamp", 0)))
        seq = [[
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

        if len(seq) < seq_len:
            seq += [[0]*feature_dim] * (seq_len - len(seq))
        else:
            seq = seq[:seq_len]

        seq = StandardScaler().fit_transform(seq)

        if addr in ground_df.index:
            X.append(seq)
            y.append(ground_df.loc[addr].tolist())

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
    units = trial.suggest_int("units", 32, 516)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])

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

def get_trained_gru_model(labeled_path, model_path, epochs=100, n_trials=100, test_size=0):
    ground_df = pd.read_csv(os.path.join(labeled_path, 'groundtruth.csv'), index_col=0)

    X, y = load_data(labeled_path, ground_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, epochs=epochs, X=X, y=y, test_size=test_size), n_trials=n_trials)

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
    weights = f1_score(y_test, y_pred, average=None)

    # Save thresholds to JSON
    filename = 'gru_txn_extension.json'
    threshold_save_path = os.path.join(model_path, filename)
    with open(threshold_save_path, 'w') as f:
        save_data = {
            'thresholds': thresholds,
            'weights': weights
        }
        json.dump(save_data, f, indent=4)

    print(f"Saved thresholds to {threshold_save_path}")

    # Save the model
    filename = 'gru_txn_model.keras'
    model_save_path = os.path.join(model_path, filename)
    tuned_gru_model.save(model_save_path)
    print(f"Saved model to {filename}")

    return tuned_gru_model, ground_df, X_train, X_test, y_train, y_test, thresholds
