import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
from pathlib import Path

app = FastAPI(title="My API", description="Example FastAPI with Swagger and ReDoc", version="1.0.0")

# Paths
PATH = Path(__file__).resolve().parents[2]
DATA_PATH = os.path.join(PATH, 'data')
LABELED_PATH = os.path.join(DATA_PATH, 'labeled')
UNLABELED_PATH = os.path.join(DATA_PATH, 'unlabeled')
LOGS_PATH = os.path.join(DATA_PATH, 'logs/training')
MODEL_PATH = os.path.join(PATH, 'backend/models')

# Add backend path
sys.path.append(str(PATH))

# Imports
from backend.utils.feature_extraction.bytecode import build_bytecode_feature_dataframe
from backend.utils.feature_extraction.transaction import build_txn_feature_dataframe
from backend.utils.feature_extraction.graph import generate_control_flow_graphs, generate_transaction_graphs, save_graph_features
from backend.utils.feature_extraction.sourcecode import build_sol_feature_dataframe

from backend.utils.models.tabular_data import get_trained_best_model
from backend.utils.models.timeline_data import get_trained_gru_model
from backend.utils.models.graph_data import get_trained_gcn_model
from backend.utils.logging.training_logging import setup_training_log_folder
from backend.utils.logging.training_result import save_confusion_logs

import psutil

def log_memory(msg=""):
    vm = psutil.virtual_memory()
    print(f"[MEMORY] {msg}")
    print(f"    Total RAM    : {vm.total / 1024 / 1024:.2f} MB")
    print(f"    Used RAM     : {vm.used / 1024 / 1024:.2f} MB")
    print(f"    Available RAM: {vm.available / 1024 / 1024:.2f} MB")
    print(f"    Usage        : {vm.percent:.2f}%")

class TrainRequest(BaseModel):
    n_trials: int

@app.get("/")
async def async_route():
    return {"message": "I am an async route"}

@app.post("/train")
def train_model(request: TrainRequest):
    try:
        log_memory("Start of training")
        if request.n_trials <= 0:
            raise HTTPException(status_code=400, detail="n_trials and epochs must be positive integers")

        print(f"[INFO] Starting extracting feature session")
        build_bytecode_feature_dataframe(os.path.join(LABELED_PATH, 'hex'), MODEL_PATH)
        log_memory("After bytecode feature extraction")

        cfg_graphs = generate_control_flow_graphs(os.path.join(LABELED_PATH, 'hex'))
        save_graph_features(LABELED_PATH, 'cfg', cfg_graphs)
        log_memory("After control flow graph feature extraction")

        txn_graphs = generate_transaction_graphs(os.path.join(LABELED_PATH, 'txn'))
        save_graph_features(LABELED_PATH, 'txn', txn_graphs)
        log_memory("After transaction graph feature extraction")

        build_sol_feature_dataframe(os.path.join(LABELED_PATH, 'sol'), MODEL_PATH)
        log_memory("After source code feature extraction")

        log_dir = setup_training_log_folder(LOGS_PATH)
        n_trials = request.n_trials

        print(f"[INFO] Starting training session")

        # === Tabular Models (byte, code, txn) ===
        for path, mode in [('hex', 'byte'), ('sol', 'code'), ('txn', 'txn')]:
            print(f"[INFO] Training tabular model: {mode}")
            model_path = os.path.join(LABELED_PATH, path)
            model, ground_df, _, _, X_test, _, y_test = get_trained_best_model(
                LABELED_PATH, model_path, MODEL_PATH, test_size=0, mode=mode, n_trials=n_trials
            )
            y_pred = model.predict(X_test)
            save_confusion_logs(y_test, y_pred, list(ground_df.columns), log_dir, model_name=f"{mode}_tabular")
            print(model)
            log_memory(f"After tabular training: {mode}")

        # === GRU Model ===
        print("[INFO] Training GRU model")
        model, ground_df, _, X_test, _, y_test, thresholds = get_trained_gru_model(
            LABELED_PATH, MODEL_PATH, n_trials=n_trials
        )
        y_pred = model.predict(X_test)
        y_pred = (y_pred > thresholds).astype(int)
        save_confusion_logs(y_test, y_pred, list(ground_df.columns), log_dir, model_name="gru")
        print(model.summary())
        log_memory("After GRU training")

        # === GCN Models ===
        print("[INFO] Training GCN models")
        for mode in ['txn', 'cfg']:
            print(f"[INFO] Training GCN: {mode}")
            model, y_test, y_pred, label_cols = get_trained_gcn_model(LABELED_PATH, mode, save_path=MODEL_PATH, n_trials=n_trials)
            save_confusion_logs(y_test, y_pred, label_cols, log_dir, model_name=f"gcn_{mode}")
            print(model)
            log_memory(f"After GCN training: {mode}")

        log_memory("Training finished")

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Missing file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
