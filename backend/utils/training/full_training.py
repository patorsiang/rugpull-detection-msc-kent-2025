import json
import joblib
import numpy as np
import optuna
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import classification_report, f1_score

from backend.utils.training.extra_classes import DatasetBuilder, Plotter
from backend.utils.training.training_objectives import TSBlocks
from backend.utils.predict.fusion import Fusion
from backend.utils.predict.anomaly_fusion import AnomalyFusion
from backend.core.meta_service import MetaService
from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, N_TRIALS, GROUND_TRUTH_FILE
from backend.utils.training.backup import BackupManager

class FullTrainer:
    """Retrain all on 100% labeled data, then re‑optimize both fusions."""

    def __init__(self, n_trials: int = N_TRIALS):
        self.n_trials = n_trials

    def run(self, source=GROUND_TRUTH_FILE) -> dict:
        BackupManager.backup_current()

        ds = DatasetBuilder.get_train_test_group(source, 0.0)
        y_train, y_test = ds["y_train"], ds["y_test"]  # same index/split
        label_names = y_train.columns.tolist()

        meta = MetaService.read_version()
        meta["version"] = meta.get("version", "v") + "_1"

        # refit models
        prob_map = {}
        for key, m in list(meta["clf_model_summary"].items()):
            if "filename" not in m: continue
            field = m["field"]
            X = ds[f"{field}_train"]
            fpath = CURRENT_MODEL_PATH / m["filename"]

            if m["filename"].endswith(".keras"):
                model = load_model(fpath)
                params = m.get("params", {"epochs":10, "batch_size":64})
                model, _ = TSBlocks.train(model, X, X, y_train.values, y_test.values, params)
                save_model(model, fpath)
                prob_map["gru"] = model.predict(X, verbose=1)
            else:
                model = joblib.load(fpath)
                model.fit(X, y_train)
                joblib.dump(model, fpath)
                plist = model.predict_proba(X)
                prob_map["general" if key.startswith("general") else "sol" if key.startswith("sol") else "opcode"] = \
                    np.array([p[:,1] for p in plist]).T

        # fusion re‑opt via Optuna (dicts)
        study_f = optuna.create_study(direction="maximize")
        study_f.optimize(lambda t: self._fusion_obj(t, prob_map, y_train.values, label_names), n_trials=self.n_trials)
        w = {k.replace("w_",""):v for k,v in study_f.best_params.items() if k.startswith("w_")}
        t = {k.replace("t_",""):v for k,v in study_f.best_params.items() if k.startswith("t_")}
        y_pred, _ = Fusion.fuse(prob_map, label_names, w, t)
        CURRENT_TRAINING_LOG_PATH.mkdir(parents=True, exist_ok=True)
        json.dump(classification_report(y_train.values, y_pred, output_dict=True, zero_division=0),
                  open(CURRENT_TRAINING_LOG_PATH / "classification_report.json","w"), indent=2)
        Plotter.multilabel_confusion(y_train.values, y_pred, label_names, CURRENT_TRAINING_LOG_PATH / "confusion_matrix.png")
        meta["clf_model_summary"]["fusion_model"] = {"f1_score": float(study_f.best_value), "weights": w, "thresholds": t}

        # anomaly re‑fit
        iso_maps = {}
        y_anom = (y_train.sum(axis=1) > 0).astype(int).values
        for key, m in list(meta["anomaly_model_summary"].items()):
            if "filename" not in m: continue
            field = m["field"]
            X = ds[f"{field}_train"]
            fpath = CURRENT_MODEL_PATH / m["filename"]

            if m["filename"].endswith(".keras"):
                model = load_model(fpath)
                params = m.get("params", {"epochs":10, "batch_size":64})
                model, _ = TSBlocks.train(model, X, X, X, X, params)
                save_model(model, fpath)
                recon = model.predict(X, verbose=1)
                err = np.mean((X - recon) ** 2, axis=(1, 2))
                err = np.nan_to_num(err, nan=0.0, posinf=1e12, neginf=0.0)

                finite = np.isfinite(err)
                if np.any(finite):
                    finite_vals = err[finite]
                    thr = float(np.percentile(finite_vals, 95))
                else:
                    med = float(np.median(err))
                    mad = float(np.median(np.abs(err - med))) or 1.0
                    thr = med + 3.0 * mad

                thr = float(min(thr, 1e6))

                iso_maps["ae_timeline"] = (err > thr).astype(int).reshape(-1)
                m["MSE"] = float(np.mean(err[finite])) if np.any(finite) else float(np.mean(err))
                m["threshold"] = thr
            else:
                model = joblib.load(fpath)
                model.fit(X)
                joblib.dump(model, fpath)
                pred = (model.predict(X) == -1).astype(int).reshape(-1)
                name = "if_general" if "if_general" in key else ("if_sol" if "if_sol" in key else "if_opcode")
                iso_maps[name] = pred

        # anomaly fusion re‑opt
        study_a = optuna.create_study(direction="maximize")
        study_a.optimize(lambda t: self._anom_obj(t, iso_maps, y_anom), n_trials=self.n_trials)
        aw = {k.replace("w_",""):v for k,v in study_a.best_params.items() if k.startswith("w_")}
        ath = float(study_a.best_params["threshold"])
        meta["anomaly_model_summary"]["fusion_model"] = {"f1_score": float(study_a.best_value), "weights": aw, "threshold": ath}
        _, score = AnomalyFusion.fuse(iso_maps, aw, ath)
        Plotter.anomaly_hist(score, ath, CURRENT_TRAINING_LOG_PATH / "anomaly_fusion_distribution.png")
        MetaService.write_version(meta)
        return meta

    def _fusion_obj(self, trial, prob_map, y_true, label_names):
        w = {f"w_{k}": trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in prob_map.keys()}
        t = {f"t_{lbl}": trial.suggest_float(f"t_{lbl}", 0.3, 0.7) for lbl in label_names}
        pred, _ = Fusion.fuse(prob_map, label_names,
                              {k.replace("w_",""):v for k,v in w.items()},
                              {k.replace("t_",""):v for k,v in t.items()})
        return f1_score(y_true, pred, average="macro")

    def _anom_obj(self, trial, iso_maps, y_true):
        w = {f"w_{k}": trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in iso_maps.keys()}
        flag, _ = AnomalyFusion.fuse(iso_maps, {k.replace("w_",""):v for k,v in w.items()},
                                     trial.suggest_float("threshold", 0.3, 0.7))
        return f1_score(y_true, flag, zero_division=0)
