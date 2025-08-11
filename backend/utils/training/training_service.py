import json
from datetime import datetime
import numpy as np
import optuna
import joblib
from tensorflow.keras.models import save_model
import tensorflow.keras.backend as tfbk
from sklearn.ensemble import IsolationForest
from backend.utils.training.extra_classes import DatasetBuilder, Plotter
from backend.utils.training.training_objectives import (
    SklearnFactory, GRUBlocks,
    Pipeline, SimpleImputer, StandardScaler, TfidfVectorizer, CountVectorizer
)
from backend.utils.predict.transform import FeatureAligner
from backend.utils.predict.fusion import Fusion
from backend.utils.predict.anomaly_fusion import AnomalyFusion
from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, N_TRIALS, GROUND_TRUTH_FILE
from backend.core.meta_service import MetaService
from backend.utils.training.backup import BackupManager
from sklearn.metrics import classification_report, f1_score

class Trainer:
    """End‑to‑end Optuna training that writes version.json (dict‑based fusion)."""

    def __init__(self, n_trials: int = N_TRIALS):
        self.n_trials = n_trials

    def train_and_save(self, test_size=0.2, source=GROUND_TRUTH_FILE) -> dict:
        BackupManager.backup_current()

        data = DatasetBuilder.get_train_test_group(source=source, test_size=test_size)
        y_train, y_test = data["y_train"], data["y_test"]
        label_names = y_train.columns.tolist()

        meta = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "clf_model_summary": {},
            "anomaly_model_summary": {},
            "label_names": label_names,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "notes": f"Optuna {self.n_trials} trials per model; test split={test_size}.",
        }

        # ---- sklearn models
        model_probs = {}
        for mode, field in [("general", "X_feature"), ("sol", "X_code"), ("opcode", "X_opcode_seq")]:
            Xtr, Xte = data[f"{field}_train"], data[f"{field}_test"]
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: self._general_objective(t, Xtr, Xte, y_train, y_test, mode), n_trials=self.n_trials)

            model = SklearnFactory.build_model_by_name(study.best_trial.user_attrs["model_name"], mode, study.best_trial.params, is_trial=False)
            model.fit(Xtr, y_train)
            probas_list = model.predict_proba(Xte)
            probs = np.array([p[:, 1] for p in probas_list]).T
            model_probs[mode] = probs

            joblib.dump(model, CURRENT_MODEL_PATH / f"{mode}_model.pkl")
            meta["clf_model_summary"][f"{mode}_model"] = {
                "filename": f"{mode}_model.pkl",
                "f1_score": float(study.best_value),
                "params": study.best_trial.params,
                "field": field,
            }

        # ---- GRU clf
        Xtr_t, Xte_t = data["X_timeline_seq_train"], data["X_timeline_seq_test"]
        study_gru = optuna.create_study(direction="maximize")
        study_gru.optimize(lambda t: self._gru_objective(t, Xtr_t, Xte_t, y_train.values, y_test.values), n_trials=self.n_trials)

        gru = GRUBlocks.build_classifier((Xtr_t.shape[1], Xtr_t.shape[2]), study_gru.best_trial.params["units"], study_gru.best_trial.params["lr"], output=y_train.shape[1])
        gru, _ = GRUBlocks.train(gru, Xtr_t, Xte_t, y_train.values, y_test.values, study_gru.best_trial.params)
        prob_gru = gru.predict(Xte_t, verbose=1)
        model_probs["gru"] = prob_gru
        save_model(gru, CURRENT_MODEL_PATH / "gru_model.keras")
        meta["clf_model_summary"]["gru_model"] = {
            "filename": "gru_model.keras",
            "f1_score": float(study_gru.best_value),
            "params": study_gru.best_trial.params,
            "field": "X_timeline_seq",
        }

        # ---- fusion (dicts)
        study_fusion = optuna.create_study(direction="maximize")
        study_fusion.optimize(lambda t: self._fusion_objective(t, model_probs, y_test.values, label_names), n_trials=self.n_trials)
        weights = {k.replace("w_", ""): v for k, v in study_fusion.best_params.items() if k.startswith("w_")}
        thresholds = {k.replace("t_", ""): v for k, v in study_fusion.best_params.items() if k.startswith("t_")}
        y_pred, fused = Fusion.fuse(model_probs, label_names, weights=weights, thresholds=thresholds)
        CURRENT_TRAINING_LOG_PATH.mkdir(parents=True, exist_ok=True)
        json.dump(classification_report(y_test.values, y_pred, output_dict=True, zero_division=0),
                  open(CURRENT_TRAINING_LOG_PATH / "classification_report.json", "w"), indent=2)
        Plotter.multilabel_confusion(y_test.values, y_pred, label_names, CURRENT_TRAINING_LOG_PATH / "confusion_matrix.png")

        meta["clf_model_summary"]["fusion_model"] = {
            "f1_score": float(study_fusion.best_value),
            "weights": weights,
            "thresholds": thresholds,
        }

        # ---- anomaly: IF + AE
        isolation_maps = {}
        y_anom_tr = (y_train.sum(axis=1) > 0).astype(int)
        y_anom_te = (y_test.sum(axis=1) > 0).astype(int).values

        for mode, field in [("general", "X_feature"), ("sol", "X_code"), ("opcode", "X_opcode_seq")]:
            Xtr, Xte = data[f"{field}_train"], data[f"{field}_test"]
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: self._if_objective(t, mode, Xtr, y_anom_tr), n_trials=self.n_trials)
            model = self._build_if(mode, study.best_trial.params)
            model.fit(Xtr)
            joblib.dump(model, CURRENT_MODEL_PATH / f"if_{mode}_model.pkl")
            isolation_maps[f"if_{mode}"] = (model.predict(Xte) == -1).astype(int).reshape(-1)
            meta["anomaly_model_summary"][f"{mode}_model"] = {
                "filename": f"if_{mode}_model.pkl",
                "f1_score": float(study.best_value),
                "params": study.best_trial.params,
                "field": field,
            }

        # AE timeline
        study_ae = optuna.create_study(direction="minimize")
        study_ae.optimize(lambda t: self._ae_objective(t, Xtr_t, Xte_t), n_trials=self.n_trials)
        ae = GRUBlocks.build_autoencoder(Xtr_t.shape[1:], study_ae.best_trial.params["units"], study_ae.best_trial.params["lr"])
        ae, _ = GRUBlocks.train(ae, Xtr_t, Xte_t, Xtr_t, Xte_t, study_ae.best_trial.params)
        save_model(ae, CURRENT_MODEL_PATH / "gru_ae_model.keras")
        recon = ae.predict(Xte_t, verbose=1)
        recon_err = np.mean((Xte_t - recon) ** 2, axis=(1, 2))
        thr = float(np.percentile(recon_err, 95))
        isolation_maps["ae_timeline"] = (recon_err > thr).astype(int).reshape(-1)
        meta["anomaly_model_summary"]["gru_model"] = {
            "filename": "gru_ae_model.keras",
            "MSE": float(np.mean(recon_err[np.isfinite(recon_err)])) if np.any(np.isfinite(recon_err)) else float(np.mean(recon_err)),
            "params": study_ae.best_trial.params,
            "threshold": thr,
            "field": "X_timeline_seq",
        }

        # anomaly fusion
        study_anom = optuna.create_study(direction="maximize")
        study_anom.optimize(lambda t: self._anomaly_fusion_objective(t, isolation_maps, y_anom_te), n_trials=self.n_trials)
        aw = {k.replace("w_", ""): v for k, v in study_anom.best_params.items() if k.startswith("w_")}
        ath = float(study_anom.best_params["threshold"])
        meta["anomaly_model_summary"]["fusion_model"] = {"f1_score": float(study_anom.best_value), "weights": aw, "threshold": ath}
        _, score = AnomalyFusion.fuse(isolation_maps, aw, ath)
        Plotter.anomaly_hist(score, ath, CURRENT_TRAINING_LOG_PATH / "anomaly_fusion_distribution.png")
        # save version
        MetaService.write_version(meta)
        return meta

    # ---- objective wrappers
    def _general_objective(self, trial, Xtr, Xte, ytr, yte, mode):
        name = trial.suggest_categorical("model", ["RandomForest","XGBoost","LightGBM","LogisticRegression","MLP","BaggingClassifier"])
        trial.set_user_attr("model_name", name)
        model = SklearnFactory.build_model_by_name(name, mode, trial, is_trial=True)
        model.fit(Xtr, ytr)
        Xf_clf = FeatureAligner.align_dataframe(Xte.copy(), model)
        return f1_score(yte, model.predict(Xf_clf), average="macro", zero_division=0)

    def _gru_objective(self, trial, Xtr, Xte, ytr, yte):
        tfbk.clear_session()
        units = trial.suggest_int("units", 32, 516)
        lr    = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        bs    = trial.suggest_int("batch_size", 16, 256, log=True)
        ep    = trial.suggest_int("epochs", 8, 100)
        m = GRUBlocks.build_classifier((Xtr.shape[1], Xtr.shape[2]), units, lr, output=ytr.shape[1])
        m, _ = GRUBlocks.train(m, Xtr, Xte, ytr, yte, {"epochs": ep, "batch_size": bs})
        return f1_score(yte, (m.predict(Xte, verbose=1) > 0.5).astype(int), average="macro")

    def _fusion_objective(self, trial, prob_map, y_true, label_names):
        w = {f"w_{k}": trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in prob_map.keys()}
        t = {f"t_{lbl}": trial.suggest_float(f"t_{lbl}", 0.3, 0.7) for lbl in label_names}
        weights   = {k.replace("w_", ""): v for k, v in w.items()}
        thresholds= {k.replace("t_", ""): v for k, v in t.items()}
        pred, _ = Fusion.fuse(prob_map, label_names, weights, thresholds)
        return f1_score(y_true, pred, average="macro")

    def _anomaly_fusion_objective(self, trial, iso_maps, y_true):
        w = {f"w_{k}": trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in iso_maps.keys()}
        flag, _ = AnomalyFusion.fuse(iso_maps, {k.replace("w_",""):v for k,v in w.items()},
                                     trial.suggest_float("threshold", 0.3, 0.7))
        return f1_score(y_true, flag, zero_division=0)

    def _build_if(self, mode, params):
        iso = IsolationForest(n_estimators=params["n_estimators"], contamination=params["contamination"], max_features=params["max_features"], random_state=42)
        if mode == "general":
            return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler(with_mean=False)), ("clf", iso)])
        if mode == "sol":
            return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"\b\w+\b",
                                                       max_features=params["n_max_features"], min_df=params["n_min_df"])),
                             ("clf", iso)])
        if mode == "opcode":
            ngram = {"1":(1,1), "2":(1,2), "3":(1,3)};  # assume params["ngram_range"] in {"1","2","3"}
            return Pipeline([("count", CountVectorizer(analyzer="word", ngram_range=ngram[params["ngram_range"]],
                                                       max_features=params["n_max_features"], min_df=params["n_min_df"])),
                             ("clf", iso)])
        raise ValueError(mode)

    def _if_objective(self, trial, mode, Xtr, ytrue):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        contamination= trial.suggest_float("contamination", 0.01, 0.2)
        max_features = trial.suggest_float("max_features", 0.5, 1.0)

        if mode == "general":
            model = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler(with_mean=False)),
                              ("clf", IsolationForest(n_estimators=n_estimators, contamination=contamination, max_features=max_features, random_state=42))])
        elif mode == "sol":
            n_max = trial.suggest_int("n_max_features", 10, 20000)
            n_min = trial.suggest_int("n_min_df", 1, 5)
            model = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"\b\w+\b", max_features=n_max, min_df=n_min)),
                              ("clf", IsolationForest(n_estimators=n_estimators, contamination=contamination, max_features=max_features, random_state=42))])
        else: # opcode
            n_max = trial.suggest_int("n_max_features", 10, 20000)
            n_min = trial.suggest_int("n_min_df", 1, 5)
            ngr   = trial.suggest_categorical("ngram_range", ["1","2","3"])
            mapping = {"1":(1,1), "2":(1,2), "3":(1,3)}
            model = Pipeline([("count", CountVectorizer(analyzer="word", ngram_range=mapping[ngr], max_features=n_max, min_df=n_min)),
                              ("clf", IsolationForest(n_estimators=n_estimators, contamination=contamination, max_features=max_features, random_state=42))])

        model.fit(Xtr)
        pred = (model.predict(Xtr) == -1).astype(int)
        return f1_score(ytrue, pred, zero_division=0)

    def _ae_objective(self, trial, Xtr, Xte):
        tfbk.clear_session()
        units = trial.suggest_int("units", 32, 516)
        lr    = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        bs    = trial.suggest_int("batch_size", 16, 256, log=True)
        ep    = trial.suggest_int("epochs", 8, 100)
        m = GRUBlocks.build_autoencoder(Xtr.shape[1:], units, lr)
        m, _ = GRUBlocks.train(m, Xtr, Xte, Xtr, Xte, {"epochs": ep, "batch_size": bs})
        recon = m.predict(Xte, verbose=1)
        return float(np.mean((Xte - recon) ** 2))
