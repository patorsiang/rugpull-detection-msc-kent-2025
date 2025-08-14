import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import load_model

from backend.utils.training.extra_classes import DatasetBuilder, Plotter
from backend.utils.constants import CURRENT_MODEL_PATH, CURRENT_TRAINING_LOG_PATH, GROUND_TRUTH_FILE
from backend.utils.training.training_objectives import GRUBlocks
from backend.utils.predict.fusion import Fusion
from backend.utils.predict.transform import FeatureAligner
from backend.core.meta_service import MetaService

class Evaluator:
    """80/20 evaluation reusing saved models; optional refit on 80%."""

    def evaluate(self, test_size=0.2, source=GROUND_TRUTH_FILE, freeze_gru=True, freeze_sklearn=False) -> dict:
        data = DatasetBuilder.get_train_test_group(source=source, test_size=test_size)
        y_train, y_test = data["y_train"], data["y_test"]
        label_names = y_train.columns.tolist()

        meta = MetaService.read_version()
        clf_summary = meta.get("clf_model_summary", {}) or {}

        prob_map = {}
        for key, m in clf_summary.items():
            # skip fusion_model row
            if "filename" not in m:
                continue

            field = m.get("field")
            if not field:
                continue

            # Decide head name once (for alignment decision later)
            # keys are like "general_model", "sol_model", "opcode_model", "gru_model"
            if "general" in key:
                mode_name = "general"
            elif "sol" in key:
                mode_name = "sol"
            elif "opcode" in key:
                mode_name = "opcode"
            elif "gru" in key:
                mode_name = "gru"
            else:
                mode_name = key  # fallback

            Xtr, Xte = data[f"{field}_train"], data[f"{field}_test"]
            fpath = CURRENT_MODEL_PATH / m["filename"]

            if m["filename"].endswith(".keras"):
                # GRU classifier
                model = load_model(fpath)
                if not freeze_gru:
                    params = m.get("params", {"epochs": 10, "batch_size": 64})
                    model, _ = GRUBlocks.train(model, Xtr, Xte, y_train.values, y_test.values, params)

                prob = model.predict(Xte, verbose=0)
                prob_map["gru"] = prob
                m["f1_score"] = f1_score(
                    y_test.values, (prob > 0.5).astype(int),
                    average="macro", zero_division=0
                )
            else:
                # Sklearn pipelines
                model = joblib.load(fpath)
                if not freeze_sklearn:
                    # re-fit on current split’s train; no alignment needed
                    model.fit(Xtr, y_train)
                    X_eval = Xte
                else:
                    # using saved model: align only for numeric ("general")
                    if mode_name == "general":
                        X_eval = FeatureAligner.align_dataframe(Xte.copy(), model)
                    else:
                        X_eval = Xte

                prob_list = model.predict_proba(X_eval)
                prob = np.array([p[:, 1] for p in prob_list]).T

                # map into prob_map by head name
                if mode_name in ("general", "sol", "opcode"):
                    prob_map[mode_name] = prob
                else:
                    # fallback to key-derived name to avoid collisions
                    prob_map[key] = prob

                m["f1_score"] = f1_score(
                    y_test, model.predict(X_eval),
                    average="macro", zero_division=0
                )

        # Fuse with saved weights/thresholds
        fcfg = clf_summary.get("fusion_model", {}) or {}
        weights    = fcfg.get("weights", {}) or {}
        thresholds = fcfg.get("thresholds", {}) or {}

        y_pred, _ = Fusion.fuse(prob_map, label_names, weights=weights, thresholds=thresholds)
        report = classification_report(y_test.values, y_pred, output_dict=True, zero_division=0)

        CURRENT_TRAINING_LOG_PATH.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_TRAINING_LOG_PATH / f"classification_report_eval_{source.split('.')[0]}_test{test_size}.json", "w") as f:
            json.dump(report, f, indent=2)

        return {
            "clf_model_summary": {
                **clf_summary,
                "fusion_model": {**fcfg, "f1_score": f1_score(y_test.values, y_pred, average="macro", zero_division=0)}
            },
            "label_names": label_names,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "notes": "80/20 eval using saved models; optional refit disabled by default."
        }
