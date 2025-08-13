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
        clf_summary = meta.get("clf_model_summary", {})

        prob_map = {}
        for key, m in clf_summary.items():
            if "filename" not in m:
                continue
            field = m.get("field")
            if not field:
                continue
            Xtr, Xte = data[f"{field}_train"], data[f"{field}_test"]
            fpath = CURRENT_MODEL_PATH / m["filename"]

            if m["filename"].endswith(".keras"):
                model = load_model(fpath)
                if not freeze_gru:
                    params = m.get("params", {"epochs": 10, "batch_size": 64})
                    model, _ = GRUBlocks.train(model, Xtr, Xte, y_train.values, y_test.values, params)
                prob = model.predict(Xte, verbose=1)
                prob_map["gru"] = prob
                m["f1_score"] = f1_score(y_test.values, (prob > 0.5).astype(int), average="macro", zero_division=0)
            else:
                model = joblib.load(fpath)
                if not freeze_sklearn:
                    model.fit(Xtr, y_train)
                else:
                    Xte = FeatureAligner.align_dataframe(Xte, model)
                prob_list = model.predict_proba(Xte)
                prob = np.array([p[:, 1] for p in prob_list]).T
                name = "general" if "general" in key else ("sol" if "sol" in key else ("opcode" if "opcode" in key else key))
                prob_map[name] = prob
                m["f1_score"] = f1_score(y_test, model.predict(Xte), average="macro", zero_division=0)

        fcfg = clf_summary.get("fusion_model", {})
        weights    = fcfg.get("weights", {})
        thresholds = fcfg.get("thresholds", {})

        y_pred, _ = Fusion.fuse(prob_map, label_names, weights=weights, thresholds=thresholds)
        report = classification_report(y_test.values, y_pred, output_dict=True, zero_division=0)
        CURRENT_TRAINING_LOG_PATH.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(CURRENT_TRAINING_LOG_PATH / f"classification_report_eval_{source.split('.')[0]}_test{test_size}.json","w"), indent=2)

        return {
            "clf_model_summary": {**clf_summary, "fusion_model": {**fcfg, "f1_score": f1_score(y_test.values, y_pred, average="macro", zero_division=0)}},
            "label_names": label_names,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "notes": "80/20 eval using saved models; optional refit disabled by default.",
        }
