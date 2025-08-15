import json
from backend.core.meta_service import MetaService
from backend.utils.training.training_service import Trainer
from backend.utils.training.full_training import FullTrainer
from backend.utils.training.evaluation import Evaluator
from backend.utils.constants import GROUND_TRUTH_FILE, N_TRIALS, CURRENT_MODEL_PATH
from backend.utils.logger import logging

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, n_trials=N_TRIALS):
        self.trainer = Trainer(n_trials)
        self.full = FullTrainer(n_trials)
        self.eval = Evaluator()

    def run(self, source=GROUND_TRUTH_FILE, eval_source=GROUND_TRUTH_FILE, test_size=0.2):
        status = MetaService.get_status()

        if status.get("current_version") in (None, "Not trained yet"):
            meta = self.trainer.train_and_save(test_size=test_size, source=source)
            self.full.run(source=source)
            return meta

        baseline = self.eval.evaluate(test_size=test_size, source=eval_source, freeze_gru=True, freeze_sklearn=True)
        trial    = self.eval.evaluate(test_size=test_size, source=source,     freeze_gru=False, freeze_sklearn=False)

        b = baseline.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")
        n = trial.get("clf_model_summary", {}).get("fusion_model", {}).get("f1_score")
        logger.info(f"[PIPELINE] baseline={b}, trial={n}, improved={n is None or b is None or n >= b}")

        if (b is None) or (n is not None and n >= b):
            self.trainer.train_and_save(test_size=test_size, source=source)
            self.full.run(source=source)

        return json.load(open(CURRENT_MODEL_PATH / "version.json", "r"))
