import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from backend.core.dataset_service import get_full_dataset
from backend.utils.constants import SEQ_LEN, GROUND_TRUTH_FILE
from backend.utils.logger import logging

logger = logging.getLogger(__name__)

class DatasetBuilder:
    @staticmethod
    def _labels_df(dataset: dict, keys: list[str]) -> pd.DataFrame:
        """
        Build a clean labels DataFrame from dataset[addr]["Label"] entries.
        Ensures each entry is a dict; raises a helpful error otherwise.
        """
        rows = []
        for a in keys:
            lab = dataset.get(a, {}).get("Label", {})
            if isinstance(lab, dict):
                rows.append(lab)
            elif lab in (None, "", [], ()):
                # treat empty-like as no labels
                rows.append({})
            else:
                # Fail early with context instead of hitting list.fillna downstream
                raise ValueError(
                    f"Label for address {a} must be an object/dict of label columns, "
                    f"but got {type(lab).__name__}. "
                    f"Check your CSV row parsing or dataset assembly."
                )
        # Build DF and coerce to numeric ints (-1/0/1), filling missing with 0
        df = pd.DataFrame.from_records(rows, index=keys)
        if df.empty:
            return pd.DataFrame(index=keys)  # no labels yet; caller can handle
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        return df

    @staticmethod
    def get_train_test_group(source=GROUND_TRUTH_FILE, test_size=0.2):
        dataset = get_full_dataset(filename=source)
        addresses = list(dataset.keys())
        tr, te = (
            train_test_split(addresses, test_size=test_size, random_state=42)
            if test_size and test_size > 0 else (addresses, addresses)
        )

        # ---- Labels (robust) ----
        y_train = DatasetBuilder._labels_df(dataset, tr)
        y_test  = DatasetBuilder._labels_df(dataset, te)

        # Align columns between splits (fill missing with 0)
        y_train, y_test = y_train.align(y_test, join="outer", axis=1, fill_value=0)
        y_train = y_train.astype(int)
        y_test  = y_test.astype(int)

        # ---- Inputs ----
        X_opcode_seq_train = [dataset[a].get("opcode_sequence", "") for a in tr]
        X_opcode_seq_test  = [dataset[a].get("opcode_sequence", "") for a in te]

        X_timeline_seq_train = [dataset[a].get("timeline_sequence", []) for a in tr]
        X_timeline_seq_test  = [dataset[a].get("timeline_sequence", []) for a in te]

        X_code_train = [dataset[a].get("sourcecode", "") for a in tr]
        X_code_test  = [dataset[a].get("sourcecode", "") for a in te]

        exclude = {"Label", "opcode_sequence", "timeline_sequence", "sourcecode"}
        X_feature_train = pd.DataFrame(
            [{k: v for k, v in dataset[a].items() if k not in exclude} for a in tr],
            index=tr
        )
        X_feature_test  = pd.DataFrame(
            [{k: v for k, v in dataset[a].items() if k not in exclude} for a in te],
            index=te
        )
        X_feature_train, X_feature_test = X_feature_train.align(
            X_feature_test, join="outer", axis=1, fill_value=0
        )

        # Pad timeline sequences
        X_timeline_seq_train = pad_sequences(
            X_timeline_seq_train, maxlen=SEQ_LEN, padding="post", dtype="float32"
        )
        X_timeline_seq_test  = pad_sequences(
            X_timeline_seq_test,  maxlen=SEQ_LEN, padding="post", dtype="float32"
        )

        return {
            "X_opcode_seq_train": X_opcode_seq_train,
            "X_opcode_seq_test":  X_opcode_seq_test,
            "X_timeline_seq_train": X_timeline_seq_train,
            "X_timeline_seq_test":  X_timeline_seq_test,
            "X_code_train": X_code_train,
            "X_code_test":  X_code_test,
            "X_feature_train": X_feature_train,
            "X_feature_test":  X_feature_test,
            "y_train": y_train,
            "y_test":  y_test,
        }

class Plotter:
    @staticmethod
    def multilabel_confusion(y_true, y_pred, labels, save_path):
        cm_list = multilabel_confusion_matrix(y_true, y_pred)
        n = len(labels)
        _, axs = plt.subplots(1, n, figsize=(6*n, 5))
        for i in range(n):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_list[i], display_labels=[f"Not {labels[i]}", labels[i]])
            disp.plot(cmap=plt.cm.Blues, ax=axs[i] if n > 1 else axs, values_format="d", colorbar=False)
            axs[i].set_title(labels[i])
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def anomaly_hist(scores, threshold, save_path):
        plt.figure(figsize=(8, 4))
        sns.histplot(scores, kde=True, bins=50)
        plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.2f}")
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
