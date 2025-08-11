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

class DatasetBuilder:
    @staticmethod
    def get_train_test_group(source=GROUND_TRUTH_FILE, test_size=0.2):
        dataset = get_full_dataset(filename=source)
        addresses = list(dataset.keys())
        tr, te = (train_test_split(addresses, test_size=test_size, random_state=42)
                  if test_size and test_size > 0 else (addresses, addresses))

        y_train = pd.DataFrame([dataset[a].get("Label", {}) for a in tr], index=tr).fillna(0).astype(int)
        y_test  = pd.DataFrame([dataset[a].get("Label", {}) for a in te], index=te).fillna(0).astype(int)

        X_opcode_seq_train = [dataset[a].get("opcode_sequence", "") for a in tr]
        X_opcode_seq_test  = [dataset[a].get("opcode_sequence", "") for a in te]

        X_timeline_seq_train = [dataset[a].get("timeline_sequence", []) for a in tr]
        X_timeline_seq_test  = [dataset[a].get("timeline_sequence", []) for a in te]

        X_code_train = [dataset[a].get("sourcecode", "") for a in tr]
        X_code_test  = [dataset[a].get("sourcecode", "") for a in te]

        exclude = {"Label", "opcode_sequence", "timeline_sequence", "sourcecode"}
        X_feature_train = pd.DataFrame([{k: v for k, v in dataset[a].items() if k not in exclude} for a in tr], index=tr)
        X_feature_test  = pd.DataFrame([{k: v for k, v in dataset[a].items() if k not in exclude} for a in te], index=te)
        X_feature_train, X_feature_test = X_feature_train.align(X_feature_test, join="outer", axis=1, fill_value=0)

        X_timeline_seq_train = pad_sequences(X_timeline_seq_train, maxlen=SEQ_LEN, padding="post", dtype="float32")
        X_timeline_seq_test  = pad_sequences(X_timeline_seq_test,  maxlen=SEQ_LEN, padding="post", dtype="float32")

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
