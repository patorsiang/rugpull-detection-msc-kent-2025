import os
import json
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.utils.comparing import _draw_confusion_matrix

def save_confusion_logs(y_test, y_pred, label_cols, log_dir, model_name="model"):
    """
    Save classification report (TXT + JSON) and confusion matrix (PNG) to disk.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Save JSON report
    json_report = classification_report(y_test, y_pred, target_names=label_cols, output_dict=True, zero_division=0)
    json_path = os.path.join(log_dir, f"{model_name}_report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=4)

    # Save confusion matrix
    fig = _draw_confusion_matrix(y_test, y_pred, label_cols)
    fig_path = os.path.join(log_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"[Saved] Logs for {model_name} → {log_dir}")
