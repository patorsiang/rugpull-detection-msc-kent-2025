from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, accuracy_score
import pandas as pd

def report_for_multiple_model(X, y):
    models = {
        "MultiOutput(LogisticRegression)": MultiOutputClassifier(LogisticRegression(class_weight='balanced', random_state=42)),
        "MultiOutput(DecisionTree)": MultiOutputClassifier(DecisionTreeClassifier(random_state=42)),
        "MultiOutput(RandomForest)": MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
        "MultiOutput(AdaBoost)": MultiOutputClassifier(AdaBoostClassifier(algorithm="SAMME",random_state=42)),
        "MultiOutput(ExtraTrees)": MultiOutputClassifier(ExtraTreesClassifier(random_state=42)),
        "MultiOutput(XGBoost)": MultiOutputClassifier(XGBClassifier(random_state=42)),
        "MultiOutput(LightGBM)": MultiOutputClassifier(LGBMClassifier(random_state=42)),
        "MultiOutput(SVC)": MultiOutputClassifier(SVC(probability=True, random_state=42)),
        "MultiOutput(GaussianNB)": MultiOutputClassifier(GaussianNB()),
        "MultiOutput(KNN)": MultiOutputClassifier(KNeighborsClassifier()),
        "MultiOutput(SGD)": MultiOutputClassifier(SGDClassifier(random_state=42)),
        "MultiOutput(MLP)": MultiOutputClassifier(MLPClassifier(random_state=42)),

        "OneVsRest(LogisticRegression)": OneVsRestClassifier(LogisticRegression(class_weight='balanced', random_state=42)),
        "OneVsRest(DecisionTree)": OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
        "OneVsRest(RandomForest)": OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
        "OneVsRest(AdaBoost)": OneVsRestClassifier(AdaBoostClassifier(algorithm="SAMME",random_state=42)),
        "OneVsRest(ExtraTrees)": OneVsRestClassifier(ExtraTreesClassifier(random_state=42)),
        "OneVsRest(XGBoost)": OneVsRestClassifier(XGBClassifier(random_state=42)),
        "OneVsRest(LightGBM)": OneVsRestClassifier(LGBMClassifier(random_state=42)),
        "OneVsRest(SVC)": OneVsRestClassifier(SVC(probability=True, random_state=42)),
        "OneVsRest(GaussianNB)": OneVsRestClassifier(GaussianNB()),
        "OneVsRest(KNN)": OneVsRestClassifier(KNeighborsClassifier()),
        "OneVsRest(SGD)": OneVsRestClassifier(SGDClassifier(random_state=42)),
        "OneVsRest(MLP)": OneVsRestClassifier(MLPClassifier(random_state=42)),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            label_map = {i: label for i, label in enumerate(y.columns)}
            results[name] = {
                "micro avg f1": report.get("micro avg", {}).get("f1-score", None),
                "macro avg f1": report.get("macro avg", {}).get("f1-score", None),
                **{f"{label} f1": report.get(str(i), {}).get("f1-score", None) for i, label in label_map.items()}
            }

        except Exception as e:
            results[name] = {"error": str(e)}

    return pd.DataFrame(results).T, X_train, X_test, y_train, y_test

def evaluate_multilabel_classification(y_true, y_pred, label_names=None, threshold=0.5, average_types=["micro", "macro", "weighted"]):
    """
    Evaluate a multi-label classification model with multiple metrics.

    Parameters:
        y_true (ndarray): shape (n_samples, n_classes), true binary labels
        y_pred (ndarray): shape (n_samples, n_classes), predicted probabilities or logits
        label_names (list): class/label names, optional
        threshold (float): threshold to binarize y_pred if probabilities
        average_types (list): which averaging types to compute

    Returns:
        report_dict (dict): summary with precision/recall/F1 for each average type and per-class
        report_df (DataFrame): detailed classification report
    """
    # Binarize predictions
    y_pred_bin = (y_pred >= threshold).astype(int)

    report_dict = {}

    # Overall metrics by average type
    for avg in average_types:
        report_dict[f"{avg}_precision"] = precision_score(y_true, y_pred_bin, average=avg, zero_division=0)
        report_dict[f"{avg}_recall"] = recall_score(y_true, y_pred_bin, average=avg, zero_division=0)
        report_dict[f"{avg}_f1"] = f1_score(y_true, y_pred_bin, average=avg, zero_division=0)

    # Exact match ratio
    report_dict["subset_accuracy"] = accuracy_score(y_true, y_pred_bin)

    # Per-label report
    cls_report = classification_report(
        y_true, y_pred_bin, target_names=label_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(cls_report).transpose()

    return report_dict, report_df
