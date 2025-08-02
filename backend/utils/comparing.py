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

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

def get_model_set():
    return {
        "MultiOutput(LogisticRegression)": MultiOutputClassifier(LogisticRegression(class_weight='balanced', random_state=42)),
        "MultiOutput(DecisionTree)": MultiOutputClassifier(DecisionTreeClassifier(random_state=42)),
        "MultiOutput(RandomForest)": MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
        "MultiOutput(AdaBoost)": MultiOutputClassifier(AdaBoostClassifier(random_state=42)),
        "MultiOutput(ExtraTrees)": MultiOutputClassifier(ExtraTreesClassifier(random_state=42)),
        "MultiOutput(XGBoost)": MultiOutputClassifier(XGBClassifier(random_state=42)),
        "MultiOutput(LightGBM)": MultiOutputClassifier(LGBMClassifier(random_state=42)),
        "MultiOutput(SVC)": MultiOutputClassifier(SVC(probability=True, random_state=42)),
        "MultiOutput(GaussianNB)": MultiOutputClassifier(GaussianNB()),
        "MultiOutput(KNN)": MultiOutputClassifier(KNeighborsClassifier()),
        "MultiOutput(SGD)": MultiOutputClassifier(SGDClassifier(random_state=42)),
        "MultiOutput(MLP)": MultiOutputClassifier(MLPClassifier(random_state=42, max_iter=500)),
    }

def evaluate_models(X_train, X_test, y_train, y_test):
    models = get_model_set()
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            label_map={i: col for i, col in enumerate(y_test.columns)}
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results[name] = {
                "micro avg f1": report.get("micro avg", {}).get("f1-score", None),
                "macro avg f1": report.get("macro avg", {}).get("f1-score", None),
                **{f"{label} f1": report.get(str(i), {}).get("f1-score", None) for i, label in label_map.items()}
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return pd.DataFrame(results).T

def report_for_multiple_model(X, y, test_size=0.2, random_state=42):
    if test_size == 0:
        X_train, y_train = X, y
        X_test, y_test = shuffle(X, y, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return evaluate_models(X_train, X_test, y_train, y_test), (X_train, X_test, y_train, y_test)

def report_for_multiple_model_as_same_set(X_train, X_test, y_train, y_test):
    return evaluate_models(X_train, X_test, y_train, y_test), (X_train, X_test, y_train, y_test)

def build_model_by_name(name, param_source, is_trial=False, random_state=42):
    if is_trial:
        trial = param_source  # Optuna trial object

        def get_float(param_name, *args, **kwargs): return trial.suggest_float(param_name, *args, **kwargs)
        def get_int(param_name, *args, **kwargs): return trial.suggest_int(param_name, *args, **kwargs)
        def get_cat(param_name, *args, **kwargs): return trial.suggest_categorical(param_name, *args, **kwargs)

    else:
        params = param_source

        def get_float(param_name, *_, **__): return params[param_name]
        def get_int(param_name, *_, **__): return params[param_name]
        def get_cat(param_name, *_, **__): return params[param_name]

    match name:
        case "LogisticRegression":
            return LogisticRegression(
                C=get_float("C", 0.01, 10.0), class_weight='balanced',
                solver="liblinear", random_state=random_state,
                verbose=0
            )
        case "DecisionTree":
            return DecisionTreeClassifier(
                max_depth=get_int("max_depth", 2, 20),
                class_weight='balanced', random_state=random_state
            )
        case "RandomForest":
            return RandomForestClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 5, 30),
                class_weight='balanced', random_state=random_state,
                verbose=0
            )
        case "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                learning_rate=get_float("learning_rate", 0.01, 1.0),
                random_state=random_state
            )
        case "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 5, 30),
                random_state=random_state,
                verbose=0
            )
        case "XGBoost":
            return XGBClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 3, 10),
                learning_rate=get_float("learning_rate", 0.01, 0.3),
                subsample=get_float("subsample", 0.5, 1.0),
                colsample_bytree=get_float("colsample_bytree", 0.5, 1.0),
                eval_metric="logloss", random_state=random_state
            )
        case "LightGBM":
            return LGBMClassifier(
                n_estimators=get_int("n_estimators", 50, 200),
                max_depth=get_int("max_depth", 3, 10),
                learning_rate=get_float("learning_rate", 0.01, 0.3),
                subsample=get_float("subsample", 0.5, 1.0),
                colsample_bytree=get_float("colsample_bytree", 0.5, 1.0),
                random_state=random_state,
                verbose=-1
                # verbose=0
            )
        case "SVC":
            return SVC(
                C=get_float("C", 0.1, 10.0),
                kernel=get_cat("kernel", ["linear", "rbf", "poly"]),
                probability=True, random_state=random_state,
                verbose=0
            )
        case "GaussianNB":
            return GaussianNB()
        case "KNN":
            return KNeighborsClassifier(
                n_neighbors=get_int("n_neighbors", 3, 15),
                weights=get_cat("weights", ["uniform", "distance"]),
            )
        case "SGD":
            return SGDClassifier(
                alpha=get_float("alpha", 1e-5, 1e-1),
                loss=get_cat("loss", ["hinge", "log_loss"]),
                random_state=random_state,
                verbose=0
            )
        case "MLP":
            hidden_options = {
                "100": (100,),
                "50_50": (50, 50),
                "100_50": (100, 50)
            }
            key = get_cat("hidden_layer_sizes", list(hidden_options.keys()))
            return MLPClassifier(
                hidden_layer_sizes=hidden_options[key],
                activation=get_cat("activation", ["relu", "tanh"]),
                learning_rate_init=get_float("learning_rate_init", 1e-4, 1e-1),
                max_iter=500, random_state=random_state,
                verbose=0
            )
        case _:
            raise ValueError(f"Unsupported model name: {name}")

def merge_n_split(ground_df, df, test_size=0.2, random_state=42):
    label_cols = list(ground_df.columns)
    merged_df = pd.concat([ground_df, df], axis=1).fillna(0)
    y = merged_df[label_cols]
    X = merged_df.drop(columns=label_cols)

    return split_train_n_test(X, y, test_size, random_state)

def split_train_n_test(X, y, test_size=0.2, random_state=42):
    if test_size == 0:
        X_train, y_train = X, y
        X_test, y_test = shuffle(X, y, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def save_model(name, model, weights, save_dir, feature_cols, vectorizer):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump({
            "model": model,
            "weights": weights,
            "feature_cols": feature_cols,
            "vectorizer": vectorizer
        }, f)

    print(f"✅ Model saved {name}.pkl")
    return save_path

def _draw_confusion_matrix(y_test, y_pred, label_cols):
    """
    Return a matplotlib Figure object of multilabel confusion matrix.
    """
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    n_labels = len(label_cols)
    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4))

    for i, matrix in enumerate(mcm):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['No ' + label_cols[i], label_cols[i]])
        disp.plot(cmap=plt.cm.Blues, ax=axes[i], colorbar=False)
        axes[i].set_title(label_cols[i])

    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred, label_cols):
    """
    Show confusion matrix plot and print classification report to console.
    """
    print(classification_report(y_test, y_pred, target_names=label_cols))
    _draw_confusion_matrix(y_test, y_pred, label_cols)
    plt.show()
