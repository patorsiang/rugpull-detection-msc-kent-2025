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

def build_model_by_name(name, param_source, is_trial=False, random_state=42):
    if is_trial:
        trial = param_source  # Optuna trial

        def get_float(param_name, *args, **kwargs):
            return trial.suggest_float(param_name, *args, **kwargs)

        def get_int(param_name, *args, **kwargs):
            return trial.suggest_int(param_name, *args, **kwargs)

        def get_cat(param_name, choices):
            return trial.suggest_categorical(param_name, choices)

    else:
        params = param_source

        def get_float(param_name, *_, **__): return params[param_name]
        def get_int(param_name, *_, **__): return params[param_name]
        def get_cat(param_name, *_, **__): return params[param_name]

    match name:
        case "LogisticRegression":
            penalty = get_cat('penalty', ['l2', 'l1', 'elasticnet'])
            solver = "liblinear" if penalty in ("l1", "l2") else "saga"
            return LogisticRegression(
                penalty=penalty,
                class_weight='balanced',
                tol=get_float('tol', 1e-10, 1e-2),
                C=get_float("C", 1e-4, 10.0, log=True),
                solver=solver,
                l1_ratio=(get_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None),
                random_state=random_state,
                max_iter=2000,
                verbose=0
            )
        case "DecisionTree":
            return DecisionTreeClassifier(
                criterion=get_cat('criterion', ['gini', 'entropy', 'log_loss']),
                max_depth=get_int("max_depth", 5, 30),
                class_weight='balanced',
                random_state=random_state,
                verbose=0
            )
        case "RandomForest":
            return RandomForestClassifier(
                n_estimators=get_int("n_estimators", 50, 1000, step=50),
                criterion=get_cat('criterion', ['gini', 'entropy', 'log_loss']),
                max_depth=get_int("max_depth", 5, 30, step=5),
                class_weight='balanced',
                random_state=random_state,
                verbose=0
            )
        case "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=get_int("n_estimators", 50, 1000, step=50),
                learning_rate=get_float("learning_rate", 0.01, 1.0),
                random_state=random_state
            )
        case "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=get_int("n_estimators", 50, 1000, step=50),
                max_depth=get_int("max_depth", 5, 30, step=5),
                random_state=random_state,
                verbose=0
            )
        case "XGBoost":
            return XGBClassifier(
                n_estimators=get_int("n_estimators", 50, 1000, step=50),
                max_depth=get_int("max_depth", 5, 30, step=5),
                learning_rate=get_float("learning_rate", 0.01, 1.0),
                subsample=get_float("subsample", 0.5, 1.0, step=0.1),
                colsample_bytree=get_float("colsample_bytree", 0.5, 1.0, step=0.1),
                eval_metric="logloss",
                random_state=random_state
            )
        case "LightGBM":
            return LGBMClassifier(
                n_estimators=get_int("n_estimators", 50, 1000, step=50),
                max_depth=get_int("max_depth", 5, 30, step=5),
                learning_rate=get_float("learning_rate", 0.01, 1.0),
                subsample=get_float("subsample", 0.5, 1.0, step=0.1),
                colsample_bytree=get_float("colsample_bytree", 0.5, 1.0, step=0.1),
                random_state=random_state,
                verbose=-1
            )
        case "GaussianNB":
            return GaussianNB()
        case "KNN":
            return KNeighborsClassifier(
                n_neighbors=get_int("n_neighbors", 5, 50, step=5),
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
    # Reset index to prevent reindexing error
    ground_df = ground_df.reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Check for 'Address' column match if needed
    if 'Address' in ground_df.columns and 'Address' in df.columns:
        merged_df = pd.merge(ground_df, df, on='Address', how='inner')
        label_cols = [col for col in ground_df.columns if col != 'Address']
    else:
        merged_df = pd.concat([ground_df, df], axis=1).fillna(0)
        label_cols = list(ground_df.columns)

    # Extract labels and features
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
