import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

def prepare_filepath_list(DATA_PATH, df):
    sol_files = []

    for file in list(Path(os.path.join(DATA_PATH, 'sol')).glob('*.sol')):
        if file.stem in df.index:
            sol_files.append(file)

    hex_files = []

    for file in list(Path(os.path.join(DATA_PATH, 'hex')).glob('*.hex')):
        if file.stem in df.index:
            hex_files.append(file)

    matched_stems = set(f.stem for f in sol_files)
    sol_filtered_df = df.loc[df.index.astype(str).isin(matched_stems)]

    matched_stems = set(f.stem for f in hex_files)
    hex_filtered_df = df.loc[df.index.astype(str).isin(matched_stems)]

    return sol_files, hex_files, sol_filtered_df, hex_filtered_df

def draw_confusion_matrix (model, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(cmap=plt.cm.Blues,values_format='g')
    cm_display.ax_.set_title(model)
    plt.show()

def get_report_all_ml(X_train, y_train, X_test, y_test):
    report_list = []

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    models = {
        "LogisticRegression()": LogisticRegression(random_state=42),
        "DecisionTreeClassifier()": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier()": RandomForestClassifier(random_state=42),
        "AdaBoostClassifier()": AdaBoostClassifier(random_state=42),
        "ExtraTreesClassifier()": ExtraTreesClassifier(random_state=42),
        "XGBClassifier()": XGBClassifier(random_state=42),
        "LGBMClassifier()": LGBMClassifier(random_state=42),
        "SVC()": SVC(random_state=42),
        "GaussianNB()": GaussianNB(),
        "KNeighborsClassifier()": KNeighborsClassifier(),
        "SGDClassifier()": SGDClassifier(random_state=42),
        "MLPClassifier()": MLPClassifier(random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Generate classification report (as dict)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        draw_confusion_matrix (name, y_test, y_pred)
        # Average scores across all labels (macro average)
        avg_scores = report_dict["macro avg"]

        report_list.append({
            "Model": name,
            "Precision": avg_scores["precision"],
            "Recall": avg_scores["recall"],
            "F1-score": avg_scores["f1-score"]
        })

    df_report = pd.DataFrame(report_list)
    df_report = df_report.sort_values("F1-score", ascending=False).reset_index(drop=True)

    return df_report
