import os
import gc
import optuna
import pandas as pd
from functools import partial
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from backend.utils.feature_extraction.bytecode import build_bytecode_feature_dataframe
from backend.utils.feature_extraction.sourcecode import build_sol_feature_dataframe
from backend.utils.feature_extraction.transaction import build_txn_feature_dataframe
from backend.utils.comparing import build_model_by_name, merge_n_split, save_model

def get_feature_df(path, model_path, max_features, min_df, use_saved_model, mode='byte'):
    match(mode):
        case 'byte':
            return build_bytecode_feature_dataframe(
                hex_dir=path,
                MODEL_PATH=model_path,
                max_features=max_features,
                min_df=min_df,
                use_saved_model=use_saved_model
            )
        case 'code':
            return build_sol_feature_dataframe(
                sol_dir=path,
                MODEL_PATH=model_path,
                max_features=max_features,
                min_df=min_df,
                use_saved_model=use_saved_model
            )

# === OBJECTIVE FUNCTION ===
def objective(trial, ground_df, path, model_path, random_state, mode, df=None):
    try:
        model_name = trial.suggest_categorical("model", [
            "LogisticRegression", "DecisionTree", "RandomForest", "AdaBoost", "ExtraTrees",
            "XGBoost", "LightGBM", "SVC", "GaussianNB", "KNN", "SGD", "MLP"
        ])

        base_model = build_model_by_name(model_name, trial, is_trial=True, random_state=random_state)

        if df is None:
            df, _ = get_feature_df(
                path,
                model_path,
                max_features=trial.suggest_int("n_max_features", 10, 20000),
                min_df=trial.suggest_int("n_min_df", 1, 5),
                use_saved_model=False,
                mode=mode
            )

        X_full, _, y_full, _ = merge_n_split(ground_df, df, test_size=0)

        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

        model = MultiOutputClassifier(base_model)
        score = cross_val_score(model, X_full, y_full, scoring="f1_macro", cv=kf).mean()
        return score
    finally:
        del base_model, model, X_full, y_full
        gc.collect()


# === OPTUNA OPTIMIZER WRAPPER ===
def get_trained_best_model(labeled_path, path, model_path, test_size=0.2, random_state=42, n_trials=50, n_jobs=1, mode='byte'):
    ground_df = pd.read_csv(os.path.join(labeled_path, 'groundtruth.csv'), index_col=0)

    df = None
    vectorizer = {}

    if mode == 'txn':
        df = build_txn_feature_dataframe(path)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence debug spam
    study = optuna.create_study(direction="maximize", study_name="my_study", storage=None, load_if_exists=False)
    study.optimize(partial(objective,
                           random_state=random_state,
                           ground_df=ground_df,
                           path=path,
                           model_path=model_path,
                           mode=mode,
                           df=df),
                           n_trials=n_trials,
                           n_jobs=n_jobs)

    print("✅ Best Params:", study.best_params)
    print("🥇 Best Score:", study.best_value)

    best_params = study.best_params
    model_name = best_params.pop("model")

    if df is None:
        n_max_features = best_params.pop("n_max_features")
        n_min_df = best_params.pop("n_min_df")
        df, vectorizer = get_feature_df(
            path,
            model_path,
            max_features=n_max_features,
            min_df=n_min_df,
            use_saved_model=False,
            mode=mode
        )

    X_train, X_test, y_train, y_test = merge_n_split(ground_df, df, test_size=test_size, random_state=random_state)

    base_model = build_model_by_name(model_name, best_params, is_trial=False, random_state=random_state)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    save_model(mode, model, weights=f1_score(y_test, y_pred, average=None), save_dir=model_path, feature_cols=list(df.columns), vectorizer=vectorizer)

    return model, ground_df, df, X_train, X_test, y_train, y_test
