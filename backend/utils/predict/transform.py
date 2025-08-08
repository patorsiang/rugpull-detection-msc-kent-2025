import pandas as pd

def _expected_feature_names(est):
    # Try the pipeline steps first
    if hasattr(est, "named_steps"):
        for step in est.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # Fallback: top-level estimator
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    return None

def _align_X_to_model(X, model):
    if isinstance(X, pd.DataFrame):
        cols = _expected_feature_names(model)
        if cols is not None:
            # Reindex to expected columns; fill missing with 0; drop unexpected
            X = X.reindex(columns=cols, fill_value=0)
    return X
