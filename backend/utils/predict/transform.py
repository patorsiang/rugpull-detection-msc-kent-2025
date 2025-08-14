import pandas as pd
from sklearn.pipeline import Pipeline

class FeatureAligner:
    @staticmethod
    def _feat_names(obj):
        # Try pipeline then estimator; fallback to None
        if hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
        if isinstance(obj, Pipeline):
            for step in ["clf", -1]:
                try:
                    est = obj.named_steps[step] if isinstance(step, str) else list(obj.named_steps.values())[step]
                    if hasattr(est, "feature_names_in_"):
                        return list(est.feature_names_in_)
                except Exception:
                    pass
        return None

    @staticmethod
    def align_dataframe(df: pd.DataFrame, fitted_model) -> pd.DataFrame:
        cols = FeatureAligner._feat_names(fitted_model)
        if cols is None:
            return df.fillna(0)
        return df.reindex(columns=cols, fill_value=0).fillna(0)
