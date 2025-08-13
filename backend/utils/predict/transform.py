import pandas as pd

class FeatureAligner:
    """Ensure runtime DataFrames match the fitted pipeline’s expected columns."""

    @staticmethod
    def _expected_feature_names(est):
        if hasattr(est, "named_steps"):
            for step in est.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
        return None

    @staticmethod
    def align_dataframe(X, model):
        if isinstance(X, pd.DataFrame):
            cols = FeatureAligner._expected_feature_names(model)
            if cols is not None:
                X = X.reindex(columns=cols, fill_value=0)
        return X
