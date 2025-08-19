from __future__ import annotations
import math
import json
from typing import Any, Mapping, Iterable
import numpy as np
import pandas as pd

def _fix_number(x: float | int) -> float | int | None:
    try:
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, (np.integer,)):
            x = int(x)
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
        return x
    except Exception:
        return None

def sanitize_json(obj: Any) -> Any:
    """
    Recursively convert any NaN/Inf to None and make numpy/pandas JSON‑safe.
    """
    # pandas
    if isinstance(obj, pd.DataFrame):
        # replace inf -> NaN, then NaN -> None
        safe = obj.replace([np.inf, -np.inf], np.nan).where(pd.notnull(obj), None)
        return json.loads(safe.to_json(orient="records"))
    if isinstance(obj, pd.Series):
        safe = obj.replace([np.inf, -np.inf], np.nan)
        return [None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
                for v in safe.where(pd.notnull(safe), None).tolist()]

    # numpy scalars/arrays
    if isinstance(obj, (np.floating, np.integer)):
        return _fix_number(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())

    # base types
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return _fix_number(obj)

    # mappings / iterables
    if isinstance(obj, Mapping):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(v) for v in obj]

    # fallback: best effort string
    try:
        json.dumps(obj)  # will succeed?
        return obj
    except Exception:
        return str(obj)
