"""
Shared helpers for the Power Quality anomaly detection frontend.
The feature logic here MUST stay identical to optimised_isolation_forest.py,
otherwise the saved model will receive different features than it was trained on.
"""

import numpy as np
import pandas as pd

LABEL_COL = "Disturbance Type"          # 0 = normal, 1..4 = disturbance types
TIME_COL = "Start Date and Time"

UP_MIN = "Upstream Voltage Min(kV)"
UP_AVG = "Upstream Voltage Avg(kV)"
UP_MAX = "Upstream Voltage Max(kV)"
DN_AVG = "Downstream Voltage Avg(V)"
DN_MAX = "Downstream Voltage Max(V)"
THD_MIN = "THD-V Min(%)"
THD_AVG = "THD-V Avg(%)"
THD_MAX = "THD-V Max(%)"

BASE_COLS = [UP_MIN, UP_AVG, UP_MAX, DN_AVG, DN_MAX, THD_MIN, THD_AVG, THD_MAX]
WINDOW = 30                              # rolling window in minutes


def load_pq_csv(source):
    """Read a CSV (path or file-like), clean headers, sort by time."""
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
        if df[TIME_COL].notna().any():
            df = df.sort_values(TIME_COL, kind="stable")
    return df.reset_index(drop=True)


def missing_columns(df):
    return [c for c in BASE_COLS if c not in df.columns]


def build_features(df):
    """Returns a dict {feature_set_name: DataFrame} (raw / full / relative)."""
    eps = 1e-6
    raw = df[BASE_COLS].astype(float).copy()

    eng = pd.DataFrame(index=df.index)
    eng["up_range"] = df[UP_MAX] - df[UP_MIN]
    eng["up_hi"] = df[UP_MAX] - df[UP_AVG]
    eng["up_lo"] = df[UP_AVG] - df[UP_MIN]
    eng["thd_range"] = df[THD_MAX] - df[THD_MIN]
    eng["thd_peak"] = df[THD_MAX] / (df[THD_AVG] + eps)
    eng["dn_peak"] = df[DN_MAX] / (df[DN_AVG] + eps)

    key = pd.concat([raw, eng[["up_range", "thd_range"]]], axis=1)
    temporal = {}
    for c in key.columns:
        s = key[c]
        roll = s.rolling(WINDOW, min_periods=1)
        temporal[c + "_diff"] = s.diff().fillna(0)
        temporal[c + "_dev"] = s - roll.mean()
        temporal[c + "_rstd"] = roll.std().fillna(0)
    temporal = pd.DataFrame(temporal, index=df.index)

    sets = {
        "raw": raw,
        "full": pd.concat([raw, eng, temporal], axis=1),
        "relative": pd.concat([eng, temporal], axis=1),
    }
    for k in sets:
        sets[k] = sets[k].replace([np.inf, -np.inf], np.nan).fillna(0)
    return sets
