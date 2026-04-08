"""
One-hot encode discretized feature levels (L/M/H) into binary vectors.
"""
from __future__ import annotations

from typing import Iterable, List
import pandas as pd


def onehot_levels(df: pd.DataFrame, feature_order: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    out_cols: List[str] = []

    for feat in feature_order:
        level_col = f"level_{feat}"
        if level_col not in df.columns:
            raise KeyError(f"Missing discretized column: {level_col}")
        dummies = pd.get_dummies(df[level_col], prefix=feat)
        # Ensure all three levels exist
        for lvl in ["L", "M", "H"]:
            col_name = f"{feat}_{lvl}"
            if col_name not in dummies:
                dummies[col_name] = 0
        dummies = dummies[[f"{feat}_L", f"{feat}_M", f"{feat}_H"]]
        out_cols.extend(dummies.columns.tolist())
        df = pd.concat([df, dummies], axis=1)

    keep_cols = ["window_id"] + out_cols
    out = df[keep_cols].copy()
    for c in out_cols:
        out[c] = out[c].astype(int)
    return out
