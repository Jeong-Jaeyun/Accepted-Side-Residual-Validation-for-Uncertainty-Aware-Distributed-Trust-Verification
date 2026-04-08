"""
Assign window_id to each AIS row based on dt_minutes and t0.
"""
from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from utils.time import assign_window_id, compute_t0


def add_windows(df: pd.DataFrame, dt_minutes: int, t0: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Add `window_id` column.
    - dt_minutes: window size in minutes.
    - t0: optional ISO timestamp (tz-aware); if None, use min ts in df.
    Returns (df_with_window_id, t0_used).
    """
    if "ts" not in df.columns:
        raise KeyError("'ts' column required for windowing")

    ts_utc = df["ts"]
    if t0 is None:
        t0_val = compute_t0(ts_utc)
    else:
        t0_val = pd.to_datetime(t0)
        if t0_val.tzinfo is None:
            t0_val = t0_val.tz_localize("UTC")
        else:
            t0_val = t0_val.tz_convert("UTC")

    df = df.copy()
    df["window_id"] = assign_window_id(ts_utc, dt_minutes, t0_val)
    n_before = len(df)
    df = df.dropna(subset=["window_id"])
    n_after = len(df)
    df["window_id"] = df["window_id"].astype(int)
    return df, t0_val
