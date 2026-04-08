"""
Discretize continuous features into L/M/H using quantiles.
"""
from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd


FEATURE_COLS = [
    "F1_unique_mmsi_count",
    "F2_new_mmsi_rate",
    "F3_message_burstiness",
    "F4_position_jump_rate",
    "F5_speed_heading_inconsistency",
    "F6_spatial_density_entropy",
]


def fit_quantiles(df: pd.DataFrame, q_low: float, q_high: float) -> Dict[str, Dict[str, float]]:
    thresholds = {}
    for col in FEATURE_COLS:
        if col not in df:
            raise KeyError(f"Missing feature column: {col}")
        ql = df[col].quantile(q_low)
        qh = df[col].quantile(q_high)
        thresholds[col] = {"low": ql, "high": qh}
    return thresholds


def apply_quantiles(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, th in thresholds.items():
        low, high = th["low"], th["high"]
        level_col = f"level_{col}"
        if pd.isna(low) or pd.isna(high) or low >= high:
            # Fallback for constant/degenerate distributions
            levels = pd.Series("M", index=df.index)
            levels[df[col] < low] = "L"
            levels[df[col] > high] = "H"
            df[level_col] = levels
        else:
            df[level_col] = pd.cut(
                df[col],
                bins=[-float("inf"), low, high, float("inf")],
                labels=["L", "M", "H"],
                include_lowest=True,
            )
    return df
