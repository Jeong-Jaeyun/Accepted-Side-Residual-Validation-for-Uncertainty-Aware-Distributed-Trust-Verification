"""
Window-level feature extraction for AIS data.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict

import pandas as pd
import numpy as np


def _new_mmsi_rate(df: pd.DataFrame, lookback_k: int) -> pd.Series:
    """Compute rate of MMSI not seen in previous K windows."""
    rates = {}
    recent = deque(maxlen=lookback_k)
    
    for win_id, grp in df.groupby("window_id", sort=True):
        cur = set(grp["mmsi"].unique())
        prev = set().union(*recent) if recent else set()
        new_cnt = len(cur - prev)
        denom = max(len(cur), 1)
        rates[win_id] = new_cnt / denom
        recent.append(cur)
    return pd.Series(rates)


def _message_burstiness(counts: pd.Series, baseline_windows: int, eps: float) -> pd.Series:
    rolling = counts.rolling(baseline_windows, min_periods=1).mean()
    return counts / (rolling + eps)


def _position_jump_rate(df: pd.DataFrame, vmax_knots: float) -> pd.Series:
    from utils.geo import haversine_km  # local import to avoid hard dependency at module import time

    if not {"mmsi", "ts", "lat", "lon", "window_id"}.issubset(df.columns):
        raise KeyError("df must contain columns: mmsi, ts, lat, lon, window_id")

    d = df.sort_values(["mmsi", "ts"]).copy()

    # Previous point per MMSI
    d["lat_prev"] = d.groupby("mmsi")["lat"].shift(1)
    d["lon_prev"] = d.groupby("mmsi")["lon"].shift(1)
    d["ts_prev"] = d.groupby("mmsi")["ts"].shift(1)

    # Time delta (hours)
    dt_h = (d["ts"] - d["ts_prev"]).dt.total_seconds() / 3600.0
    d["dt_h"] = dt_h

    # Distance (km) for valid pairs only
    valid = d["lat_prev"].notna() & d["lon_prev"].notna() & d["dt_h"].notna() & (d["dt_h"] > 0)

    # Initialize implied speed as NaN (float-friendly)
    d["v_knots"] = np.nan

    # Compute implied speed for valid rows (row-wise apply only on valid subset)
    # (최적화는 나중에 가능: numpy 벡터화/numba 등)
    def _dist_km(row) -> float:
        return haversine_km(row["lat_prev"], row["lon_prev"], row["lat"], row["lon"])

    dist_km = d.loc[valid].apply(_dist_km, axis=1)
    v_kmh = dist_km / d.loc[valid, "dt_h"]
    v_knots = v_kmh / 1.852  # 1 knot = 1.852 km/h
    d.loc[valid, "v_knots"] = v_knots

    # Jump flag
    flag = (d["v_knots"] > float(vmax_knots))

    # Aggregate per window (assign to the later point's window_id = current row's window_id)
    out = flag.groupby(d["window_id"]).mean()

    # Windows with no valid pairs won't appear; caller fillna(0) later is fine
    return out



def _speed_heading_inconsistency(
    df: pd.DataFrame, sog_high: float, cog_jump_deg: float, sog_jump_per_min: float
) -> pd.Series:
    """
    Flag rows where high speed pairs with heading jump or sudden speed jump.
    Uses per-MMSI temporal diffs.
    """
    df = df.sort_values(["mmsi", "ts"]).copy()
    df["ts_delta_min"] = df.groupby("mmsi")["ts"].diff().dt.total_seconds() / 60.0
    df["cog_delta"] = df.groupby("mmsi")["cog"].diff().abs()
    df["cog_delta"] = df["cog_delta"].clip(upper=360.0)
    df["cog_delta"] = df["cog_delta"].apply(lambda x: 360 - x if x > 180 else x)

    df["sog_delta_per_min"] = (
        df.groupby("mmsi")["sog"].diff().abs() / df["ts_delta_min"].replace(0, pd.NA)
    )

    flag = (
        ((df["sog"] > sog_high) & (df["cog_delta"] > cog_jump_deg))
        | (df["sog_delta_per_min"] > sog_jump_per_min)
    )
    return flag.groupby(df["window_id"]).mean()


def _spatial_density_entropy(df: pd.DataFrame, eps: float) -> pd.Series:
    """Shannon entropy of spatial density per window using gx, gy buckets."""
    ent = {}
    for win_id, grp in df.groupby("window_id"):
        counts = grp.groupby(["gx", "gy"]).size()
        probs = counts / counts.sum()
        h = -(probs * np.log(probs + eps)).sum()
        ent[win_id] = h
    return pd.Series(ent)


def compute_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Compute window-level features F1..F6.
    Expects columns: window_id, mmsi, ts, sog, cog, gx, gy.
    """
    required_cols = ["window_id", "mmsi", "ts", "sog", "cog", "gx", "gy"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for feature extraction: {missing}")

    lookback_k = cfg["features"].get("lookback_K", 12)
    vmax = cfg["features"].get("vmax_knots", 60.0)
    eps = cfg["features"].get("eps", 1e-9)
    burst_w = cfg["features"].get("burst_baseline_windows", 60)
    cog_jump = cfg["features"].get("cog_jump_deg", 90.0)
    sog_high = cfg["features"].get("sog_high_knots", 10.0)
    sog_jump_rate = cfg["features"].get("sog_jump_knots_per_min", 20.0)

    # F1: unique MMSI count
    f1 = df.groupby("window_id")["mmsi"].nunique()

    # F2: new MMSI rate
    f2 = _new_mmsi_rate(df, lookback_k)

    # F3: message burstiness
    counts = df.groupby("window_id").size()
    f3 = _message_burstiness(counts, burst_w, eps)

    # F4: position jump rate (proxy via speed threshold)
    f4 = _position_jump_rate(df, vmax)

    # F5: speed/heading inconsistency
    f5 = _speed_heading_inconsistency(df, sog_high, cog_jump, sog_jump_rate)

    # F6: spatial density entropy
    f6 = _spatial_density_entropy(df, eps)

    feats = pd.concat(
        [
            f1.rename("F1_unique_mmsi_count"),
            f2.rename("F2_new_mmsi_rate"),
            f3.rename("F3_message_burstiness"),
            f4.rename("F4_position_jump_rate"),
            f5.rename("F5_speed_heading_inconsistency"),
            f6.rename("F6_spatial_density_entropy"),
        ],
        axis=1,
    )
    feats.index.name = "window_id"
    feats = feats.reset_index()

    for col in ["F2_new_mmsi_rate", "F4_position_jump_rate", "F5_speed_heading_inconsistency"]:
        feats[col] = feats[col].clip(lower=0.0)

    feats = feats.fillna(0.0)
    return feats
