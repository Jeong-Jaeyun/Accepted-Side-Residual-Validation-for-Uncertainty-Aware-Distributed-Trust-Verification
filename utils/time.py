# utils/time.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    dt_minutes: int
    tz: str = "UTC"


def to_datetime_utc(series: pd.Series, tz_hint: str = "UTC") -> pd.Series:
    """
    Convert a timestamp column to timezone-aware UTC pandas datetime.
    - If input is naive, localize using tz_hint then convert to UTC.
    - If input has timezone, convert to UTC.
    """
    ts = pd.to_datetime(series, errors="coerce", utc=False)

    # If ts is timezone-aware already, tz will be set.
    # Pandas uses dtype datetime64[ns, tz] for tz-aware.
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("UTC")

    # Naive -> localize with hint then convert to UTC
    ts = ts.dt.tz_localize(tz_hint, ambiguous="NaT", nonexistent="NaT")
    return ts.dt.tz_convert("UTC")


def compute_t0(ts_utc: pd.Series) -> pd.Timestamp:
    """
    Use minimum timestamp as t0; must be timezone-aware UTC.
    """
    if ts_utc.isna().all():
        raise ValueError("All timestamps are NaT after conversion.")
    t0 = ts_utc.min()
    if t0.tz is None:
        raise ValueError("t0 must be timezone-aware (UTC).")
    return t0


def assign_window_id(ts_utc: pd.Series, dt_minutes: int, t0: pd.Timestamp) -> pd.Series:
    """
    window_id = floor((ts - t0) / dt)
    """
    if dt_minutes <= 0:
        raise ValueError("dt_minutes must be positive.")
    if t0.tz is None:
        raise ValueError("t0 must be timezone-aware (UTC).")

    dt = pd.Timedelta(minutes=dt_minutes)
    delta = (ts_utc - t0)
    window_id = (delta / dt).apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    return window_id.astype("Int64")


def window_bounds(window_id: int, dt_minutes: int, t0: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (ts_start, ts_end) for a given window_id in UTC.
    """
    dt = pd.Timedelta(minutes=dt_minutes)
    ts_start = t0 + window_id * dt
    ts_end = ts_start + dt
    return ts_start, ts_end
