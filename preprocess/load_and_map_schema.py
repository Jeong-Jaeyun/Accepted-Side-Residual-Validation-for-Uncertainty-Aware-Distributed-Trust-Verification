"""
Load raw AIS CSV, map columns to a standard schema, and sanitize values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

REQUIRED_STD_COLS = ["ts", "mmsi", "lat", "lon", "sog", "cog"]


@dataclass(frozen=True)
class SchemaMapping:
    """Map raw column names -> standardized names used downstream."""

    ts: str
    mmsi: str
    lat: str
    lon: str
    sog: str
    cog: str
    heading: Optional[str] = None
    nav_status: Optional[str] = None


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load raw AIS CSV with flexible dtype inference."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded CSV is empty: {path}")
    return df


def map_schema(df: pd.DataFrame, mapping: SchemaMapping) -> pd.DataFrame:
    """Rename raw columns to standardized names and keep only relevant columns."""
    raw_cols = df.columns.tolist()
    needed = {
        mapping.ts: "ts",
        mapping.mmsi: "mmsi",
        mapping.lat: "lat",
        mapping.lon: "lon",
        mapping.sog: "sog",
        mapping.cog: "cog",
    }
    if mapping.heading:
        needed[mapping.heading] = "heading"
    if mapping.nav_status:
        needed[mapping.nav_status] = "nav_status"

    missing = [c for c in needed.keys() if c not in raw_cols]
    if missing:
        raise KeyError(f"Missing raw columns: {missing}. Available columns: {raw_cols[:50]}")

    out = df[list(needed.keys())].rename(columns=needed)
    return out


def sanitize(df: pd.DataFrame, tz_hint: str = "UTC") -> pd.DataFrame:
    """
    Enforce types and drop obviously invalid rows.
    - ts: timezone-aware UTC datetime
    - mmsi: string
    - lat/lon: float within bounds
    - sog: 0..102.3 (AIS spec special value 102.2)
    - cog: 0..360
    """
    from utils.time import to_datetime_utc  # local import to avoid cycles

    df = df.copy()

    df["ts"] = to_datetime_utc(df["ts"], tz_hint=tz_hint)
    df["mmsi"] = df["mmsi"].astype(str).str.strip()

    for col in ["lat", "lon", "sog", "cog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts", "mmsi", "lat", "lon", "sog", "cog"])

    df = df[(df["lat"] >= -90.0) & (df["lat"] <= 90.0)]
    df = df[(df["lon"] >= -180.0) & (df["lon"] <= 180.0)]
    df = df[(df["sog"] >= 0.0) & (df["sog"] <= 102.3)]
    df = df[(df["cog"] >= 0.0) & (df["cog"] <= 360.0)]

    if "heading" in df.columns:
        df["heading"] = pd.to_numeric(df["heading"], errors="coerce")
    if "nav_status" in df.columns:
        df["nav_status"] = df["nav_status"].astype(str).str.strip()

    df = df.sort_values(["mmsi", "ts"]).reset_index(drop=True)
    return df


def load_and_prepare(csv_path: str, mapping_dict: Dict[str, Any], tz_hint: str = "UTC") -> pd.DataFrame:
    """Helper: raw csv -> schema map -> sanitize."""
    mapping = SchemaMapping(**mapping_dict)
    df_raw = load_raw_csv(csv_path)
    df_std = map_schema(df_raw, mapping)
    df_clean = sanitize(df_std, tz_hint=tz_hint)

    for col in REQUIRED_STD_COLS:
        if col not in df_clean.columns:
            raise RuntimeError(f"Standard column missing after sanitize: {col}")

    return df_clean
