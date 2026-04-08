"""
Map lat/lon to uniform grid indices gx, gy within a port bbox.
"""
from __future__ import annotations

import pandas as pd
from utils.geo import BBox, grid_index


def add_grid_indices(df: pd.DataFrame, bbox_vals, nx: int, ny: int, clamp: bool = True) -> pd.DataFrame:
    if not {"lat","lon"}.issubset(df.columns):
        raise KeyError("df must have 'lat' and 'lon'")
    if len(bbox_vals) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
    bbox = BBox(*bbox_vals)

    def _map(row):
        gx, gy = grid_index(row["lat"], row["lon"], bbox, nx, ny, clamp=clamp)
        return pd.Series({"gx": gx, "gy": gy})

    df = df.copy()
    grid_df = df.apply(_map, axis=1)
    
    df[["gx", "gy"]] = grid_df
    df = df.dropna(subset=["gx", "gy"])
    df[["gx", "gy"]] = df[["gx", "gy"]].astype(int)
    return df
