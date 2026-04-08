"""
Filter AIS dataframe to a given port using bbox or polygon from ports/ports.yaml.
"""
from __future__ import annotations

import yaml
import pandas as pd
from typing import Dict, Iterable, Tuple

from utils.geo import BBox


def _load_ports(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _point_in_poly(lon: float, lat: float, poly: Iterable[Tuple[float, float]]) -> bool:
    # Ray casting algorithm for lon/lat polygon
    inside = False
    pts = list(poly)
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        intersects = ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
        )
        if intersects:
            inside = not inside
    return inside


def filter_by_port(
    df: pd.DataFrame,
    port_name: str,
    ports_path: str,
    use_polygon: bool = False,
    bbox_override=None,
) -> pd.DataFrame:
    """
    Keep rows that fall within the port bbox/polygon.
    - If bbox_override is provided, it takes precedence.
    - Otherwise uses ports.yaml entry for the port.
    - If use_polygon is True and polygon exists, polygon test is used; else bbox filter.
    """
    ports = _load_ports(ports_path)
    entry = ports.get(port_name, {})
    if bbox_override is None and port_name not in ports:
        raise KeyError(f"port '{port_name}' not found in {ports_path}")

    bbox_vals = bbox_override if bbox_override else entry.get("bbox")
    if not bbox_vals or len(bbox_vals) != 4:
        raise ValueError(f"bbox missing/invalid for port '{port_name}'")
    bbox = BBox(*bbox_vals)

    if use_polygon and entry.get("polygon"):
        poly = [(float(p[0]), float(p[1])) for p in entry["polygon"]]
        mask = df.apply(lambda r: _point_in_poly(r["lon"], r["lat"], poly), axis=1)
        if not {"lat", "lon"}.issubset(df.columns):
            raise KeyError("df must have 'lat' and 'lon'")

    else:
        mask = (
            (df["lon"] >= bbox.min_lon)
            & (df["lon"] <= bbox.max_lon)
            & (df["lat"] >= bbox.min_lat)
            & (df["lat"] <= bbox.max_lat)
        )

    return df.loc[mask].reset_index(drop=True)
