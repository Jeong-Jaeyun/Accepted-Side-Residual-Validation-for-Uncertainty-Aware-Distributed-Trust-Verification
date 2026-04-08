# utils/geo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import math


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points (degrees) in kilometers.
    """
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@dataclass(frozen=True)
class BBox:
    """
    Bounding box in lon/lat space.
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon) and (self.min_lat <= lat <= self.max_lat)

    def width(self) -> float:
        return self.max_lon - self.min_lon

    def height(self) -> float:
        return self.max_lat - self.min_lat


def grid_index(
    lat: float,
    lon: float,
    bbox: BBox,
    nx: int,
    ny: int,
    clamp: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Map (lat, lon) to (gx, gy) for a bbox-based uniform grid.

    - gx in [0, nx-1] for longitude
    - gy in [0, ny-1] for latitude

    If clamp=False and the point is outside bbox, returns (None, None).
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")

    inside = bbox.contains(lat, lon)
    if (not inside) and (not clamp):
        return None, None

    # Normalize to [0, 1]
    x = (lon - bbox.min_lon) / (bbox.width() + 1e-15)
    y = (lat - bbox.min_lat) / (bbox.height() + 1e-15)

    # Convert to indices
    gx = int(math.floor(x * nx))
    gy = int(math.floor(y * ny))

    # Points on the max edge fall into nx/ny; clamp to nx-1/ny-1
    if clamp:
        gx = min(max(gx, 0), nx - 1)
        gy = min(max(gy, 0), ny - 1)
    else:
        if not (0 <= gx < nx and 0 <= gy < ny):
            return None, None

    return gx, gy
