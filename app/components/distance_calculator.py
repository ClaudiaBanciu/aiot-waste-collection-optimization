import os
from abc import ABC, abstractmethod

import numpy as np
import requests


class DistanceCalculator(ABC):
    """Abstract base class for distance calculators.

    All methods work with points = [(lat, lon), ...].
    Subclasses must implement matrix() — Python will raise TypeError
    at instantiation time if they don't.
    """

    method: str = "unknown"

    @abstractmethod
    def matrix(self, points: list[tuple]) -> list[list[float]]:
        """Build and return the full N×N distance matrix (km)."""

    def route_length(self, points: list[tuple], matrix: list | None = None) -> float:
        """Sum of consecutive distances (km) in the GIVEN order of the points."""
        m = matrix if matrix is not None else self.matrix(points)
        return sum(m[i][i + 1] for i in range(len(points) - 1))

    def optimize(
        self,
        points: list[tuple],
        matrix: list | None = None,
        fixed_last: bool = False,
    ) -> tuple[list[int], float]:
        """Nearest-neighbour heuristic: start from the first point and always
        move to the closest not-yet-visited point.

        If fixed_last is True, the last point is treated as a fixed endpoint
        (e.g. a landfill/dump) and is always placed at the end — only the
        intermediate stops are reordered.

        Returns (order_indices, total_distance_km).
        """
        n = len(points)
        if n < 2:
            return list(range(n)), 0.0
        m = matrix if matrix is not None else self.matrix(points)
        last = n - 1 if fixed_last else None
        remaining = set(range(1, n - 1 if fixed_last else n))
        order, current, total = [0], 0, 0.0
        while remaining:
            next_pt = min(remaining, key=lambda j: m[current][j])
            total += m[current][next_pt]
            order.append(next_pt)
            remaining.discard(next_pt)
            current = next_pt
        if fixed_last and last is not None:
            total += m[current][last]
            order.append(last)
        return order, total

    def compare(self, points: list[tuple], fixed_last: bool = False) -> dict:
        """Build the matrix ONCE and return unoptimized distance, optimized
        distance, savings (%) and the optimal order of points.

        fixed_last — see optimize().
        """
        n = len(points)
        if n < 2:
            return {
                "method": self.method,
                "unoptimized_km": 0.0,
                "optimized_km": 0.0,
                "savings_%": 0.0,
                "optimal_order": list(range(n)),
            }
        m = self.matrix(points)
        unoptimized = self.route_length(points, m)
        order, optimized = self.optimize(points, m, fixed_last=fixed_last)
        savings = (1 - optimized / unoptimized) * 100 if unoptimized else 0.0
        return {
            "method": self.method,
            "unoptimized_km": unoptimized,
            "optimized_km": optimized,
            "savings_%": savings,
            "optimal_order": order,
        }


class StandardDistanceCalculator(DistanceCalculator):
    """Straight-line distance using the haversine formula."""

    method = "standard"
    R = 6371.0  # Earth's radius (km)

    def matrix(self, points: list[tuple]) -> list[list[float]]:
        pts = np.radians(np.asarray(points, dtype=float))
        lat = pts[:, 0][:, None]
        lon = pts[:, 1][:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
        )
        return (2 * self.R * np.arcsin(np.sqrt(a))).tolist()


class OSRMDistanceCalculator(DistanceCalculator):
    """Real road-network distance via the public OSRM server.

    Uses the Route service: a single request gives the road length of a trip
    through the given points in order. Long trips are split into overlapping
    chunks to stay within URL length limits.

    The optimized order is computed geometrically (nearest-neighbour on
    straight-line distances), then measured on the real road network.
    Falls back to StandardDistanceCalculator if the server is unreachable.
    """

    method = "osrm"

    def __init__(
        self,
        url: str | None = None,
        fallback: DistanceCalculator | None = None,
        chunk: int = 90,
    ):
        self.url = (
            url or os.environ.get("OSRM_URL", "https://router.project-osrm.org")
        ).rstrip("/")
        self.fallback = fallback or StandardDistanceCalculator()
        self.chunk = chunk
        self._available: bool | None = None

    def available(self) -> bool:
        """Check once per session whether the OSRM server is reachable."""
        if self._available is None:
            try:
                r = requests.get(
                    f"{self.url}/route/v1/driving/24.15,45.79;24.16,45.80"
                    "?overview=false",
                    timeout=6,
                )
                self._available = (
                    r.status_code == 200 and r.json().get("code") == "Ok"
                )
            except Exception:
                self._available = False
        return self._available

    def _segment(self, points: list[tuple]) -> tuple[float, list[tuple]]:
        """Single Route request: returns (road_length_km, geometry)."""
        coord = ";".join(f"{lon},{lat}" for lat, lon in points)
        url = f"{self.url}/route/v1/driving/{coord}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        route = r.json()["routes"][0]
        distance = route["distance"] / 1000
        # GeoJSON gives [lon, lat] → return as (lat, lon) for folium
        geometry = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
        return distance, geometry

    def road_route(self, points: list[tuple]) -> tuple[float, list[tuple]]:
        """Full road trip in the given order, split into overlapping chunks.
        Returns (total_distance_km, geometry)."""
        total, geometry, step = 0.0, [], self.chunk - 1
        for start in range(0, len(points) - 1, step):
            chunk = points[start : start + self.chunk]
            if len(chunk) >= 2:
                dist, geom = self._segment(chunk)
                total += dist
                geometry.extend(geom)
        return total, geometry

    def matrix(self, points: list[tuple]) -> list[list[float]]:
        """N×N road-distance matrix (km) via the OSRM table service.

        Uses the /table/v1 endpoint which returns all pairwise road distances
        in a single request — much more efficient than N² route calls and
        correct for nearest-neighbour optimization on real roads.

        Falls back to the straight-line matrix if the request fails.
        """
        try:
            coord = ";".join(f"{lon},{lat}" for lat, lon in points)
            url = f"{self.url}/table/v1/driving/{coord}?annotations=distance"
            r = requests.get(url, timeout=40)
            r.raise_for_status()
            raw = r.json().get("distances", [])
            if not raw:
                raise ValueError("Empty distance matrix from OSRM table.")
            return [[d / 1000 for d in row] for row in raw]
        except Exception:
            return self.fallback.matrix(points)

    def compare(self, points: list[tuple], fixed_last: bool = False) -> dict:
        n = len(points)
        if n < 2:
            return {
                "method": self.method,
                "unoptimized_km": 0.0,
                "optimized_km": 0.0,
                "savings_%": 0.0,
                "optimal_order": list(range(n)),
                "geom_unopt": [],
                "geom_opt": [],
            }
        try:
            # Build the road-distance matrix and optimize on it — nearest-neighbour
            # now minimizes real road distances, not straight-line distances.
            road_matrix = self.matrix(points)
            order, _ = self.optimize(points, road_matrix, fixed_last=fixed_last)

            # Measure full road distances with geometry for map rendering.
            unoptimized, geom_unopt = self.road_route(points)
            optimized, geom_opt = self.road_route([points[i] for i in order])
            self.method = "osrm"
        except Exception:
            result = self.fallback.compare(points, fixed_last=fixed_last)
            result["method"] = "standard (fallback)"
            result["geom_unopt"] = result["geom_opt"] = []
            return result
        savings = (1 - optimized / unoptimized) * 100 if unoptimized else 0.0
        return {
            "method": self.method,
            "unoptimized_km": unoptimized,
            "optimized_km": optimized,
            "savings_%": savings,
            "optimal_order": order,
            "geom_unopt": geom_unopt,
            "geom_opt": geom_opt,
        }
