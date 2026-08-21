import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Roads are ~35% longer than straight-line distances on average.
# Applied in StandardDistanceCalculator so the fallback distances are
# a realistic proxy for road distances rather than a systematic underestimate.
_DETOUR_FACTOR = 1.35

# Seconds to wait before the 2nd and 3rd attempt on transient OSRM failures.
_RETRY_DELAYS = (1, 3)


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

    def compare(self, points: list[tuple]) -> dict:
        """Build the matrix once, run NearestNeighborSolver, return distances and optimal order."""
        from app.ml.nn.nearest_neighbor import NearestNeighborSolver
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
        path_names, optimized = NearestNeighborSolver().solve(
            np.array(m), [f"P{i}" for i in range(n)], verbose=False
        )
        order = [int(name[1:]) for name in path_names]
        savings = (1 - optimized / unoptimized) * 100 if unoptimized else 0.0
        return {
            "method": self.method,
            "unoptimized_km": unoptimized,
            "optimized_km": optimized,
            "savings_%": savings,
            "optimal_order": order,
        }


class StandardDistanceCalculator(DistanceCalculator):
    """Straight-line distance using the haversine formula, scaled by a detour
    factor to better approximate real road distances when used as a fallback."""

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
        return (2 * self.R * np.arcsin(np.sqrt(a)) * _DETOUR_FACTOR).tolist()


class OSRMDistanceCalculator(DistanceCalculator):
    """Real road-network distance via the public OSRM server.

    Uses the Route service: a single request gives the road length of a trip
    through the given points in order. Long trips are split into overlapping
    chunks to stay within URL length limits.

    The optimized order is computed on the real road-distance matrix, then
    measured on the road network with geometry for map rendering.
    Falls back to StandardDistanceCalculator if the server is unreachable.
    """

    method = "osrm"

    def __init__(
        self,
        url: str | None = None,
        fallback: DistanceCalculator | None = None,
        chunk: int = 90,
        timeout: tuple[float, float] = (6.0, 40.0),
    ):
        self.url = (
            url or os.environ.get("OSRM_URL", "https://router.project-osrm.org")
        ).rstrip("/")
        self.fallback = fallback or StandardDistanceCalculator()
        self.chunk = chunk
        self.timeout = timeout  # (connect_timeout, read_timeout) in seconds
        self._available: bool | None = None

    def available(self) -> bool:
        """Check once per session whether the OSRM server is reachable."""
        if self._available is None:
            try:
                r = requests.get(
                    f"{self.url}/route/v1/driving/24.15,45.79;24.16,45.80"
                    "?overview=false",
                    timeout=self.timeout[0],
                )
                self._available = (
                    r.status_code == 200 and r.json().get("code") == "Ok"
                )
            except Exception:
                self._available = False
        return self._available

    def _get(self, url: str) -> requests.Response:
        """GET with retries on transient network failures (timeout, connection error).
        Non-transient HTTP errors (4xx/5xx) surface immediately."""
        last_exc: Exception = RuntimeError("no attempts made")
        for delay in (None, *_RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                r = requests.get(url, timeout=self.timeout[1])
                r.raise_for_status()
                return r
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
        raise last_exc

    def _segment(self, points: list[tuple]) -> tuple[float, list[tuple]]:
        """Single Route request: returns (road_length_km, geometry)."""
        coord = ";".join(f"{lon},{lat}" for lat, lon in points)
        url = f"{self.url}/route/v1/driving/{coord}?overview=full&geometries=geojson"
        r = self._get(url)
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

        Falls back to the StandardDistanceCalculator matrix if the request fails.
        """
        try:
            coord = ";".join(f"{lon},{lat}" for lat, lon in points)
            url = f"{self.url}/table/v1/driving/{coord}?annotations=distance"
            r = self._get(url)
            raw = r.json().get("distances", [])
            if not raw:
                raise ValueError("Empty distance matrix from OSRM table.")
            return [[d / 1000 for d in row] for row in raw]
        except Exception:
            return self.fallback.matrix(points)

    def compare(self, points: list[tuple]) -> dict:
        from app.ml.nn.nearest_neighbor import NearestNeighborSolver
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
            road_matrix = self.matrix(points)
            path_names, _ = NearestNeighborSolver().solve(
                np.array(road_matrix), [f"P{i}" for i in range(n)], verbose=False
            )
            order = [int(name[1:]) for name in path_names]
            unoptimized, geom_unopt = self.road_route(points)
            optimized, geom_opt = self.road_route([points[i] for i in order])
        except Exception:
            result = self.fallback.compare(points)
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


_MATRIX_DIR = Path(__file__).resolve().parents[2] / "data" / "distance_matrices" / "google_maps"


class SequentialRouteDistanceCalculator:
    """Computes total driven distance from precomputed Google Maps distance
    matrices following stops in their original recorded order:
    depot → stop_1 → stop_2 → ... → landfill (no return trip).

    Matrix files must be generated first:
        python src/compute_distance_matrices.py
    """

    def __init__(self, matrix_files: list[str]):
        self.matrix_files = matrix_files

    @classmethod
    def for_car(cls, car: str) -> "SequentialRouteDistanceCalculator":
        """Convenience constructor: build from a single car name."""
        return cls([str(_MATRIX_DIR / f"{car}_distance_matrix.csv")])

    @staticmethod
    def available(car: str) -> bool:
        """True if a precomputed matrix CSV exists for this car."""
        return (_MATRIX_DIR / f"{car}_distance_matrix.csv").exists()

    def load_matrix(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, index_col=0)

    def calculate_sequential_distance(self, df: pd.DataFrame) -> float:
        """Sum consecutive legs: point_0 → point_1 → ... → point_n."""
        total_distance = 0.0
        num_points = len(df)
        for i in range(num_points - 1):
            distance = df.iloc[i, i + 1]
            total_distance += distance
        return total_distance

    def process_all(self) -> None:
        for file_path in self.matrix_files:
            df = self.load_matrix(file_path)
            total = self.calculate_sequential_distance(df)
            file_name = os.path.basename(file_path)
            print(f"🔹 {file_name}: Sequential total distance: {total:.2f} km")

    def distance_for_first_file(self) -> float:
        """Return the sequential distance for the first (and typically only) file."""
        df = self.load_matrix(self.matrix_files[0])
        return self.calculate_sequential_distance(df)
