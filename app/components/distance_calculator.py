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

# The public OSRM server rejects a /table request with "TooBig" once
# sources × destinations exceeds 10,000 (its max-table-size of 100, squared).
# A 100×100 block sits exactly on that cap while keeping the URL near 5 KB,
# comfortably under the usual 8 KB header limit.
_TABLE_BLOCK = 100

# Courtesy pause between block requests so the shared demo server isn't hammered.
_TABLE_PAUSE = 0.2


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
        table_block: int = _TABLE_BLOCK,
        table_pause: float = _TABLE_PAUSE,
    ):
        self.url = (
            url or os.environ.get("OSRM_URL", "https://router.project-osrm.org")
        ).rstrip("/")
        self.fallback = fallback or StandardDistanceCalculator()
        self.chunk = chunk
        self.timeout = timeout  # (connect_timeout, read_timeout) in seconds
        # Rows/columns per /table request. Raise this when pointing at your own
        # OSRM instance started with a larger --max-table-size.
        self.table_block = table_block
        self.table_pause = table_pause
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
        """Single Route request: returns (road_length_km, geometry).

        `continue_straight=false` is essential here. It defaults to true, which
        forbids a U-turn at each waypoint and sends the route around the block
        instead — inflating a dense collection round by ~10%. The /table matrix
        the route order is optimized on has no such constraint, so leaving the
        default in place would measure a different route than the one optimized.
        A collection truck can reverse direction at a stop, so false is also the
        physically correct choice.
        """
        coord = ";".join(f"{lon},{lat}" for lat, lon in points)
        url = (
            f"{self.url}/route/v1/driving/{coord}"
            "?overview=full&geometries=geojson&continue_straight=false"
        )
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

    def _table_block(
        self,
        points: list[tuple],
        rows: list[int],
        cols: list[int],
    ) -> list[list[float | None]]:
        """One /table request for the sub-matrix rows × cols (metres).

        Only the coordinates actually needed are sent, and `sources` /
        `destinations` select which of them are origins and which are
        destinations. This keeps both the cell count (rows × cols) and the
        URL length within the server's limits.
        """
        if rows == cols:
            # Diagonal block — send each coordinate once, sources/destinations
            # both default to "all".
            sel = rows
            query = "annotations=distance"
        else:
            sel = rows + cols
            src = ";".join(str(k) for k in range(len(rows)))
            dst = ";".join(str(k) for k in range(len(rows), len(sel)))
            query = f"annotations=distance&sources={src}&destinations={dst}"

        coord = ";".join(f"{points[i][1]},{points[i][0]}" for i in sel)
        r = self._get(f"{self.url}/table/v1/driving/{coord}?{query}")
        raw = r.json().get("distances")
        if not raw:
            raise ValueError("Empty distance matrix from OSRM table.")
        return raw

    def matrix(self, points: list[tuple]) -> list[list[float]]:
        """N×N road-distance matrix (km) via the OSRM table service.

        The public OSRM server caps a table request at
        `sources × destinations <= _TABLE_MAX_CELLS`, so anything past ~100
        points fails as `TooBig` in a single request. The matrix is therefore
        assembled from square blocks of at most `_TABLE_BLOCK` rows/columns,
        which keeps every request inside both that cap and the server's URL
        length limit.

        Unroutable pairs come back as null and are filled from the haversine
        fallback. If a whole block fails, only that block falls back, so a
        transient error no longer silently degrades the entire matrix.
        """
        n = len(points)
        if n == 0:
            return []

        fallback: list[list[float]] | None = None

        def fallback_matrix() -> list[list[float]]:
            nonlocal fallback
            if fallback is None:
                fallback = self.fallback.matrix(points)
            return fallback

        result = [[0.0] * n for _ in range(n)]
        block = self.table_block

        for i0 in range(0, n, block):
            rows = list(range(i0, min(i0 + block, n)))
            for j0 in range(0, n, block):
                cols = list(range(j0, min(j0 + block, n)))
                try:
                    raw = self._table_block(points, rows, cols)
                except Exception:
                    fb = fallback_matrix()
                    for a, i in enumerate(rows):
                        for b, j in enumerate(cols):
                            result[i][j] = fb[i][j]
                    continue

                for a, i in enumerate(rows):
                    for b, j in enumerate(cols):
                        d = raw[a][b]
                        # null = no route found between this pair
                        result[i][j] = (
                            d / 1000 if d is not None else fallback_matrix()[i][j]
                        )

                if self.table_pause:
                    time.sleep(self.table_pause)

        return result

    def snap_report(self, points: list[tuple], threshold_m: float = 100.0) -> list[dict]:
        """Flag points that sit far from any drivable road.

        OSRM snaps every coordinate to the nearest road before routing. A large
        snap distance means the coordinate is not where the address says it is —
        usually a bad geocode — and it silently produces long phantom legs.
        Returns one entry per point exceeding `threshold_m`, worst first.

        Uses the /nearest service, one request per point, so call it on a
        route's points once rather than in a hot loop.
        """
        flagged = []
        for i, (lat, lon) in enumerate(points):
            try:
                r = self._get(f"{self.url}/nearest/v1/driving/{lon},{lat}?number=1")
                wp = r.json()["waypoints"][0]
            except Exception:
                continue
            if wp["distance"] > threshold_m:
                flagged.append({
                    "index": i,
                    "input": (lat, lon),
                    "snapped_to": wp.get("name") or "(unnamed road)",
                    "snap_distance_m": wp["distance"],
                })
            if self.table_pause:
                time.sleep(self.table_pause)
        return sorted(flagged, key=lambda d: -d["snap_distance_m"])

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
