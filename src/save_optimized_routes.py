"""Save NN-optimized routes for all 3 distance methods (Google Maps, Standard, OSRM).

One txt file per car × method in outputs/routes/:
    SB-25-SOM_route_google_maps.txt
    SB-25-SOM_route_standard.txt
    SB-25-SOM_route_osrm.txt        (skipped if OSRM server is unreachable)
    ... (same for SB-30-SOM, SB-45-SOM)

Each file contains:
  - full point path  (Point_1 -> Point_5 -> ... -> Point_N)
  - total optimized distance
  - per-stop block with point label, container ID, address, and capacity

Run from the project root:
    python3 src/save_optimized_routes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.distance_calculator import (
    OSRMDistanceCalculator,
    StandardDistanceCalculator,
)
from app.ml.nn.nearest_neighbor import NearestNeighborSolver

GEOCODED_CSV = PROJECT_ROOT / "data" / "processed" / "data_geocoded.csv"
GM_MATRIX_DIR = PROJECT_ROOT / "data" / "distance_matrices" / "google_maps"
OSRM_MATRIX_DIR = PROJECT_ROOT / "data" / "distance_matrices" / "osrm"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "routes"
CARS = ["SB-25-SOM", "SB-30-SOM", "SB-45-SOM"]


def _format_path(path: list[str], max_nodes: int = 500) -> str:
    if len(path) <= max_nodes:
        return " -> ".join(path)
    half = max_nodes // 2
    return " -> ".join(path[:half]) + " -> ... -> " + " -> ".join(path[-half:])


def _write_route(
    f,
    car: str,
    method: str,
    path: list[str],
    total_km: float,
    car_df: pd.DataFrame,
) -> None:
    """Write the route block into an already-open file handle."""
    f.write(f"Optimized route — {car}  ({method})\n")
    f.write(f"Total distance : {total_km:.3f} km\n")
    f.write(f"Points         : {len(path)}\n")
    f.write(f"Path           : {_format_path(path)}\n")
    f.write("=" * 70 + "\n\n")

    indices = [int(name.split("_")[1]) - 1 for name in path]
    ordered = car_df.iloc[indices].reset_index(drop=True)

    for stop_num, (point_label, (_, row)) in enumerate(
        zip(path, ordered.iterrows()), start=1
    ):
        if pd.isna(row["Id"]):
            kind = "DEPOT" if stop_num == 1 else "LANDFILL"
            f.write(f"Stop {stop_num:>3}  {point_label:<12}  [{kind}]\n")
        else:
            f.write(
                f"Stop {stop_num:>3}  {point_label:<12}  "
                f"[{row['Id']}]  cap:{row['Capacity']}\n"
            )
        f.write(f"              {row['Address']}\n\n")


def save_google_maps(car: str, car_df: pd.DataFrame) -> bool:
    matrix_path = GM_MATRIX_DIR / f"{car}_distance_matrix.csv"
    if not matrix_path.exists():
        print(f"[{car}] Google Maps matrix not found — skipping.")
        return False

    path, total_km = NearestNeighborSolver().from_file(str(matrix_path), verbose=False)
    if path is None:
        print(f"[{car}] Google Maps NN failed — skipping.")
        return False

    out = OUTPUT_DIR / f"{car}_route_google_maps.txt"
    with open(out, "w", encoding="utf-8") as f:
        _write_route(f, car, "Google Maps NN", path, total_km, car_df)
    print(f"[{car}] google_maps → {out.name}")
    return True


def save_standard(car: str, car_df: pd.DataFrame) -> None:
    points = list(zip(car_df["Latitude"], car_df["Longitude"]))
    n = len(points)
    point_labels = [f"Point_{i + 1}" for i in range(n)]

    calc = StandardDistanceCalculator()
    matrix = np.array(calc.matrix(points))
    path, total_km = NearestNeighborSolver().solve(matrix, point_labels, verbose=False)
    if path is None:
        print(f"[{car}] Standard NN failed — skipping.")
        return

    out = OUTPUT_DIR / f"{car}_route_standard.txt"
    with open(out, "w", encoding="utf-8") as f:
        _write_route(f, car, "Standard (haversine × 1.35)", path, total_km, car_df)
    print(f"[{car}] standard    → {out.name}")


def save_osrm(car: str, car_df: pd.DataFrame) -> None:
    osrm = OSRMDistanceCalculator()
    if not osrm.available():
        print(f"[{car}] OSRM server unreachable — skipping.")
        return

    points = list(zip(car_df["Latitude"], car_df["Longitude"]))
    n = len(points)
    point_labels = [f"Point_{i + 1}" for i in range(n)]

    matrix = np.array(osrm.matrix(points))

    # Persist the matrix so the app reuses it instead of re-querying OSRM.
    OSRM_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=point_labels, columns=point_labels).to_csv(
        OSRM_MATRIX_DIR / f"{car}_distance_matrix.csv"
    )

    path, total_km = NearestNeighborSolver().solve(matrix, point_labels, verbose=False)
    if path is None:
        print(f"[{car}] OSRM NN failed — skipping.")
        return

    out = OUTPUT_DIR / f"{car}_route_osrm.txt"
    with open(out, "w", encoding="utf-8") as f:
        _write_route(f, car, "OSRM (road network)", path, total_km, car_df)
    print(f"[{car}] osrm        → {out.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(GEOCODED_CSV)

    for car in CARS:
        car_df = df_all[df_all["Car"] == car].reset_index(drop=True)
        print(f"\n── {car} ({len(car_df)} points) ──")
        save_google_maps(car, car_df)
        save_standard(car, car_df)
        save_osrm(car, car_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
