import os
import time
from pathlib import Path

import pandas as pd
import googlemaps
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "data_geocoded.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "distance_matrices"


class DistanceMatrix:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)

    def compute_distance_matrix(self, locations):
        """Computes the distance matrix (in kilometers) between all given locations."""
        matrix = []

        for i, origin in enumerate(locations):
            row = []
            for j in range(0, len(locations), 25):  # Google API limit
                chunk = locations[j:j + 25]
                try:
                    result = self.gmaps.distance_matrix(
                        origins=[origin], destinations=chunk, mode="driving"
                    )
                    distances = [
                        el["distance"]["value"] / 1000 if el["status"] == "OK" else float("inf")
                        for el in result["rows"][0]["elements"]
                    ]
                    row.extend(distances)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error fetching distances from {origin}: {e}")
                    row.extend([float("inf")] * len(chunk))
            matrix.append(row)
            print(f"  Row {i + 1}/{len(locations)} done")

        return matrix

    def save_distance_matrix(self, matrix, locations, output_path):
        """Saves the distance matrix to a CSV file with appropriate labels."""
        labels = [f"Point_{i + 1}" for i in range(len(locations))]
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.to_csv(output_path)
        print(f"Distance matrix saved to: {output_path}")


if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_MAPS_DISTANCE_MATRIX_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_DISTANCE_MATRIX_KEY not set in .env")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    calculator = DistanceMatrix(api_key)

    for car, group in df.groupby("Car"):
        locations = list(zip(group["Latitude"], group["Longitude"]))
        formatted = [f"{lat},{lon}" for lat, lon in locations]

        print(f"\n[{car}] Computing matrix for {len(formatted)} locations…")
        matrix = calculator.compute_distance_matrix(formatted)

        output_path = OUTPUT_DIR / f"{car}_distance_matrix.csv"
        calculator.save_distance_matrix(matrix, formatted, output_path)
