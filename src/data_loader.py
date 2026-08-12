"""Loads and combines raw CSV files into a single DataFrame.

Run as a script to produce data/processed/data_combined.csv:
    python src/data_loader.py

Or import and call programmatically:
    from src.data_loader import DataLoader
    df = DataLoader().load()
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEPOT_END, DEPOT_START, FINAL_COLUMNS, RAW_FILES
from data_preprocess import CSVPreprocessor


class DataLoader:
    """Reads raw CSVs, attaches depot rows, and assigns simulated fill levels.

    Usage:
        loader = DataLoader()
        df = loader.load()

    Or with custom files:
        loader = DataLoader(raw_files=[("path/to/file.csv", 1)])
        df = loader.load()
    """

    OUTPUT_FILE: str = "data/processed/data_combined.csv"

    def __init__(self, raw_files: list[tuple[str, int]] | None = None):
        self.raw_files = raw_files if raw_files is not None else RAW_FILES

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read, clean, combine all routes and return the final DataFrame."""
        frames = [self._load_route(file_path, route_id)
                  for file_path, route_id in self.raw_files]
        combined = pd.concat(frames, ignore_index=True)
        combined = self._assign_fill_levels(combined)
        return combined[FINAL_COLUMNS]

    def save(self, df: pd.DataFrame, output: str | None = None) -> str:
        """Save the DataFrame to CSV and return the output path."""
        path = output or self.OUTPUT_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_route(self, file_path: str, route_id: int) -> pd.DataFrame:
        """Clean one CSV and wrap it with depot rows."""
        df = CSVPreprocessor(file_path).clean_data()
        df["route_id"] = route_id
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        car = df["Car"].iloc[0]
        return pd.concat(
            [
                self._depot_row(route_id, car, DEPOT_START),
                df,
                self._depot_row(route_id, car, DEPOT_END),
            ],
            ignore_index=True,
        )

    @staticmethod
    def _depot_row(route_id: int, car: str, address: str) -> pd.DataFrame:
        """Return a single-row DataFrame representing a depot stop."""
        return pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "Car": car,
                    "Datetime": pd.NaT,
                    "Id": None,
                    "Capacity": None,
                    "Address": address,
                    "fill_level": None,
                }
            ]
        )

    @staticmethod
    def _assign_fill_levels(df: pd.DataFrame) -> pd.DataFrame:
        """Assign random fill levels to real stops (depots stay None)."""
        result = df.copy()
        rng = np.random.default_rng(42)
        real_stops = result["Id"].notna()
        result.loc[real_stops, "fill_level"] = rng.integers(
            0, 101, size=real_stops.sum()
        )
        return result


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    path = loader.save(df)
    print(f"Saved {len(df)} rows → {path}")
