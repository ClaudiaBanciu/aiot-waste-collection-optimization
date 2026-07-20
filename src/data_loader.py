import os
import numpy as np
import pandas as pd
from data_preprocess import CSVPreprocessor

RAW_FILES = [
    ("data/raw/SB25SOM.csv", 1),
    ("data/raw/SB30SOM.csv", 2),
    ("data/raw/SB45SOM.csv", 3),
]

FINAL_COLUMNS = ["route_id", "Car", "Datetime", "Id", "Capacity", "Address", "fill_level"]

DEPOT_START = "Strada Șelimbărului 90, Cisnădie, Romania"
DEPOT_END   = "DN1 FN, Cristian 557085"


def _depot_row(route_id: int, car: str, address: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "route_id":   route_id,
        "Car":        car,
        "Datetime":   pd.NaT,
        "Id":         None,
        "Capacity":   None,
        "Address":    address,
        "fill_level": None,
    }])


def load_and_combine() -> pd.DataFrame:
    frames = []
    for file_path, route_id in RAW_FILES:
        processor = CSVPreprocessor(file_path)
        df = processor.clean_data()
        df["route_id"] = route_id
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        car = df["Car"].iloc[0]

        df = pd.concat([
            _depot_row(route_id, car, DEPOT_START),
            df,
            _depot_row(route_id, car, DEPOT_END),
        ], ignore_index=True)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    np.random.seed(42)
    mask = combined["fill_level"].isna()
    combined.loc[mask, "fill_level"] = None  # depot rows stay None

    non_depot = combined["fill_level"].isna() == False
    combined.loc[~mask & False, "fill_level"] = np.random.randint(  # placeholder
        0, 101, size=(~mask).sum()
    )

    # assign fill_level only to real stops (not depot rows)
    real = combined["Id"].notna()
    combined.loc[real, "fill_level"] = np.random.randint(0, 101, size=real.sum())

    return combined[FINAL_COLUMNS]


if __name__ == "__main__":
    df = load_and_combine()
    out = "data/processed/data_combined.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows → {out}")