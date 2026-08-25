"""
Streamlit application entry point.

Run with:
    streamlit run app/main.py
    # or:
    python -m streamlit run app/main.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.data_reader import DataReader
from app.components.fill_level import FillLevel
from app.components.interface import run
from app.components.predicted_bins import add_matrix_point
from config import INPUT_FILE

st.set_page_config(page_title="Waste Management - Sibiu", layout="wide")


@st.cache_data
def load_data(input_file: str) -> pd.DataFrame:
    """Read, clean, and enrich the geocoded CSV.

    Steps:
      1. Read with DataReader (auto-detects encoding and separator).
      2. Number the rows against the Google Maps distance matrices.
      3. Coerce Latitude / Longitude to float.
      4. Drop rows with missing coordinates.
      5. Parse Datetime.
      6. Normalise fill level via FillLevel (adds Fill_num column).
      7. Derive time_numeric (decimal hours) and time (HH:MM string).
    """
    # 1. Read
    df = DataReader(input_file).read()

    # 2. Link each row to its row in the car's Google Maps matrix. Must happen
    #    while the frame is still in file order — that ordering is the
    #    numbering the saved matrices use.
    df = add_matrix_point(df)

    # 3. Coerce coordinates
    for col in ("Latitude", "Longitude"):
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    # 4. Drop missing coordinates
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # 5. Parse Datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # 6. Normalise fill level — FillLevel adds the Fill_num column in-place
    df = FillLevel(df).data

    # 7. Time helpers used in interface.py and webscript.py
    df["time_numeric"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    df["time"] = df["Datetime"].dt.strftime("%H:%M")

    return df


df = load_data(INPUT_FILE)

if __name__ == "__main__":
    run(df)
