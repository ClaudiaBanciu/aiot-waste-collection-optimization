"""
Pasul 5: Interfața web interactivă (Streamlit)
-------------------------------------------------
Instalare (o dată):
    pip install streamlit streamlit-folium

Rulare:
    streamlit run app/main.py   (sau: python -m streamlit run app/main.py)
"""

import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st
import sys, os

#import app.components 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.citeste_csv_robust import Read_Data
from app.components.fill_level import Fill_level
from app.components.distante import Distante
from app.components.interfata import run as run_interfata

st.set_page_config(page_title="Gestiune deșeuri - Sibiu", layout="wide")

INPUT_FILE = "data/processed/data_geocoded.csv"

@st.cache_data
def incarca_date(INPUT_FILE):
    df = Read_Data.citeste_csv_robust(INPUT_FILE)
    for col in ["Latitude", "Longitude"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Fill_num"] = Fill_level.normalizeaza_fill_level(df["fill_level"]).astype("Int64")
    df["ora_numerica"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    df["ora"] = df["Datetime"].dt.strftime("%H:%M")
    return df

df = incarca_date(INPUT_FILE)

if __name__ == "__main__":
    run_interfata(df)
