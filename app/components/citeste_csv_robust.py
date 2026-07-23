import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st
import sys, os



class Read_Data:
    def __init__(self, file_path):
        self.file_path = file_path
        #self.df = self.citeste_csv_robust(file_path)

    @staticmethod
    def normalizeaza_fill_level(serie):
        """Fill Level poate fi text ('68%') sau, dacă fișierul a fost citit ca
        Excel, un număr zecimal (0.68) — normalizăm la un procent 0-100."""
        if serie.dtype == object:
            return pd.to_numeric(serie.astype(str).str.rstrip("%"), errors="coerce")
        numeric = pd.to_numeric(serie, errors="coerce")
        if numeric.max() <= 1:
            numeric = numeric * 100
        return numeric

    @staticmethod
    def citeste_csv_robust(cale):
        """Citește CSV-ul indiferent cum a fost salvat (UTF-8, cp1252,
        separator ',' sau ';', sau chiar dacă e de fapt un .xlsx redenumit)."""
        with open(cale, "rb") as f:
            semnatura = f.read(4)
        if semnatura[:2] == b"PK":
            return pd.read_excel(cale)
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            for sep in [",", ";"]:
                try:
                    df = pd.read_csv(cale, encoding=enc, sep=sep)
                    if "Address" in df.columns:
                        return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        raise SystemExit(f"Nu am putut citi {cale} cu niciun encoding/separator cunoscut.")
