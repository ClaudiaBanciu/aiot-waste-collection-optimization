import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
from geopy.distance import geodesic



class Fill_level:
    def __init__(self, df):
        self.df = df
        self.df["Fill Level"] = self.normalizeaza_fill_level(self.df["Fill Level"])

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