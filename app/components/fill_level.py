import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
from geopy.distance import geodesic

import streamlit as st
import sys, os


def normalizeaza_fill_level(serie):
    """Fill Level poate fi text ('68%') sau, dacă fișierul a fost citit ca
    Excel, un număr zecimal (0.68) — normalizăm la un procent 0-100."""
    if serie.dtype == object:
        return pd.to_numeric(serie.astype(str).str.rstrip("%"), errors="coerce")
    numeric = pd.to_numeric(serie, errors="coerce")
    if numeric.max() <= 1:
        numeric = numeric * 100
    return numeric