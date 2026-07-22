import pandas as pd
import streamlit as st

INPUT_FILE = "data/processed/data_geocoded.csv"

def citeste_csv_robust(cale):
    """Citește CSV-ul indiferent cum a fost salvat."""
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

def normalizeaza_fill_level(serie):
    """Normalizăm la un procent 0-100."""
    if serie.dtype == object:
        return pd.to_numeric(serie.astype(str).str.rstrip("%"), errors="coerce")
    numeric = pd.to_numeric(serie, errors="coerce")
    if numeric.max() <= 1:
        numeric = numeric * 100
    return numeric

@st.cache_data
def incarca_date():
    """Funcția principală apelată de interfață pentru a obține datele."""
    df = citeste_csv_robust(INPUT_FILE)
    for col in ["Latitude", "Longitude"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Fill_num"] = normalizeaza_fill_level(df["fill_level"]).astype("Int64")
    df["ora_numerica"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    df["ora"] = df["Datetime"].dt.strftime("%H:%M")
    return df