"""
Calculul distanței neoptimizate și optimizate (linie dreaptă, geopy)
--------------------------------------------------------------------------
Pentru fiecare mașină (Car), luăm containerele în ordinea în care apar în
date (ordonate după Datetime) și calculăm distanța totală, în linie
dreaptă (geodezică), folosind geopy.

Rulare:
    python distante.py
"""

import pandas as pd
from geopy.distance import geodesic

INPUT_FILE = "data/processed/data_geocoded.csv"


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


def distanta_traseu_km(puncte):
    """Suma distanțelor consecutive (km), în ordinea dată a punctelor."""
    total = 0.0
    for i in range(len(puncte) - 1):
        total += geodesic(puncte[i], puncte[i + 1]).km
    return total


def optimizeaza_nearest_neighbor_km(puncte):
    """Euristică greedy 'cel mai apropiat vecin'."""
    if len(puncte) < 2:
        return puncte, 0.0

    ramase = list(range(1, len(puncte)))
    ordine = [0]
    curent = 0
    total = 0.0

    while ramase:
        distante = [(j, geodesic(puncte[curent], puncte[j]).km) for j in ramase]
        urmator, dist = min(distante, key=lambda t: t[1])
        total += dist
        ordine.append(urmator)
        ramase.remove(urmator)
        curent = urmator

    traseu_optimizat = [puncte[i] for i in ordine]
    return traseu_optimizat, total


df = citeste_csv_robust(INPUT_FILE)
for col in ["Latitude", "Longitude"]:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
df = df.dropna(subset=["Latitude", "Longitude"])
df["Datetime"] = pd.to_datetime(df["Datetime"])

print("Distanța neoptimizată, per mașină (ordinea cronologică din date):\n")

distante_totale = {}

for masina, grup in df.groupby("Car"):
    grup = grup.sort_values("Datetime").reset_index(drop=True)
    puncte = list(zip(grup["Latitude"], grup["Longitude"]))

    distanta_totala_km = distanta_traseu_km(puncte)

    distante_totale[masina] = distanta_totala_km
    print(f"  {masina}: {len(grup)} opriri -> {distanta_totala_km:.2f} km")

print(f"\nTotal (toate mașinile): {sum(distante_totale.values()):.2f} km")