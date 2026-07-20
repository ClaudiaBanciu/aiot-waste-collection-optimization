"""
Generează un tabel cu adresele NErezolvate, ca să le completezi manual.
--------------------------------------------------------------------------
Rulare:
    python genereaza_corectii_manuale.py

Creează corectii_manuale.csv cu coloanele: Address, Latitude, Longitude
(Latitude/Longitude goale — le completezi tu).

Cum găsești coordonatele unei adrese, manual:
  1. Deschide https://www.google.com/maps
  2. Caută adresa (ex. "Strada Pacii 2, Sibiu")
  3. Click dreapta pe locația exactă de pe hartă -> primul rând din
     meniu arată coordonatele (ex. 45.797123, 24.152456)
  4. Copiază-le în coloanele Latitude / Longitude din CSV, pentru rândul
     cu adresa respectivă.

După ce completezi fișierul, rulează:
    python aplica_corectii_manuale.py
"""

import json
import os
import pandas as pd

INPUT_FILE = "data/processed/dataset_geocodat.csv"          # sau dataset_clean.csv, dacă nu ai geocodat.csv
CACHE_FILE = "data/processed/geocoding_cache.json"
OUTPUT_TEMPLATE = "data/processed/corectii_manuale.csv"

# ---------------------------------------------------------------------
# Colectăm adresele fără coordonate (din dataset_geocodat.csv, dacă
# există; altfel din cache)
# ---------------------------------------------------------------------
adrese_negasite = []

if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)
    negasite = df[df["Latitude"].isna()]["Address"].dropna().unique()
    adrese_negasite = sorted(negasite)
elif os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    adrese_negasite = sorted(a for a, c in cache.items() if c[0] is None)
else:
    raise SystemExit("Nu găsesc nici dataset_geocodat.csv, nici geocoding_cache.json.")

print(f"Adrese fără coordonate: {len(adrese_negasite)}")

# ---------------------------------------------------------------------
# Dacă fișierul de corecții există deja, păstrăm ce a completat userul
# și adăugăm doar adresele noi (ca să nu suprascriem munca deja făcută)
# ---------------------------------------------------------------------
if os.path.exists(OUTPUT_TEMPLATE):
    existent = pd.read_csv(OUTPUT_TEMPLATE)
    deja_in_tabel = set(existent["Address"])
    randuri_noi = [
        {"Address": a, "Latitude": "", "Longitude": ""}
        for a in adrese_negasite if a not in deja_in_tabel
    ]
    if randuri_noi:
        existent = pd.concat([existent, pd.DataFrame(randuri_noi)], ignore_index=True)
        existent.to_csv(OUTPUT_TEMPLATE, index=False, encoding="utf-8-sig")
        print(f"Am adăugat {len(randuri_noi)} adrese noi la {OUTPUT_TEMPLATE} existent.")
    else:
        print(f"{OUTPUT_TEMPLATE} există deja și e la zi — nimic de adăugat.")
else:
    tabel = pd.DataFrame({
        "Address": adrese_negasite,
        "Latitude": ["" for _ in adrese_negasite],
        "Longitude": ["" for _ in adrese_negasite],
    })
    tabel.to_csv(OUTPUT_TEMPLATE, index=False, encoding="utf-8-sig")
    print(f"Am creat {OUTPUT_TEMPLATE} — completează manual Latitude/Longitude, "
          f"apoi rulează aplica_corectii_manuale.py")