"""
Aplică corecțiile manuale (din corectii_manuale.csv) peste cache și dataset.
--------------------------------------------------------------------------
Rulare (după ce ai completat corectii_manuale.csv):
    python aplica_corectii_manuale.py

Actualizează geocoding_cache.json cu valorile completate manual, apoi
regenerează dataset_geocodat.csv și dataset_geocodat_sibiu.csv.
"""

import json
import os
import pandas as pd

INPUT_FILE = "data/processed/dataset_clean.csv"
OUTPUT_FILE = "data/processed/dataset_geocodat.csv"
OUTPUT_FILE_SIBIU = "data/processed/dataset_geocodat_sibiu.csv"
CACHE_FILE = "data/processed/geocoding_cache.json"
CORECTII_FILE = "data/processed/corectii_manuale.csv"

if not os.path.exists(CORECTII_FILE):
    raise SystemExit(f"Nu găsesc {CORECTII_FILE}. Rulează întâi genereaza_corectii_manuale.py.")

def citeste_csv_robust(cale):
    """Citește corectii_manuale.csv indiferent cum l-a salvat Excel:
    - uneori Excel salvează de fapt un .xlsx cu extensia .csv (semnătură ZIP)
    - alteori salvează CSV cu encoding cp1252, separator ';' și virgulă zecimală
    Încercăm toate variantele plauzibile."""
    with open(cale, "rb") as f:
        semnatura = f.read(4)

    if semnatura[:2] == b"PK":
        # E de fapt un fișier Excel (.xlsx), doar cu extensia .csv
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


def normalizeaza_numar(valoare):
    """Convertește '45,797123' (virgulă zecimală) în 45.797123 (float)."""
    if pd.isna(valoare):
        return None
    s = str(valoare).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


corectii = citeste_csv_robust(CORECTII_FILE)
corectii["Latitude"] = corectii["Latitude"].apply(normalizeaza_numar)
corectii["Longitude"] = corectii["Longitude"].apply(normalizeaza_numar)

with open(CACHE_FILE, "r", encoding="utf-8") as f:
    cache = json.load(f)

aplicate = 0
sarite_goale = 0

for _, r in corectii.iterrows():
    adresa = r["Address"]
    lat = r["Latitude"]
    lon = r["Longitude"]

    if lat is None or lon is None:
        sarite_goale += 1
        continue

    cache[adresa] = [float(lat), float(lon)]
    aplicate += 1

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"Corecții aplicate: {aplicate}")
print(f"Rânduri lăsate goale în {CORECTII_FILE} (neaplicate): {sarite_goale}")

# ---------------------------------------------------------------------
# Regenerăm dataset_geocodat.csv și dataset_geocodat_sibiu.csv
# ---------------------------------------------------------------------
try:
    df = pd.read_excel(INPUT_FILE)
except Exception:
    df = pd.read_csv(INPUT_FILE)

df["Latitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[0])
df["Longitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[1])
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

df_sibiu = df.dropna(subset=["Latitude", "Longitude"]).copy()
df_sibiu.to_csv(OUTPUT_FILE_SIBIU, index=False, encoding="utf-8-sig")

print(f"\nTotal rânduri: {len(df)}")
print(f"Rânduri cu coordonate valide: {len(df_sibiu)}")
print(f"Fișiere regenerate: {OUTPUT_FILE}, {OUTPUT_FILE_SIBIU}")