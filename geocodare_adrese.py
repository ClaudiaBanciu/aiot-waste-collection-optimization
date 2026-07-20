"""
Pasul 1 + Pasul 2: Geocodarea adreselor CU CACHE
--------------------------------------------------
Geocodează fiecare adresă unică o singură dată în viață. Rezultatele
se salvează în geocoding_cache.json. La fiecare rulare ulterioară,
scriptul verifică întâi cache-ul și face cereri către Nominatim DOAR
pentru adresele noi, negăsite încă în cache.

Instalare (o singură dată):
    pip install geopy folium streamlit streamlit-folium pandas openpyxl
"""

import json
import os
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

INPUT_FILE = "data/processed/dataset_clean.csv"
OUTPUT_FILE = "data/processed/dataset_geocodat.csv"
CACHE_FILE = "data/processed/geocoding_cache.json"

# ---------------------------------------------------------------------
# 1. Citirea fișierului sursă
# ---------------------------------------------------------------------
try:
    df = pd.read_excel(INPUT_FILE)
except Exception:
    df = pd.read_csv(INPUT_FILE)

print(f"Am încărcat {len(df)} rânduri, {df['Address'].nunique()} adrese unice.")

# ---------------------------------------------------------------------
# 2. Încărcăm cache-ul existent (dacă există deja pe disc)
# ---------------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache găsit: {len(cache)} adrese deja geocodate anterior.")
else:
    cache = {}
    print("Nu există cache încă — pornim de la zero.")


def salveaza_cache():
    """Scrie cache-ul curent pe disc (apelată după fiecare cerere nouă,
    ca să nu pierdem progresul dacă scriptul e întrerupt)."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 3. Geocoder Nominatim, cu limită de 1 cerere/secundă
# ---------------------------------------------------------------------
geolocator = Nominatim(user_agent="proiect_gestiune_deseuri_sibiu")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# ---------------------------------------------------------------------
# 4. Geocodăm DOAR adresele care nu sunt deja în cache
# ---------------------------------------------------------------------
adrese_unice = df["Address"].dropna().unique()
adrese_noi = [a for a in adrese_unice if a not in cache]

print(f"Adrese noi de geocodat: {len(adrese_noi)} "
      f"(restul de {len(adrese_unice) - len(adrese_noi)} sunt deja în cache).")

for i, adresa in enumerate(adrese_noi, start=1):
    try:
        location = geocode(adresa)
        if location:
            cache[adresa] = [location.latitude, location.longitude]
            print(f"[{i}/{len(adrese_noi)}] OK  -> {adresa} "
                  f"({location.latitude:.5f}, {location.longitude:.5f})")
        else:
            cache[adresa] = [None, None]
            print(f"[{i}/{len(adrese_noi)}] NU S-A GĂSIT -> {adresa}")
    except Exception as e:
        cache[adresa] = [None, None]
        print(f"[{i}/{len(adrese_noi)}] EROARE -> {adresa}: {e}")

    salveaza_cache()   # salvăm progresiv, după fiecare adresă
    time.sleep(1)      # regula obligatorie: max 1 cerere/secundă

print("\nToate adresele sunt acum în cache.")

# ---------------------------------------------------------------------
# 5. Aplicăm rezultatele din cache pe ÎNTREGUL dataset (toate cele
#    714 rânduri, nu doar cele 523 unice)
# ---------------------------------------------------------------------
df["Latitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[0])
df["Longitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[1])

negasite = df["Latitude"].isna().sum()
print(f"Rânduri fără coordonate: {negasite} din {len(df)}.")

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Fișier salvat: {OUTPUT_FILE}")