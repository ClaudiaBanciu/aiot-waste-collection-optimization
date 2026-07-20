"""
Reparare geocodare — STRICT doar Sibiu
-----------------------------------------
Reia adresele problematice (negăsite sau prea departe de Sibiu) și
încearcă să le corecteze DOAR în interiorul orașului Sibiu:
  - Nivel 1: adresa exactă (stradă + număr), query structurat city=Sibiu
  - Nivel 2: dacă numărul exact nu există în OSM, doar strada (fără număr)

Orice adresă care tot nu se găsește în Sibiu e EXCLUSĂ din setul final
(nu se acceptă coordonate din alt oraș, gen Mediaș).

La final:
  - regenerează dataset_geocodat.csv (cu toate rândurile, unele NaN)
  - salvează dataset_geocodat_sibiu.csv (DOAR rândurile cu coordonate
    confirmate în Sibiu — acesta e fișierul de folosit mai departe)

Rulare:
    python reparare_geocodare.py
"""

import json
import os
import re
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

INPUT_FILE = "data/processed/dataset_clean.csv"
OUTPUT_FILE = "data/processed/dataset_geocodat.csv"
OUTPUT_FILE_SIBIU = "data/processed/dataset_geocodat_sibiu.csv"
CACHE_FILE = "data/processed/geocoding_cache.json"

SIBIU_LAT, SIBIU_LON = 45.7983, 24.1256
PRAG_GRADE = 0.3

# ---------------------------------------------------------------------
# Corecții cunoscute de typo-uri / prefixe problematice din adrese
# ---------------------------------------------------------------------
CORECTII = [
    ("Bălacescu", "Bălcescu"),      # typo: numele corect e Nicolae Bălcescu
    ("Strada Rampa ", "Strada "),   # "Rampa" nu face parte din numele străzii
    ("G-ral ", ""),                 # abrevierea încurcă Nominatim
    ("Strada Piața", "Piața"),      # "Piața" nu e "Strada"
]


def strada_fara_numar(strada):
    """Elimină numărul de casă de la finalul șirului (ultimul token numeric),
    păstrând restul numelui străzii intact (ex. 'Piața 1 Decembrie 1918')."""
    return re.sub(r"\s+\d+\s*[A-Za-z]?$", "", strada).strip()


def fix_mojibake(s):
    """Repară diacritice corupte (UTF-8 citit greșit ca Windows-1252)."""
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def genereaza_variante(adresa):
    """Generează variante corectate ale unei adrese, de încercat pe rând."""
    variante = [adresa]

    reparata = fix_mojibake(adresa)
    if reparata != adresa:
        variante.append(reparata)

    for gresit, corect in CORECTII:
        for v in list(variante):
            if gresit in v:
                variante.append(v.replace(gresit, corect))

    vazute = set()
    unice = []
    for v in variante:
        if v not in vazute:
            unice.append(v)
            vazute.add(v)
    return unice


# ---------------------------------------------------------------------
# Încărcăm datele și cache-ul existent
# ---------------------------------------------------------------------
try:
    df = pd.read_excel(INPUT_FILE)
except Exception:
    df = pd.read_csv(INPUT_FILE)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache găsit: {len(cache)} adrese.")
elif os.path.exists(OUTPUT_FILE):
    print(f"Nu găsesc {CACHE_FILE} — îl construiesc din {OUTPUT_FILE} existent.")
    df_existent = pd.read_csv(OUTPUT_FILE)
    cache = {}
    for _, r in df_existent.dropna(subset=["Address"]).iterrows():
        lat = r["Latitude"] if pd.notna(r["Latitude"]) else None
        lon = r["Longitude"] if pd.notna(r["Longitude"]) else None
        cache[r["Address"]] = [lat, lon]
    print(f"Cache construit: {len(cache)} adrese.")
else:
    raise SystemExit(
        f"Nu găsesc nici {CACHE_FILE}, nici {OUTPUT_FILE}. "
        "Rulează întâi geocodare_adrese.py."
    )


def salveaza_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Identificăm adresele problematice: negăsite SAU prea departe de Sibiu
# (coordonatele din alt oraș, gen Mediaș, sunt tratate ca "negăsite")
# ---------------------------------------------------------------------
def e_valida_sibiu(coord):
    if coord is None or coord[0] is None:
        return False
    lat, lon = coord
    return abs(lat - SIBIU_LAT) <= PRAG_GRADE and abs(lon - SIBIU_LON) <= PRAG_GRADE


adrese_unice = df["Address"].dropna().unique()
problematice = [a for a in adrese_unice if not e_valida_sibiu(cache.get(a))]
print(f"Adrese problematice de reîncercat: {len(problematice)}")

# ---------------------------------------------------------------------
# Geocoder — query STRUCTURAT, city="Sibiu" impus explicit
# ---------------------------------------------------------------------
geolocator = Nominatim(user_agent="proiect_gestiune_deseuri_sibiu")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def geocodeaza_in_sibiu(strada_si_numar):
    return geocode(
        {"street": strada_si_numar, "city": "Sibiu", "country": "Romania"}
    )


cache_aproximative = {}  # rezolvate doar la nivel de stradă (fără număr exact)
neexcluse = []           # adrese care rămân excluse din setul final

for i, adresa_originala in enumerate(problematice, start=1):
    gasit = False
    variante = genereaza_variante(adresa_originala)

    # --- Nivel 1: adresa exactă (stradă + număr), strict în Sibiu ---
    for varianta in variante:
        strada = varianta.split(", Sibiu")[0].strip()
        try:
            location = geocodeaza_in_sibiu(strada)
        except Exception as e:
            location = None
            print(f"  eroare la '{strada}': {e}")
        time.sleep(1)

        if location and e_valida_sibiu([location.latitude, location.longitude]):
            cache[adresa_originala] = [location.latitude, location.longitude]
            salveaza_cache()
            print(f"[{i}/{len(problematice)}] REPARAT -> {adresa_originala} "
                  f"(încercare: '{strada}')  "
                  f"({location.latitude:.5f}, {location.longitude:.5f})")
            gasit = True
            break
    if gasit:
        continue

    # --- Nivel 2: numărul exact nu există în OSM — doar strada, tot
    #     strict în Sibiu (nu acceptăm alt oraș) ---
    for varianta in variante:
        strada_completa = varianta.split(", Sibiu")[0].strip()
        strada_scurta = strada_fara_numar(strada_completa)
        if strada_scurta == strada_completa:
            continue

        try:
            location = geocodeaza_in_sibiu(strada_scurta)
        except Exception as e:
            location = None
            print(f"  eroare la '{strada_scurta}': {e}")
        time.sleep(1)

        if location and e_valida_sibiu([location.latitude, location.longitude]):
            cache[adresa_originala] = [location.latitude, location.longitude]
            cache_aproximative[adresa_originala] = strada_scurta
            salveaza_cache()
            print(f"[{i}/{len(problematice)}] APROXIMAT (doar stradă) -> "
                  f"{adresa_originala} (încercare: '{strada_scurta}')  "
                  f"({location.latitude:.5f}, {location.longitude:.5f})")
            gasit = True
            break
    if gasit:
        continue

    # Nu s-a găsit strict în Sibiu -> excludem, NU acceptăm alt oraș
    cache[adresa_originala] = [None, None]
    salveaza_cache()
    neexcluse.append(adresa_originala)
    print(f"[{i}/{len(problematice)}] EXCLUS (nu există strict în Sibiu) -> "
          f"{adresa_originala}")

if cache_aproximative:
    print(f"\n{len(cache_aproximative)} adrese plasate APROXIMATIV "
          f"(doar la nivel de stradă, numărul exact nu există în OSM):")
    for a in cache_aproximative:
        print("  -", a)

if neexcluse:
    print(f"\n{len(neexcluse)} adrese EXCLUSE din setul final "
          f"(nu există deloc în Sibiu, în datele OSM):")
    for a in neexcluse:
        print("  -", a)

# ---------------------------------------------------------------------
# Regenerăm dataset_geocodat.csv (complet, inclusiv rânduri excluse = NaN)
# ---------------------------------------------------------------------
df["Latitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[0])
df["Longitude"] = df["Address"].map(lambda a: cache.get(a, [None, None])[1])
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------
# Salvăm și versiunea STRICT Sibiu — doar rânduri cu coordonate valide
# ---------------------------------------------------------------------
df_sibiu = df.dropna(subset=["Latitude", "Longitude"]).copy()
df_sibiu.to_csv(OUTPUT_FILE_SIBIU, index=False, encoding="utf-8-sig")

print(f"\nTotal rânduri: {len(df)}")
print(f"Rânduri cu coordonate valide în Sibiu: {len(df_sibiu)}")
print(f"Rânduri excluse (fără coordonate în Sibiu): {len(df) - len(df_sibiu)}")
print(f"\nFișiere salvate:")
print(f"  {OUTPUT_FILE}  (toate rândurile, unele cu Latitude/Longitude goale)")
print(f"  {OUTPUT_FILE_SIBIU}  (DOAR rândurile strict din Sibiu — folosește pe acesta mai departe)")