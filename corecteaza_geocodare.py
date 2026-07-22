import pandas as pd
import time
from geopy.geocoders import Nominatim

SIBIU_BBOX = [(45.70, 24.05), (45.85, 24.30)]

df = pd.read_csv("data/processed/dataset_geocoded.csv", encoding="utf-8")
cache = pd.read_csv("data/processed/geocache.csv", encoding="utf-8")

aberante = df[(df["lat"] < 45.5) | (df["lat"] > 46.0)]["Address"].unique()
print(f"Adrese de re-geocodat: {len(aberante)}")

geolocator = Nominatim(user_agent="aiot_waste_collection_project_v2")

def geocodeaza_cu_reincercare(adresa, viewbox=None, bounded=False, incercari_max=3):
    """Încearcă geocodarea, cu pauză crescândă dacă primim 429."""
    for incercare in range(incercari_max):
        try:
            return geolocator.geocode(adresa, viewbox=viewbox, bounded=bounded, timeout=10)
        except Exception as e:
            if "429" in str(e):
                pauza = 10 * (incercare + 1)  # 10s, 20s, 30s...
                print(f"  Blocat (429), aștept {pauza}s...")
                time.sleep(pauza)
            else:
                print(f"  Eroare: {e}")
                return None
    return None

for adresa in aberante:
    loc = geocodeaza_cu_reincercare(adresa, viewbox=SIBIU_BBOX, bounded=True)
    time.sleep(2)  # pauză mai mare între cereri, ca să nu mai fim blocați

    if not loc:
        strada_fara_numar = adresa.split(",")[0]
        strada_fara_numar = " ".join(strada_fara_numar.split(" ")[:-1])
        adresa_scurta = f"{strada_fara_numar}, Sibiu, Romania"
        loc = geocodeaza_cu_reincercare(adresa_scurta, viewbox=SIBIU_BBOX, bounded=True)
        if loc:
            print(f"OK (aproximat pe stradă): {adresa} -> ({loc.latitude}, {loc.longitude})")
        time.sleep(2)

    if loc:
        cache.loc[cache["Address"] == adresa, "lat"] = loc.latitude
        cache.loc[cache["Address"] == adresa, "lon"] = loc.longitude
        print(f"OK: {adresa} -> ({loc.latitude}, {loc.longitude})")
    else:
        print(f"NEGĂSIT complet: {adresa}")

cache.to_csv("data/processed/geocache.csv", index=False, encoding="utf-8")
print("\nCache actualizat, salvat în data/processed/geocache.csv")

df_original = pd.read_csv("data/processed/dataset_clean.csv", encoding="utf-8")
df_corectat = df_original.merge(cache, on="Address", how="left")
df_corectat.to_csv("data/processed/dataset_geocoded.csv", index=False, encoding="utf-8")
print("Dataset actualizat, salvat în data/processed/dataset_geocoded.csv")

aberante_ramase = df_corectat[(df_corectat["lat"] < 45.5) | (df_corectat["lat"] > 46.0)]
print(f"\nPuncte aberante rămase după corectare: {len(aberante_ramase)}")