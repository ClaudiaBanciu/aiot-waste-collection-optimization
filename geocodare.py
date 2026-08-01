import pandas as pd
import time
import os
from geopy.geocoders import Nominatim

df = pd.read_csv("data/processed/dataset_clean.csv", encoding="utf-8")
print(f"Total rânduri: {len(df)}")

adrese_unice = df["Address"].unique()
print(f"Adrese unice de geocodat: {len(adrese_unice)}")

cale_cache = "data/processed/geocache.csv"

if os.path.exists(cale_cache):
    cache = pd.read_csv(cale_cache, encoding="utf-8")
    print(f"Cache existent găsit: {len(cache)} adrese deja geocodate")
else:
    cache = pd.DataFrame(columns=["Address", "lat", "lon"])
    print("Nu există cache încă, pornim de la zero")

adrese_deja_stiute = set(cache["Address"])

# Geocodăm doar adresele care NU sunt deja în cache

geolocator = Nominatim(user_agent="aiot_waste_collection_project")

rezultate_noi = []

for adresa in adrese_unice:
    if adresa in adrese_deja_stiute:
        continue  # deja o avem, nu mai sunăm serviciul

    try:
        loc = geolocator.geocode(adresa, timeout=10)
        if loc:
            rezultate_noi.append({"Address": adresa, "lat": loc.latitude, "lon": loc.longitude})
            print(f"OK: {adresa} -> ({loc.latitude}, {loc.longitude})")
        else:
            rezultate_noi.append({"Address": adresa, "lat": None, "lon": None})
            print(f"NEGĂSIT: {adresa}")
    except Exception as e:
        rezultate_noi.append({"Address": adresa, "lat": None, "lon": None})
        print(f"EROARE la {adresa}: {e}")

    time.sleep(1)  # OBLIGATORIU - Nominatim permite max 1 cerere/secundă

if rezultate_noi:
    cache = pd.concat([cache, pd.DataFrame(rezultate_noi)], ignore_index=True)
    cache.to_csv(cale_cache, index=False, encoding="utf-8")
    print(f"\nCache actualizat, salvat în {cale_cache}")
else:
    print("\nNimic nou de geocodat, cache-ul era deja complet")

# Aplicăm cache-ul pe tot dataset-ul și salvăm rezultatul final

df_geocodat = df.merge(cache, on="Address", how="left")

print(f"\nRânduri fără coordonate (geocodare eșuată): {df_geocodat['lat'].isna().sum()}")

df_geocodat.to_csv("data/processed/dataset_geocoded.csv", index=False, encoding="utf-8")
print("Fișier salvat: data/processed/dataset_geocoded.csv")