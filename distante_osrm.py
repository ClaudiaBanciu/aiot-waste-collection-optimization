import pandas as pd
import requests

# ============================================================
# Încărcăm datasetul geocodat (Ziua 3)
# ============================================================
df = pd.read_csv("data/processed/dataset_geocoded.csv", encoding="utf-8")
df = df.dropna(subset=["lat", "lon"])

# ============================================================
# Calculăm distanța pentru fiecare rută separat
# ============================================================
def distanta_ruta_osrm(df_ruta):
    """
    Calculează distanța totală (pe șosea, în ordinea din date) pentru o rută,
    folosind serviciul Route al OSRM local.
    """
    coords_str = ";".join(f"{row['lon']},{row['lat']}" for _, row in df_ruta.iterrows())

    url = f"http://localhost:5000/route/v1/driving/{coords_str}?overview=false"
    r = requests.get(url)

    if r.status_code != 200:
        print(f"Eroare HTTP {r.status_code} pentru {len(df_ruta)} puncte")
        return None

    data = r.json()

    if data.get("code") != "Ok":
        print(f"Eroare OSRM: {data.get('code')} - {data.get('message', '')}")
        return None

    distanta_metri = data["routes"][0]["distance"]
    return distanta_metri / 1000  # km


rezultate = []

for route_id in sorted(df["route_id"].unique()):
    df_ruta = df[df["route_id"] == route_id].sort_values("Datetime").reset_index(drop=True)
    distanta_km = distanta_ruta_osrm(df_ruta)
    rezultate.append({"route_id": route_id, "distanta_osrm_km": distanta_km, "nr_containere": len(df_ruta)})
    print(f"Ruta {route_id}: {distanta_km:.2f} km (pe {len(df_ruta)} containere)")

df_rezultate = pd.DataFrame(rezultate)
df_rezultate.to_csv("data/processed/distante_osrm.csv", index=False, encoding="utf-8")
print("\nSalvat: data/processed/distante_osrm.csv")