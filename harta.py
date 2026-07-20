"""
Pasul 3: Prima hartă (vizualizare statică)
--------------------------------------------
Citește dataset_geocodat.csv (produs la Pasul 1+2) și generează o hartă
HTML interactivă cu folium. Fiecare container e un punct, colorat diferit
în funcție de Route_id.

Rulare:
    python harta.py
Apoi deschide harta.html în browser (dublu-click pe fișier).
"""

import pandas as pd
import folium

INPUT_FILE = "data/processed/dataset_geocodat.csv"
OUTPUT_MAP = "harta.html"

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


df = citeste_csv_robust(INPUT_FILE)

# normalizăm Latitude/Longitude în caz că au fost citite ca text cu virgulă zecimală
for col in ["Latitude", "Longitude"]:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

# Eliminăm rândurile fără coordonate (adrese negăsite la geocodare)
inainte = len(df)
df = df.dropna(subset=["Latitude", "Longitude"])
print(f"Puncte cu coordonate valide: {len(df)} din {inainte}.")

# Avertizăm dacă există coordonate aberante (foarte departe de Sibiu,
# semn că geocodarea a greșit pentru acea adresă)
SIBIU_LAT, SIBIU_LON = 45.7983, 24.1256
departe = df[
    (df["Latitude"].sub(SIBIU_LAT).abs() > 0.5) |
    (df["Longitude"].sub(SIBIU_LON).abs() > 0.5)
]
if len(departe) > 0:
    print(f"ATENȚIE: {len(departe)} puncte par să fie departe de Sibiu — verifică-le:")
    print(departe[["Address", "Latitude", "Longitude"]].drop_duplicates())

# ---------------------------------------------------------------------
# Culori distincte per Route_id
# ---------------------------------------------------------------------
culori = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]
rute = sorted(df["Route_id"].unique())
culoare_ruta = {ruta: culori[i % len(culori)] for i, ruta in enumerate(rute)}

# ---------------------------------------------------------------------
# Construim harta, centrată pe media coordonatelor
# ---------------------------------------------------------------------
harta = folium.Map(
    location=[df["Latitude"].mean(), df["Longitude"].mean()],
    zoom_start=13,
)

for _, rand in df.iterrows():
    folium.CircleMarker(
        location=[rand["Latitude"], rand["Longitude"]],
        radius=5,
        color=culoare_ruta[rand["Route_id"]],
        fill=True,
        fill_opacity=0.8,
        popup=(
            f"Adresă: {rand['Address']}<br>"
            f"Mașină: {rand['Car']}<br>"
            f"Rută: {rand['Route_id']}<br>"
            f"Nivel umplere: {rand['Fill Level']}<br>"
            f"Capacitate: {rand['Capacity']}"
        ),
    ).add_to(harta)

# Legendă simplă
legenda_html = "<div style='position: fixed; bottom: 30px; left: 30px; z-index: 1000; " \
               "background: white; padding: 10px; border: 1px solid grey; border-radius: 5px;'>" \
               "<b>Rute</b><br>"
for ruta, culoare in culoare_ruta.items():
    legenda_html += f"<span style='color:{culoare};'>&#9679;</span> Ruta {ruta}<br>"
legenda_html += "</div>"
harta.get_root().html.add_child(folium.Element(legenda_html))

harta.save(OUTPUT_MAP)
print(f"Harta a fost salvată în: {OUTPUT_MAP} — deschide-o în browser.")