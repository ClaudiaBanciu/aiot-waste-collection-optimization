import pandas as pd
from geopy.geocoders import Nominatim

import folium
from folium.plugins import MarkerCluster

df = pd.read_csv("data\\processed\\dataset_coordinates.csv")

# --- GENERAREA HĂRȚII CU FOLIUM ---

# păstrăm doar rândurile care chiar au coordonate (geocodarea poate eșua pt unele adrese)
df_harta = df.dropna(subset=['latitudine', 'longitudine']).copy()
df_harta['latitudine'] = df_harta['latitudine'].astype(float)
df_harta['longitudine'] = df_harta['longitudine'].astype(float)

# centrăm harta pe media coordonatelor (practic, centrul "norului" de puncte)
harta = folium.Map(
    location=[df_harta['latitudine'].mean(), df_harta['longitudine'].mean()],
    zoom_start=13
)

# folosim un cluster de markere - util cand ai multe puncte apropiate,
# ca sa nu se suprapuna vizual pe hartă
cluster = MarkerCluster().add_to(harta)

for _, row in df_harta.iterrows():
    folium.Marker(
        location=[row['latitudine'], row['longitudine']],
        popup=row['Address'],
        tooltip=row['Address']
    ).add_to(cluster)

harta_path = "data\\maps\\harta_coordonate.html"
harta.save(harta_path)
print(f"Harta a fost salvată în {harta_path}")
