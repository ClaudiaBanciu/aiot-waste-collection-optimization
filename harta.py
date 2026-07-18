import pandas as pd
import folium

df = pd.read_csv("data/processed/dataset_geocoded.csv", encoding="utf-8")

# eliminăm rândurile fără coordonate valide
df_valid = df.dropna(subset=["lat", "lon"])
print(f"Puncte de afișat pe hartă: {len(df_valid)} din {len(df)}")

# centrăm harta pe Sibiu
harta = folium.Map(location=[45.7983, 24.1256], zoom_start=13)

culori = {1: "red", 2: "blue", 3: "green"}

for _, rand in df_valid.iterrows():
    folium.CircleMarker(
        location=[rand["lat"], rand["lon"]],
        radius=4,
        color=culori.get(rand["route_id"], "gray"),
        fill=True,
        fill_opacity=0.7,
        popup=f"{rand['Address']} (ruta {rand['route_id']})"
    ).add_to(harta)

harta.save("harta_containere.html")
print("Hartă salvată: harta_containere.html")