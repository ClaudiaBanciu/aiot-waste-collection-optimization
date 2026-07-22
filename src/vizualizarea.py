import pandas as pd
import folium
import os

def genereaza_harta_statica(cale_intrare, cale_iesire):
    print(f"Încărcare date din: {cale_intrare}...")
    df = pd.read_csv(cale_intrare)
    
    # 1. Curățarea datelor (eliminăm rândurile care nu au coordonate valide)
    df_clean = df.dropna(subset=['Latitude', 'Longitude'])
    
    if df_clean.empty:
        print("Eroare: Nu există date cu coordonate valide pentru a genera harta.")
        return

    # 2. Calculăm centrul hărții (media tuturor coordonatelor pentru a centra ecranul)
    centru_lat = df_clean['Latitude'].mean()
    centru_lon = df_clean['Longitude'].mean()
    
    # Inițializăm harta
    harta = folium.Map(location=[centru_lat, centru_lon], zoom_start=13)

    # 3. Definim o listă de culori suportate de Folium
    culori_disponibile = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'pink', 'lightblue', 'lightgreen', 'black'
    ]

    # Asociazăm fiecărui 'route_id' unic câte o culoare
    # (Presupunem că ai o coloană 'route_id' în CSV. Dacă se numește altfel, modifică aici)
    rute_unice = df_clean['route_id'].dropna().unique()
    culori_rute = {}
    for i, ruta in enumerate(rute_unice):
        culori_rute[ruta] = culori_disponibile[i % len(culori_disponibile)]

    # 4. Adăugăm fiecare container pe hartă
    for index, rand in df_clean.iterrows():
        # Extragem ruta; dacă lipsește, îi dăm culoarea gri
        ruta_curenta = rand.get('route_id')
        culoare = culori_rute.get(ruta_curenta, 'gray')
        
        # Informațiile care apar la click pe punct
        popup_text = f"Rută: {ruta_curenta}<br>Adresă: {rand.get('Address', 'N/A')}"
        
        folium.CircleMarker(
            location=[rand['Latitude'], rand['Longitude']],
            radius=6, # Dimensiunea cercului
            popup=popup_text,
            color=culoare,
            fill=True,
            fill_color=culoare,
            fill_opacity=0.8
        ).add_to(harta)

    # 5. Salvăm rezultatul
    harta.save(cale_iesire)
    print(f"✅ Harta a fost generată! Deschide fișierul: {cale_iesire}")

if __name__ == "__main__":
    # Căile aferente structurii proiectului tău
    CSV_INTRARE = "data/processed/data_geocoded.csv"
    HTML_IESIRE = "data/processed/harta_statica.html"
    
    genereaza_harta_statica(CSV_INTRARE, HTML_IESIRE)