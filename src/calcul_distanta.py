import pandas as pd
from geopy.distance import geodesic

def calculeaza_distante_neoptimizate(cale_intrare):
    print(f"Încărcare date din: {cale_intrare}...")
    df = pd.read_csv(cale_intrare)
    
    # Curățăm rândurile fără coordonate pentru a nu da eroare la calcul
    df_clean = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    # --- AICI VERIFICI NUMELE COLOANEI ---
    # Trebuie să pui numele coloanei care identifică mașina/ruta 
    # (ex: 'route_id', 'Masina', 'Vehicle')
    nume_coloana_masina = 'route_id' 
    
    if nume_coloana_masina not in df_clean.columns:
        print(f"Eroare: Coloana '{nume_coloana_masina}' nu există în fișier.")
        print(f"Coloanele disponibile sunt: {list(df_clean.columns)}")
        return

    distante_totale = {}

    # Grupăm datele în funcție de mașină (ar trebui să rezulte 3 grupuri)
    for masina, grup in df_clean.groupby(nume_coloana_masina):
        distanta_masina_km = 0.0
        
        # Extragem coordonatele sub formă de listă, păstrând ordinea existentă (neoptimizată)
        coordonate = list(zip(grup['Latitude'], grup['Longitude']))
        
        # Calculăm distanța de la punctul A la punctul B, apoi B la C, etc.
        for i in range(len(coordonate) - 1):
            punct_curent = coordonate[i]
            punct_urmator = coordonate[i+1]
            
            # Folosim formula geodesic pentru acuratețe maximă pe hartă
            distanta_segment = geodesic(punct_curent, punct_urmator).kilometers
            distanta_masina_km += distanta_segment
            
        distante_totale[masina] = distanta_masina_km

    # Afișăm rezultatele
    print("\n" + "="*40)
    print(" REZULTATE DISTANȚE NEOPTIMIZATE")
    print("="*40)
    
    distanta_globala = 0
    for masina, distanta in distante_totale.items():
        print(f"🚛 Mașina/Ruta [{masina}]: {distanta:.2f} km")
        distanta_globala += distanta
        
    print("-" * 40)
    print(f"🌍 Distanța totală (toate cele 3 mașini): {distanta_globala:.2f} km")
    print("="*40)

if __name__ == "__main__":
    CALE_CSV = "data/processed/data_geocoded.csv"
    calculeaza_distante_neoptimizate(CALE_CSV)