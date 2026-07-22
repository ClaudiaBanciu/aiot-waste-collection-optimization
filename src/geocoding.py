import pandas as pd
import googlemaps
import os
import json
from dotenv import load_dotenv

class CoordinatesCalculator:
    def __init__(self, api_key, cache_file="data/processed/cache_coordonate.json"):
        self.api_key = api_key
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.cache_file = cache_file
        
        # Încărcăm cache-ul imediat ce inițializăm clasa
        self.cache = self._load_cache()

    def _load_cache(self):
        """Citește cache-ul din fișier, dacă există."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Salvează dicționarul curent în fișierul JSON."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def get_coordinates(self, address):
        if pd.isna(address) or str(address).strip() == "":
            return None, None
            
        address_str = str(address).strip()

        # PASUL 1: Verificăm dacă adresa există deja în cache
        if address_str in self.cache:
            print(f"[CACHE] Adresă găsită: {address_str}")
            return self.cache[address_str]['lat'], self.cache[address_str]['lng']

        # PASUL 2: Dacă nu e în cache, folosim API-ul Google Maps
        print(f"[API] Caut coordonate noi pentru: {address_str}...")
        try:
            result = self.gmaps.geocode(address_str)
            if result:
                location = result[0]['geometry']['location']
                
                # Adăugăm în cache și salvăm fișierul
                self.cache[address_str] = {'lat': location['lat'], 'lng': location['lng']}
                self._save_cache()
                
                return location['lat'], location['lng']
        except Exception as e:
            print(f"Error fetching coordinates for {address_str}: {e}")
            
        return None, None

    def add_coordinates(self, df):
        # Folosim metoda lambda modificată ușor pentru a prinde și None-urile din cache
        df[['Latitude', 'Longitude']] = df['Address'].apply(
            lambda x: pd.Series(self.get_coordinates(x))
        )
        return df

    def process_file(self, input_path, output_path):
        print(f"Încărcare date din: {input_path}...")
        df = pd.read_csv(input_path)
        
        print("Începere geocodare...")
        df = self.add_coordinates(df)
        
        df.to_csv(output_path, index=False)
        print(f"Salvat: {output_path}")

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    if not API_KEY:
        print("EROARE: Nu s-a găsit cheia API. Verifică fișierul .env!")
    else:
        calculator = CoordinatesCalculator(api_key=API_KEY)
        calculator.process_file(
            input_path="data/processed/data_combined.csv",
            output_path="data/processed/data_geocoded.csv",
        )