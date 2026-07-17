import time
import pandas as pd
from geopy.geocoders import Nominatim

df = pd.read_csv("data\\processed\\dataset_clean.csv")
df['latitudine'] = None
df['longitudine'] = None

geolocator = Nominatim(user_agent="my_sibiu_geocoder_app")

for index, row in df.iterrows():
    #print(f"Geocoding: {df.Address}")
    
    try:
        print(f"Geocoding: {row['Address']}")
        location = geolocator.geocode(row['Address'])
        if location:
            print(f"Found! Coordinates: {location.latitude}, {location.longitude}")
            df.at[index, 'latitudine'] = location.latitude
            df.at[index, 'longitudine'] = location.longitude
        else:
            print("Could not find address.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
    
    # --- THIS IS THE MAGIC LINE ---
    
    time.sleep(1.5)

#print(df.head())

csv_path = "data\\processed\\dataset_coordinates.csv"
df.to_csv(csv_path, index=False)
print(f"Coordonatele au fost salvate în {csv_path}")
