import pandas as pd
import googlemaps
import os
from dotenv import load_dotenv


class CoordinatesCalculator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.gmaps = googlemaps.Client(key=self.api_key)

    def get_coordinates(self, address):
        try:
            result = self.gmaps.geocode(address)
            if result:
                location = result[0]['geometry']['location']
                return location['lat'], location['lng']
        except Exception as e:
            print(f"Error fetching coordinates for {address}: {e}")
        return None, None

    def add_coordinates(self, df):
        df[['Latitude', 'Longitude']] = df['Address'].apply(
            lambda x: pd.Series(self.get_coordinates(x)) if pd.notna(x) else pd.Series([None, None])
        )
        return df

    def process_file(self, input_path, output_path):  # output_path is now explicit
        df = pd.read_csv(input_path)
        df = self.add_coordinates(df)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    calculator = CoordinatesCalculator(api_key=API_KEY)
    calculator.process_file(
        input_path="data/processed/data_combined.csv",
        output_path="data/processed/data_geocoded.csv",
    )