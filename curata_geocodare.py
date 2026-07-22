import pandas as pd

df = pd.read_csv("data/processed/dataset_geocoded.csv", encoding="utf-8")

aberante = df[(df["lat"] < 45.5) | (df["lat"] > 46.0)]

print(f"Total rânduri: {len(df)}")
print(f"Puncte aberante găsite: {len(aberante)}")
print("\nAdrese aberante (unice):")
for adresa in aberante["Address"].unique():
    print(f"  - {adresa}")