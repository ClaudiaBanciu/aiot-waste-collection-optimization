import pandas as pd
import re

fisiere = {
    "SB30SOM": ("data/raw/SB30SOM.csv", 1),
    "SB45SOM": ("data/raw/SB45SOM.csv", 2),
    "SB25SOM": ("data/raw/SB25SOM.csv", 3),
}


pattern_numar_valid = re.compile(r"^\d+(\s?[A-Za-z])?$|^\d+\s?-\s?\d+$")

def numar_valid(x):
    if pd.isna(x):
        return False
    return bool(pattern_numar_valid.match(str(x).strip()))

dataframes_curate = {}  

for nume, (cale, route_id) in fisiere.items():
    print("=" * 50)
    print(f"FIȘIER: {nume} (route_id={route_id})")
    print("=" * 50)

    df = pd.read_csv(cale, encoding="utf-8")
    total_initial = len(df)

    lipsa_strada = df["Street"].isna()
    lipsa_numar = df["Number"].isna()
    numar_gresit = (~lipsa_numar) & (~df["Number"].apply(numar_valid))

    invalid = lipsa_strada | lipsa_numar | numar_gresit

    print(f"Total rânduri: {total_initial}")
    print(f"  - eliminate din lipsă Street: {lipsa_strada.sum()}")
    print(f"  - eliminate din lipsă Number: {(lipsa_numar & ~lipsa_strada).sum()}")
    print(f"  - eliminate din format greșit Number: {numar_gresit.sum()}")
    print(f"  Total eliminate: {invalid.sum()}")

    df_curat = df[~invalid].copy()
    print(f"  Rânduri rămase: {len(df_curat)}")

    df_curat["Street"] = df_curat["Street"].str.strip()
    df_curat["Number"] = df_curat["Number"].astype(str).str.strip()
    df_curat["Datetime"] = pd.to_datetime(df_curat["Datetime"])
    df_curat["route_id"] = route_id

    dataframes_curate[nume] = df_curat
    print()

print(" Fișiere curățate păstrate în memorie: ", list(dataframes_curate.keys()))

df_unificat = pd.concat(dataframes_curate.values(), ignore_index=True)
print(f"\nDataset unificat: {len(df_unificat)} rânduri (din {sum(len(d) for d in dataframes_curate.values())} verificate)")


df_unificat["City"] = df_unificat["City"].str.strip()
df_unificat["Country"] = df_unificat["Country"].str.strip()

df_unificat["Address"] = (
    df_unificat["Street"] + " " + df_unificat["Number"] + ", " +
    df_unificat["City"] + ", " + df_unificat["Country"]
)

print("\nExemplu Address construit:")
print(df_unificat["Address"].iloc[0])

import numpy as np
np.random.seed(42)  
df_unificat["fill_level"] = np.random.randint(0, 101, size=len(df_unificat))


coloane_finale = ["route_id", "Car", "Datetime", "Id", "Capacity", "Address", "fill_level"]
df_final = df_unificat[coloane_finale]

print("\nVerificare finală:")
print(df_final.head())
print(f"\nDimensiune finală: {df_final.shape}")
print(f"Valori lipsă rămase:\n{df_final.isnull().sum()}")

df_final.to_csv("data/processed/dataset_clean.csv", index=False, encoding="utf-8")
print("\nFișier salvat: data/processed/dataset_clean.csv")