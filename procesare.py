import pandas as pd

df = pd.read_csv("data/raw/SB25SOM.csv")

print(df.head())
print(df.info())


print("\nValori lipsa pe fiecare coloana:")
print(df.isnull().sum())

df = df.dropna(subset=["Street", "Number"])

df["Street"] = df["Street"].str.strip()
df["Number"] = df["Number"].astype(str).str.strip()

df["Datetime"] = pd.to_datetime(df["Datetime"])


print("\nDimensiunea după preprocesare:")
print(df.shape)

df.to_csv("data/raw/SB25SOM_preprocesat.csv", index=False)

