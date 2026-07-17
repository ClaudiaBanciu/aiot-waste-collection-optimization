import pandas as pd
import numpy as np
import os

nume_fisier = ["data\\raw\\SB25SOM.csv","data\\raw\\SB30SOM.csv","data\\raw\\SB45SOM.csv"]
traseu = ["1","2","3"]

# Salvarea DataFrame-ului curățat într-un fișier CSV
csv_path = "data\\processed\\dataset_clean.csv"

# Colectăm toate DataFrame-urile curățate pentru a le combina la final
dataframes = []

os.makedirs(os.path.dirname(csv_path), exist_ok=True)

for i in range(0, 3):

    #nmf = "data\\raw\\SB25SOM.csv"
    df = pd.read_csv(nume_fisier[i])

    #print(df.head())
    print(df.describe())

    # Aflăm numărul de rânduri
    randuri = len(df)
    print(f"Setul de date are {randuri} rânduri.")

    # conversia la datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    #__________________________________________________

    # lucrul cu datetime 
    # Data cea mai veche (valoarea minimă)
    data_veche = df['Datetime'].min()

    # Data cea mai nouă (valoarea maximă)
    data_noua = df['Datetime'].max()

    print(f"Intervalul este de la: {data_veche} până la: {data_noua}")


    #__________________________________________________

    # Crează un DataFrame nou, curățat
    df_curat = df.dropna()
    randuri1 = len(df_curat)
    print(f"Setul de date curățat are {randuri1} rânduri. La naiba!!!")

    #__________________________________________________

    # Concatenarea

    # 1. Ne asigurăm că extragem corect numele ultimelor 4 coloane
    ultimele_coloane = df_curat.columns[-4:]
    col1, col2, col3, col4 = ultimele_coloane

    # 2. Le convertim în text (string) și le unim conform regulii tale:
    # - Spațiu simplu între prima și a doua coloană
    # - Virgulă și spațiu între a doua și a treia coloană
    # - Virgulă și spațiu între a treia și a patra coloană
    df_curat['Address'] = (
        df_curat[col1].astype(str) + " " + 
        df_curat[col2].astype(str) + ", " + 
        df_curat[col3].astype(str) + ", " + 
        df_curat[col4].astype(str)
    )

    # 3. Ștergem cele 4 coloane vechi dacă nu mai ai nevoie de ele
    df_curat.drop(columns=ultimele_coloane, inplace=True)

    #__________________________________________________

    # Adaugă o coloană nouă la final
    df_curat['route_id'] = traseu[i]

    # 1. Generăm numere întregi aleatorii între 0 și 100 pentru fiecare rând
    numere_aleatorii = np.random.randint(0, 101, size=len(df_curat))

    # 2. Le convertim în format text și adăugăm simbolul "%" la final
    df_curat['fill_level'] = [f"{val}%" for val in numere_aleatorii]

    #__________________________________________________

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_curat)

    dataframes.append(df_curat)

    #__________________________________________________

# Combinăm toate DataFrame-urile într-unul singur
merged_df = pd.concat(dataframes, ignore_index=True)

# Salvăm rezultatul final în fișierul CSV
merged_df.to_csv(csv_path, index=False)
print(f"Datele au fost salvate în {csv_path}")
