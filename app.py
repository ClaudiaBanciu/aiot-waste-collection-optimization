"""
Pasul 5: Interfața web interactivă (Streamlit)
-------------------------------------------------
Instalare (o dată):
    pip install streamlit streamlit-folium

Rulare:
    streamlit run app.py   (sau: python -m streamlit run app.py)
"""

import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st

st.set_page_config(page_title="Gestiune deșeuri - Sibiu", layout="wide")

INPUT_FILE = "data/processed/dataset_geocodat.csv"


def citeste_csv_robust(cale):
    """Citește CSV-ul indiferent cum a fost salvat (UTF-8, cp1252,
    separator ',' sau ';', sau chiar dacă e de fapt un .xlsx redenumit)."""
    with open(cale, "rb") as f:
        semnatura = f.read(4)
    if semnatura[:2] == b"PK":
        return pd.read_excel(cale)
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        for sep in [",", ";"]:
            try:
                df = pd.read_csv(cale, encoding=enc, sep=sep)
                if "Address" in df.columns:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise SystemExit(f"Nu am putut citi {cale} cu niciun encoding/separator cunoscut.")


def normalizeaza_fill_level(serie):
    """Fill Level poate fi text ('68%') sau, dacă fișierul a fost citit ca
    Excel, un număr zecimal (0.68) — normalizăm la un procent 0-100."""
    if serie.dtype == object:
        return pd.to_numeric(serie.astype(str).str.rstrip("%"), errors="coerce")
    numeric = pd.to_numeric(serie, errors="coerce")
    if numeric.max() <= 1:
        numeric = numeric * 100
    return numeric


@st.cache_data
def incarca_date():
    df = citeste_csv_robust(INPUT_FILE)
    for col in ["Latitude", "Longitude"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Fill_num"] = normalizeaza_fill_level(df["Fill Level"]).astype(int)
    df["ora_numerica"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    df["ora"] = df["Datetime"].dt.strftime("%H:%M")
    return df


def distanta_traseu(puncte):
    """Suma distanțelor consecutive (km), în ordinea dată a punctelor."""
    total = 0.0
    for i in range(len(puncte) - 1):
        total += geodesic(puncte[i], puncte[i + 1]).km
    return total


def optimizeaza_nearest_neighbor(puncte):
    """Euristică greedy 'cel mai apropiat vecin': pornește din primul punct,
    la fiecare pas sare la cel mai apropiat punct nevizitat. Returnează
    traseul reordonat și distanța totală."""
    if len(puncte) < 2:
        return puncte, 0.0

    ramase = list(range(1, len(puncte)))
    ordine = [0]
    curent = 0
    total = 0.0

    while ramase:
        distante = [(j, geodesic(puncte[curent], puncte[j]).km) for j in ramase]
        urmator, dist = min(distante, key=lambda t: t[1])
        total += dist
        ordine.append(urmator)
        ramase.remove(urmator)
        curent = urmator

    traseu_optimizat = [puncte[i] for i in ordine]
    return traseu_optimizat, total


df = incarca_date()

st.title("🚛 Gestiune deșeuri - Sibiu")

# ---------------------------------------------------------------------
# Sidebar: selectarea RUTEI (una singură)
# ---------------------------------------------------------------------
st.sidebar.header("Filtre")

rute_disponibile = sorted(df["Route_id"].unique())
ruta_selectata = st.sidebar.selectbox("Rută", rute_disponibile)

nivel_min, nivel_max = st.sidebar.slider(
    "Nivel de umplere (%) — interval", 0, 100, (0, 100)
)

df_ruta = df[df["Route_id"] == ruta_selectata].copy()
masini_ruta = sorted(df_ruta["Car"].unique())

df_filtrat = df_ruta[df_ruta["Fill_num"].between(nivel_min, nivel_max)].copy()

st.caption(f"Ruta {ruta_selectata} este deservită de: {', '.join(masini_ruta)}")

# ---------------------------------------------------------------------
# Statistici rapide
# ---------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Containere afișate", len(df_filtrat))
col2.metric("Nivel mediu umplere",
            f"{df_filtrat['Fill_num'].mean():.1f}%" if len(df_filtrat) else "-")
col3.metric("Mașini pe rută", len(masini_ruta))

# ---------------------------------------------------------------------
# Hartă — punctele rutei, ÎN ORDINE (traseu conectat, numerotat)
# ---------------------------------------------------------------------
st.subheader("Hartă — traseul rutei, în ordine cronologică")

culori = ["blue", "red", "green", "purple", "orange", "darkred", "cadetblue"]
culoare_masina = {m: culori[i % len(culori)] for i, m in enumerate(masini_ruta)}

if len(df_filtrat) > 0:
    harta = folium.Map(
        location=[df_filtrat["Latitude"].mean(), df_filtrat["Longitude"].mean()],
        zoom_start=13,
    )

    for masina in masini_ruta:
        grup = df_filtrat[df_filtrat["Car"] == masina].sort_values("Datetime").reset_index(drop=True)
        if len(grup) == 0:
            continue
        culoare = culoare_masina[masina]
        puncte = list(zip(grup["Latitude"], grup["Longitude"]))

        # linia care unește opririle, în ordine
        folium.PolyLine(puncte, color=culoare, weight=2, opacity=0.6).add_to(harta)

        for i, rand in grup.iterrows():
            folium.CircleMarker(
                location=[rand["Latitude"], rand["Longitude"]],
                radius=6,
                color=culoare,
                fill=True,
                fill_opacity=0.9,
                popup=(
                    f"#{i+1} — {rand['Address']}<br>"
                    f"Id: {rand['Id']}<br>Mașină: {masina}<br>"
                    f"Oră: {rand['ora']}<br>Umplere: {rand['Fill_num']}%"
                ),
            ).add_to(harta)
            # numărul de ordine, ca etichetă mică
            folium.map.Marker(
                [rand["Latitude"], rand["Longitude"]],
                icon=folium.DivIcon(html=(
                    f'<div style="font-size:9px;color:white;font-weight:bold;'
                    f'transform:translate(6px,-6px);">{i+1}</div>'
                )),
            ).add_to(harta)

    st_folium(harta, width=1100, height=550, key="harta_ruta")
else:
    st.warning("Niciun container nu corespunde filtrelor selectate.")

# ---------------------------------------------------------------------
# Tabel — Id, Capacitate, Fill Level, oră
# ---------------------------------------------------------------------
st.subheader("Containere pe rută")

with st.expander("Vezi datele tabelar", expanded=True):
    tabel = df_filtrat[["Id", "Car", "Address", "ora", "Fill_num", "Capacity"]].copy()
    tabel = tabel.rename(columns={"Fill_num": "Fill Level (%)", "ora": "Oră"})
    tabel = tabel.sort_values("Oră")
    st.dataframe(tabel, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Distanța NEOPTIMIZATĂ vs OPTIMIZATĂ, per mașină, pentru ruta selectată
# ---------------------------------------------------------------------
st.subheader("Distanța: neoptimizată vs optimizată")

total_neoptim = 0.0
total_optim = 0.0

for masina in masini_ruta:
    grup = df_ruta[df_ruta["Car"] == masina].sort_values("Datetime").reset_index(drop=True)
    if len(grup) < 2:
        continue
    puncte = list(zip(grup["Latitude"], grup["Longitude"]))

    dist_neoptim = distanta_traseu(puncte)
    _, dist_optim = optimizeaza_nearest_neighbor(puncte)

    total_neoptim += dist_neoptim
    total_optim += dist_optim

    economie = (1 - dist_optim / dist_neoptim) * 100 if dist_neoptim > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{masina} — neoptimizată", f"{dist_neoptim:.2f} km")
    c2.metric(f"{masina} — optimizată", f"{dist_optim:.2f} km")
    c3.metric(f"{masina} — economie", f"{economie:.1f}%")

st.markdown(
    f"**Total rută {ruta_selectata}:** "
    f"{total_neoptim:.2f} km neoptimizat  →  {total_optim:.2f} km optimizat "
    f"({(1 - total_optim/total_neoptim)*100:.1f}% economie)" if total_neoptim > 0 else ""
)

st.caption(
    "Optimizarea folosește euristica 'cel mai apropiat vecin' (nearest neighbor): "
    "pornind din prima oprire, se alege mereu cea mai apropiată oprire nevizitată. "
    "Nu garantează traseul optim absolut, dar e un algoritm simplu, rapid și "
    "folosit frecvent ca punct de plecare în probleme reale de rutare."
)

# ---------------------------------------------------------------------
# Predicție fill_level
# ---------------------------------------------------------------------
st.subheader("Predicție nivel de umplere (fill level)")

st.write(
    "Model simplu de regresie liniară: nivelul de umplere tinde să crească "
    "cu ora din zi (containerele se umplu pe parcursul zilei). Alege un "
    "container cu mai multe citiri pentru o predicție individuală, sau "
    "folosește tendința generală a rutei."
)

# --- Predicție generală, pe baza tendinței întregii rute ---
if len(df_ruta) >= 2:
    x = df_ruta["ora_numerica"].values
    y = df_ruta["Fill_num"].values
    panta, intercept = np.polyfit(x, y, 1)

    ora_aleasa_time = st.slider(
        "Oră pentru predicție (rută întreagă)",
        min_value=dt.time(0, 0), max_value=dt.time(23, 50),
        value=dt.time(12, 0), step=dt.timedelta(minutes=10),
    )
    ora_aleasa = ora_aleasa_time.hour + ora_aleasa_time.minute / 60
    predictie_ruta = panta * ora_aleasa + intercept
    predictie_ruta = min(max(predictie_ruta, 0), 100)

    st.metric(
        f"Nivel de umplere estimat pe ruta {ruta_selectata}, la ora {ora_aleasa_time.strftime('%H:%M')}",
        f"{predictie_ruta:.1f}%"
    )
    st.caption(f"Tendință: +{panta:.2f}% umplere / oră (pantă regresie liniară pe toată ruta).")

    chart_data = df_ruta[["ora_numerica", "Fill_num", "ora", "Address"]].copy()
    chart_data = chart_data.rename(columns={"Fill_num": "Fill Level (%)"})

    linie_regresie = pd.DataFrame({
        "ora_numerica": [x.min(), x.max()],
    })
    linie_regresie["Fill Level (%)"] = panta * linie_regresie["ora_numerica"] + intercept

    puncte_chart = alt.Chart(chart_data).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("ora_numerica", title="Ora din zi"),
        y=alt.Y("Fill Level (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("ora", title="Ora"),
            alt.Tooltip("Fill Level (%)", title="Fill Level (%)"),
            alt.Tooltip("Address", title="Adresă"),
        ],
    )
    linie_chart = alt.Chart(linie_regresie).mark_line(color="red").encode(
        x="ora_numerica", y="Fill Level (%)"
    )
    st.altair_chart((puncte_chart + linie_chart).properties(height=350), use_container_width=True)
    st.caption("Ora pe axă e afișată ca număr zecimal (ex: 7.85 = ora 07:51). "
               "Treci cu mouse-ul peste un punct ca să vezi ora exactă și adresa.")
else:
    st.info("Nu sunt suficiente date pe această rută pentru o predicție.")

# --- Predicție per container individual (dacă are >=2 citiri) ---
PRAG_MINIM_ORE = 0.5  # sub 30 de minute între citiri -> extrapolarea nu e sigură


def interval_citiri_ore(id_container):
    valori = df_ruta.loc[df_ruta["Id"] == id_container, "ora_numerica"]
    return valori.max() - valori.min()


id_counts = df_ruta["Id"].value_counts()
candidati = id_counts[id_counts >= 2].index.tolist()
id_cu_istoric = sorted(
    i for i in candidati if interval_citiri_ore(i) >= PRAG_MINIM_ORE
)

if id_cu_istoric:
    st.markdown("**Predicție pentru un container specific (cu istoric ≥2 citiri):**")
    id_ales = st.selectbox("Alege un Id de container", id_cu_istoric)

    istoric = df_ruta[df_ruta["Id"] == id_ales].sort_values("Datetime")
    st.dataframe(
        istoric[["Datetime", "ora", "Fill_num"]].rename(columns={"Fill_num": "Fill Level (%)"}),
        hide_index=True,
    )

    x_i = istoric["ora_numerica"].values
    y_i = istoric["Fill_num"].values

    if len(set(x_i)) < 2:
        st.caption("Citirile sunt la aceeași oră — nu se poate calcula o tendință.")
    else:
        panta_i, intercept_i = np.polyfit(x_i, y_i, 1)

        ora_default_h = min(int(x_i.max()) + 1, 23)
        ora_default_m = int(round((x_i.max() % 1) * 60 / 10) * 10) % 60

        ora_viitoare_time = st.slider(
            "Predicție pentru ora",
            min_value=dt.time(0, 0), max_value=dt.time(23, 50),
            value=dt.time(ora_default_h, ora_default_m), step=dt.timedelta(minutes=10),
            key="ora_container",
        )
        ora_viitoare = ora_viitoare_time.hour + ora_viitoare_time.minute / 60
        predictie_i = min(max(panta_i * ora_viitoare + intercept_i, 0), 100)
        st.metric(f"Fill level estimat pentru {id_ales} la ora {ora_viitoare_time.strftime('%H:%M')}",
                  f"{predictie_i:.1f}%")
        if panta_i > 0:
            ora_plin = (100 - intercept_i) / panta_i
            if ora_plin > x_i.max():
                h_plin = int(ora_plin) % 24
                m_plin = int(round((ora_plin % 1) * 60))
                st.caption(f"La rata actuală de umplere, containerul ar atinge 100% în jurul orei {h_plin:02d}:{m_plin:02d}.")
else:
    st.info("Niciun container din această rută nu are cel puțin 2 citiri, "
             "deci nu se poate face o predicție individuală (doar cea pe rută întreagă, de mai sus).")
