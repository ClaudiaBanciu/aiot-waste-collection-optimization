"""
Interfața web interactivă (Streamlit)
-------------------------------------------------
Instalare (o dată):
    pip install streamlit streamlit-folium scikit-learn altair

Rulare:
    streamlit run app.py   (sau: python -m streamlit run app.py)
"""

import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
import fill_level_model as flm
from streamlit_folium import st_folium
import streamlit as st

st.set_page_config(page_title="Gestiune deșeuri - Sibiu", layout="wide")

INPUT_FILE = "data/processed/data_geocoded.csv"


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
    """fill_level poate fi text ('68%'), un număr zecimal (0.68), sau deja
    un procent numeric (0-100, cazul fișierului curent) — normalizăm la
    un procent 0-100 indiferent de format."""
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
    df["fill_level"] = normalizeaza_fill_level(df["fill_level"])
    return df


@st.cache_resource
def incarca_model_predictiv(_df):
    """Rulează pipeline-ul din fill_level_model.py (identic cu notebook-ul
    fill_level_prediction_draft.ipynb) o singură dată, ținut în cache —
    conține și modelul antrenat, plus istoricul simulat."""
    return flm.run_full_pipeline(_df)


def distanta_traseu_km(puncte):
    """Suma distanțelor consecutive (km), în ordinea dată a punctelor."""
    total = 0.0
    for i in range(len(puncte) - 1):
        total += geodesic(puncte[i], puncte[i + 1]).km
    return total


def optimizeaza_nearest_neighbor_km(puncte):
    """Euristică greedy 'cel mai apropiat vecin': pornește din primul punct,
    la fiecare pas sare la cel mai apropiat punct nevizitat."""
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


df_brut = incarca_date()
rezultat_model = incarca_model_predictiv(df_brut)

# df, pentru harta/tabel/distanțe, e restrâns la containere reale (fără
# rândurile de depozit plecare/sosire, care nu au coordonate utile de afișat
# ca "container")
df = rezultat_model["containers"].copy()
df = df.dropna(subset=["Latitude", "Longitude"])
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Fill_num"] = df["fill_level"].round().astype(int)
df["ora_numerica"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
df["ora"] = df["Datetime"].dt.strftime("%H:%M")

st.title("🚛 Gestiune deșeuri - Sibiu")

# ---------------------------------------------------------------------
# Sidebar: selectarea RUTEI (una singură)
# ---------------------------------------------------------------------
st.sidebar.header("Filtre")

rute_disponibile = sorted(df["route_id"].unique())
ruta_selectata = st.sidebar.selectbox("Rută", rute_disponibile)

nivel_min, nivel_max = st.sidebar.slider(
    "Nivel de umplere (%) — interval", 0, 100, (0, 100)
)

df_ruta = df[df["route_id"] == ruta_selectata].copy()
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
# Distanța NEOPTIMIZATĂ vs OPTIMIZATĂ
# ---------------------------------------------------------------------
st.subheader("Distanța: neoptimizată vs optimizată")

total_neoptim = 0.0
total_optim = 0.0

for masina in masini_ruta:
    grup = df_ruta[df_ruta["Car"] == masina].sort_values("Datetime").reset_index(drop=True)
    if len(grup) < 2:
        continue
    puncte = list(zip(grup["Latitude"], grup["Longitude"]))

    dist_neoptim = distanta_traseu_km(puncte)
    _, dist_optim = optimizeaza_nearest_neighbor_km(puncte)

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
    "Distanțele sunt calculate în linie dreaptă (geodezic). Optimizarea "
    "folosește euristica 'cel mai apropiat vecin' (nearest neighbor): "
    "pornind din prima oprire, se alege mereu cea mai apropiată oprire "
    "nevizitată. Nu garantează traseul optim absolut, dar e un algoritm "
    "simplu, rapid și folosit frecvent ca punct de plecare în probleme "
    "reale de rutare."
)

# ---------------------------------------------------------------------
# Predicție fill_level — regresie + progresia rutelor
# (bazat pe fill_level_prediction_draft.ipynb: istoric simulat 30 zile +
# regresie liniară, Etapele 2-7)
# ---------------------------------------------------------------------
st.subheader("Predicție nivel de umplere — progresia rutei")

st.write(
    f"Model de regresie liniară antrenat pe un istoric simulat de "
    f"{flm.N_DAYS} de zile per container (identic cu notebook-ul de "
    f"predicție). Eroarea medie a modelului (MAE, pe setul de testare): "
    f"**{rezultat_model['mae']:.2f} puncte procentuale**."
)

history_ruta = rezultat_model["history"][rezultat_model["history"]["route_id"] == ruta_selectata]
current_state_ruta = rezultat_model["current_state"][
    rezultat_model["current_state"]["route_id"] == ruta_selectata
].dropna(subset=["Latitude", "Longitude"])
rule_ruta = rezultat_model["rule_collection"][
    rezultat_model["rule_collection"]["route_id"] == ruta_selectata
]
predictive_ruta = rezultat_model["predictive_collection"][
    rezultat_model["predictive_collection"]["route_id"] == ruta_selectata
]

c1, c2, c3 = st.columns(3)
c1.metric("Containere pe rută", df_ruta["Id"].nunique())
c2.metric("Selectate — regulă simplă", len(rule_ruta))
c3.metric("Selectate — model predictiv", len(predictive_ruta))


# =====================================================================
# Distribuția containerelor după nivelul de umplere (grafic de bare)
# =====================================================================
st.markdown("**Distribuția containerelor după nivelul de umplere (azi / predicție):**")

df_bar = current_state_ruta.copy()

if df_bar["predicted_fill_level"].max() <= 1.0:
    df_bar["predicted_fill_level"] = df_bar["predicted_fill_level"] * 100

df_bar["status"] = np.where(
    df_bar["predicted_fill_level"] >= df_bar["threshold"],
    "va necesita colectare (predicție)",
    "sub prag (predicție)"
)

bins = [-0.1, 20.0, 40.0, 60.0, 80.0, 100.1]
labels = ["0–20%", "21–40%", "41–60%", "61–80%", "81–100%"]

df_bar["Interval Umplere"] = pd.cut(
    df_bar["predicted_fill_level"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

chart_progresie = (
    alt.Chart(df_bar.dropna(subset=["Interval Umplere"]))
    .mark_bar()
    .encode(
        x=alt.X("Interval Umplere:O", title="Interval Nivel de Umplere (%)", sort=labels),
        y=alt.Y("count():Q", title="Număr Containere"),
        color=alt.Color(
            "status:N",
            title="Status predicție",
            scale=alt.Scale(
                domain=["va necesita colectare (predicție)", "sub prag (predicție)"],
                range=["#d9534f", "#2077b4"]
            )
        ),
        tooltip=[
            alt.Tooltip("Interval Umplere:O", title="Interval"),
            alt.Tooltip("count():Q", title="Număr containere"),
            alt.Tooltip("status:N", title="Status")
        ]
    )
)

st.altair_chart(chart_progresie.properties(height=350), use_container_width=True)
st.caption("Graficul arată distribuția containerelor de pe rută în funcție de "
           "procentul de umplere estimat pentru ziua de mâine.")

# =====================================================================
# Hartă — containerele de pe rută, colorate după statusul predicției
# =====================================================================
st.markdown("**Hartă — containere colorate după predicție (mâine):**")

if len(df_bar) > 0:
    harta_predictie = folium.Map(
        location=[df_bar["Latitude"].mean(), df_bar["Longitude"].mean()],
        zoom_start=13,
    )
    harta_predictie.fit_bounds([
        [df_bar["Latitude"].min(), df_bar["Longitude"].min()],
        [df_bar["Latitude"].max(), df_bar["Longitude"].max()],
    ])

    culoare_status = {
        "va necesita colectare (predicție)": "red",
        "sub prag (predicție)": "blue",
    }

    # desenăm întâi albastru, apoi roșu LA URMĂ — ca punctele roșii să nu
    # rămână ascunse sub cele albastre, dacă sunt foarte aproape unele de altele
    for status_ordine in ["sub prag (predicție)", "va necesita colectare (predicție)"]:
        subset_status = df_bar[df_bar["status"] == status_ordine]
        for _, rand in subset_status.iterrows():
            folium.CircleMarker(
                location=[rand["Latitude"], rand["Longitude"]],
                radius=7 if status_ordine.startswith("va necesita") else 6,
                color=culoare_status[status_ordine],
                fill=True,
                fill_opacity=0.9,
                weight=2,
                popup=(
                    f"{rand['Address']}<br>"
                    f"Id: {rand['Id']}<br>"
                    f"Nivel azi: {rand['simulated_fill_level']:.1f}%<br>"
                    f"Predicție mâine: {rand['predicted_fill_level']:.1f}%<br>"
                    f"Prag: {rand['threshold']:.0f}%"
                ),
            ).add_to(harta_predictie)

    legenda_predictie = (
        "<div style='position: fixed; bottom: 30px; left: 30px; z-index: 1000; "
        "background: white; padding: 10px; border: 1px solid grey; border-radius: 5px;'>"
        "<b>Predicție mâine</b><br>"
        "<span style='color:red;'>&#9679;</span> va necesita colectare<br>"
        "<span style='color:blue;'>&#9679;</span> sub prag"
        "</div>"
    )
    harta_predictie.get_root().html.add_child(folium.Element(legenda_predictie))

    st_folium(harta_predictie, width=1100, height=550, key="harta_predictie")
else:
    st.info("Niciun container cu coordonate valide pe această rută.")

st.caption("Roșu = modelul prezice că va trece pragul de colectare mâine; "
           "albastru = rămâne sub prag, conform predicției.")


# --- Detaliu pe un container specific ---
containere_ruta_ids = sorted(current_state_ruta["Id"].unique())
if containere_ruta_ids:
    st.markdown("**Detaliu container:**")
    id_ales = st.selectbox("Alege un container", containere_ruta_ids)

    uid_ales = current_state_ruta[current_state_ruta["Id"] == id_ales]["uid"].iloc[0]
    istoric_container = history_ruta[history_ruta["uid"] == uid_ales].sort_values("day")
    stare_container = current_state_ruta[current_state_ruta["uid"] == uid_ales].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel curent (ziua 0)", f"{stare_container['simulated_fill_level']:.1f}%")
    c2.metric("Predicție mâine", f"{stare_container['predicted_fill_level']:.1f}%")
    c3.metric("Prag colectare", f"{stare_container['threshold']:.0f}%")

    chart_container = alt.Chart(istoric_container).mark_line(point=True).encode(
        x=alt.X("day", title="Zi (0 = azi)"),
        y=alt.Y("simulated_fill_level", title="Fill level simulat (%)", scale=alt.Scale(domain=[0, 100])),
    )
    st.altair_chart(chart_container.properties(height=250), use_container_width=True)

st.caption(
    "Istoricul de 30 de zile e SIMULAT (o singură citire reală există per "
    "container) — vezi notebook-ul fill_level_prediction_draft.ipynb, "
    "Etapa 4.7, pentru limitarea acestei abordări."
)