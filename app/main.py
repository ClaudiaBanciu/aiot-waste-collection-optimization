"""
Interactive web interface (Streamlit)
-------------------------------------
Run:
    streamlit run app/main.py   (or: python -m streamlit run app/main.py)

Uses the classes from app/components/:
  - StandardDistanceCalculator / OSRMDistanceCalculator  (app/components/distance_calculator.py)
  - FillLevelPredictor  (Random Forest / decision trees, app/components/fill_predictor.py)
"""

import os
import sys

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))            # so we can find the "components" package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from components import (
    StandardDistanceCalculator,
    OSRMDistanceCalculator,
    FillLevelPredictor,
)

st.set_page_config(page_title="Waste Management - Sibiu", layout="wide")

INPUT_FILE = "data/processed/data_geocoded.csv"
PRAG = 80  # fill threshold (%) above which a container "needs collection"


# =====================================================================
# Loading and preparing the data
# =====================================================================
def citeste_csv_robust(cale):
    """Read the CSV no matter how it was saved (UTF-8, cp1252, separator
    ',' or ';', or even if it is actually a renamed .xlsx)."""
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
    raise SystemExit(f"Could not read {cale} with any known encoding/separator.")


def normalizeaza_fill_level(serie):
    """Fill level can be text ('68%') or a decimal number (0.68) -> percent 0-100."""
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
    df["Fill_num"] = normalizeaza_fill_level(df["fill_level"])
    df["ora"] = df["Datetime"].dt.strftime("%H:%M")
    return df


def puncte_rutei(df_ruta):
    """The real order of the trip: the departure depot (first row without Id),
    then the stops in chronological order. The vehicle does NOT return to a depot,
    so the arrival depot is dropped and the last stop is the last bin.
    Adds a per-route bin number that starts at route_id * 1000 (route 1 -> 1000,
    route 2 -> 2000, route 3 -> 3000). Returns (list_of_points[(lat,lon)],
    ordered DataFrame)."""
    depozite = df_ruta[df_ruta["Id"].isna()]
    opriri = df_ruta[df_ruta["Id"].notna()].sort_values("Datetime")
    parti = []
    if len(depozite):
        parti.append(depozite.iloc[[0]])   # departure depot only (no return)
    parti.append(opriri)
    ordonat = pd.concat(parti).reset_index(drop=True)

    # bin numbering per route, starting at route_id * 1000
    base = int(df_ruta["route_id"].iloc[0]) * 1000
    numere, k = [], 0
    for _, r in ordonat.iterrows():
        if pd.isna(r["Id"]):
            numere.append(None)          # depots are not bins
        else:
            numere.append(base + k)
            k += 1
    ordonat["nr_pubela"] = numere

    puncte = list(zip(ordonat["Latitude"], ordonat["Longitude"]))
    return puncte, ordonat


# =====================================================================
# Cached functions on top of the classes from components
# =====================================================================
@st.cache_data(show_spinner="Computing distances (standard + OSRM road)...")
def comparatie_distanta(puncte_tuple):
    """Compare the two distance methods for a trip (non-optimized vs optimized):
    standard (straight line) and OSRM (on the road, the public server)."""
    puncte = [tuple(p) for p in puncte_tuple]
    standard = StandardDistanceCalculator().compara(puncte)
    osrm_calc = OSRMDistanceCalculator()
    osrm_ok = osrm_calc.disponibil()
    osrm = osrm_calc.compara(puncte) if osrm_ok else None
    return {"standard": standard, "osrm": osrm, "osrm_ok": osrm_ok}


@st.cache_resource(show_spinner=False)
def antreneaza_predictor():
    """Train the Random Forest model once on all containers."""
    df = incarca_date()
    containere = df[df["Id"].notna() & df["Capacity"].notna() & df["Fill_num"].notna()][
        ["Id", "Capacity", "route_id"]
    ].copy()
    containere["fill_level"] = df.loc[containere.index, "Fill_num"].astype(float)
    predictor = FillLevelPredictor(n_zile=7)
    predictor.antreneaza(containere)
    predictii = predictor.prezice_ziua_urmatoare()
    return predictor.mae_, predictii


def deseneaza_traseu(puncte, ordonat, culoare, titlu_popup, geometrie=None):
    """Build a folium map with the given trip. The road is drawn as a subtle
    dotted line (visible but not prominent); each bin is a Pin marker. If
    'geometrie' is provided (the road outline from OSRM), the dotted line
    follows the streets; otherwise it connects the stops in a straight line."""
    harta = folium.Map(
        location=[sum(p[0] for p in puncte) / len(puncte),
                  sum(p[1] for p in puncte) / len(puncte)],
        zoom_start=13,
    )
    # black dotted route line, a bit thicker
    linie = geometrie if geometrie else puncte
    folium.PolyLine(linie, color="black", weight=3, opacity=0.7,
                    dash_array="2, 8").add_to(harta)
    ordine_parcurs = 0
    for i, (lat, lon) in enumerate(puncte):
        rand = ordonat.iloc[i]
        if pd.isna(rand["Id"]):
            detalii = detalii_punct(rand, titlu_popup, None)
            icon = folium.Icon(color="gray", icon="home", prefix="fa")  # depot
        else:
            ordine_parcurs += 1                                          # order along the route
            detalii = detalii_punct(rand, titlu_popup, ordine_parcurs)
            icon = folium.Icon(color=culoare, icon="trash", prefix="fa")  # bin (Pin)
        folium.Marker(
            location=[lat, lon],
            # tooltip = shown on HOVER (cursor over the pin); popup = stays on click
            tooltip=folium.Tooltip(detalii, sticky=True),
            popup=folium.Popup(detalii, max_width=300),
            icon=icon,
        ).add_to(harta)
    return harta


def detalii_punct(rand, titlu, ordine_parcurs):
    """Build the HTML text with a point's details (shown on hover/click).
    ordine_parcurs = the bin's position along the route (1, 2, 3, ...), shown in
    addition to the fixed per-route bin number."""
    if pd.isna(rand["Id"]):  # depot (departure / arrival)
        return (f"<b>Depot</b><br>"
                f"{rand['Address']}<br><i>{titlu}</i>")

    umplere = f"{rand['Fill_num']:.0f}%" if pd.notna(rand["Fill_num"]) else "-"
    ora = rand["ora"] if pd.notna(rand["ora"]) else "-"
    nr = int(rand["nr_pubela"]) if pd.notna(rand["nr_pubela"]) else "-"
    return (
        f"<b>Stop {ordine_parcurs} &middot; Bin #{nr} &middot; {rand['Id']}</b><br>"
        f"{rand['Address']}<br>"
        f"Capacity: {rand['Capacity']}<br>"
        f"Fill: {umplere}<br>"
        f"Time: {ora}<br>"
        f"Vehicle: {rand['Car']}"
    )


# =====================================================================
# UI
# =====================================================================
df = incarca_date()
st.title("Waste Management - Sibiu")

st.sidebar.header("Filters")
rute_disponibile = sorted(df["route_id"].unique())
ruta_selectata = st.sidebar.selectbox("Route", rute_disponibile)
nivel_min, nivel_max = st.sidebar.slider("Fill level (%) - range", 0, 100, (0, 100))

df_ruta = df[df["route_id"] == ruta_selectata].copy()
masini_ruta = sorted(df_ruta["Car"].dropna().unique())
df_filtrat = df_ruta[df_ruta["Fill_num"].between(nivel_min, nivel_max)].copy()

st.caption(f"Route {ruta_selectata} is served by: {', '.join(masini_ruta)}")

col1, col2, col3 = st.columns(3)
col1.metric("Containers shown", int(df_filtrat["Id"].notna().sum()))
col2.metric("Average fill level",
            f"{df_filtrat['Fill_num'].mean():.1f}%" if df_filtrat["Fill_num"].notna().any() else "-")
col3.metric("Vehicles on route", len(masini_ruta))

# ---------------------------------------------------------------------
# Container table
# ---------------------------------------------------------------------
st.subheader("Containers on route")
with st.expander("View data as table", expanded=False):
    tabel = df_filtrat[df_filtrat["Id"].notna()][
        ["Id", "Car", "Address", "ora", "Fill_num", "Capacity"]
    ].copy()
    tabel = tabel.rename(columns={"Fill_num": "Fill Level (%)", "ora": "Time",
                                  "Car": "Vehicle"}).sort_values("Time")
    st.dataframe(tabel, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# ROUTE COMPARISON: non-optimized vs optimized
# ---------------------------------------------------------------------
st.subheader("Route comparison: non-optimized vs optimized")

puncte, ordonat = puncte_rutei(df_ruta)

if len(puncte) < 2:
    st.warning("The route does not have enough points to compute the distance.")
else:
    rez = comparatie_distanta(tuple(puncte))
    std, osrm, osrm_ok = rez["standard"], rez["osrm"], rez["osrm_ok"]

    # --- metrics: non-optimized vs optimized (standard distance, straight line) ---
    st.markdown("**Standard distance (straight line):**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Non-optimized", f"{std['neoptimizat_km']:.2f} km")
    c2.metric("Optimized", f"{std['optimizat_km']:.2f} km")
    c3.metric("Saving", f"{std['economie_%']:.1f}%")

    grafic_randuri = [
        {"method": "straight line", "route": "non-optimized", "km": std["neoptimizat_km"]},
        {"method": "straight line", "route": "optimized", "km": std["optimizat_km"]},
    ]

    # --- second method: real road distance (OSRM) ---
    if not osrm_ok:
        st.info("The public OSRM server is not responding (no internet?) - only the "
                "standard distance is shown. When it is back, the real road distance appears too.")
    else:
        st.markdown("**Road distance (OSRM):**")
        o1, o2, o3 = st.columns(3)
        o1.metric("Non-optimized", f"{osrm['neoptimizat_km']:.2f} km")
        o2.metric("Optimized", f"{osrm['optimizat_km']:.2f} km")
        factor = osrm["neoptimizat_km"] / std["neoptimizat_km"] if std["neoptimizat_km"] else 0
        o3.metric("Road / straight-line factor", f"{factor:.2f}x")
        grafic_randuri += [
            {"method": "road (OSRM)", "route": "non-optimized", "km": osrm["neoptimizat_km"]},
            {"method": "road (OSRM)", "route": "optimized", "km": osrm["optimizat_km"]},
        ]

    # --- comparison chart: the two methods x the two trips ---
    grafic = alt.Chart(pd.DataFrame(grafic_randuri)).mark_bar().encode(
        x=alt.X("route:N", title=None),
        y=alt.Y("km:Q", title="Distance (km)"),
        color=alt.Color("route:N", legend=None),
        column=alt.Column("method:N", title=None),
        tooltip=["method", "route", alt.Tooltip("km:Q", format=".2f")],
    ).properties(width=180, height=280)
    st.altair_chart(grafic, use_container_width=False)

    if osrm_ok:
        st.caption(
            "The road distance (OSRM) is larger than the straight line, because roads are "
            "not straight. Note: nearest-neighbour optimization minimizes the **straight-line** "
            "distance; measured on the road, the optimized order is not always shorter than the "
            "real one - an example that 'geometric optimum' does not automatically mean 'road optimum'."
        )

    # --- side-by-side maps: the trip in chronological order vs optimized ---
    st.markdown("**Map view** (left: real order / right: optimized order)")
    ordine_opt = std["ordine_optima"]
    puncte_opt = [puncte[i] for i in ordine_opt]
    ordonat_opt = ordonat.iloc[ordine_opt].reset_index(drop=True)

    # always draw the REAL OSRM road trip (follows the streets) when available
    geom_neopt = geom_opt = None
    if osrm_ok and osrm.get("geom_neopt"):
        geom_neopt = osrm["geom_neopt"]
        geom_opt = osrm["geom_opt"]

    m_stanga, m_dreapta = st.columns(2)
    with m_stanga:
        eticheta = f"{osrm['neoptimizat_km']:.2f} km on road" if geom_neopt else f"{std['neoptimizat_km']:.2f} km straight line"
        st.caption(f"Non-optimized - {eticheta}")
        st_folium(deseneaza_traseu(puncte, ordonat, "red", "non-optimized route", geom_neopt),
                  height=430, use_container_width=True, key="harta_neopt")
    with m_dreapta:
        eticheta = f"{osrm['optimizat_km']:.2f} km on road" if geom_opt else f"{std['optimizat_km']:.2f} km straight line"
        st.caption(f"Optimized (nearest-neighbour) - {eticheta}")
        st_folium(deseneaza_traseu(puncte_opt, ordonat_opt, "green", "optimized route", geom_opt),
                  height=430, use_container_width=True, key="harta_opt")

    st.caption(
        "The optimization uses the 'nearest neighbour' heuristic: starting from the depot, it "
        "always picks the closest not-yet-visited stop. It does not guarantee the absolute optimal "
        "route, but it is fast and often used as a starting point."
    )

# ---------------------------------------------------------------------
# fill_level PREDICTION with Random Forest (decision trees)
# ---------------------------------------------------------------------
st.subheader("Fill level prediction - Random Forest (decision trees)")

mae, predictii = antreneaza_predictor()
st.write(
    "The model predicts **tomorrow's** fill level for each container, based on the current "
    "level, the previous day's level and the growth rate (simulated 7-day history). "
    "Model: **RandomForestRegressor**."
)
st.metric("Mean absolute error (MAE) on the test set", f"{mae:.2f} percentage points")

pred_ruta = predictii[predictii["route_id"] == ruta_selectata].copy()
sel_azi = int((pred_ruta["fill_curent"] >= PRAG).sum())
sel_maine = int((pred_ruta["fill_prezis"] >= PRAG).sum())

cc1, cc2, cc3 = st.columns(3)
cc1.metric(f"Containers >= {PRAG}% TODAY (rule)", sel_azi)
cc2.metric(f"Containers >= {PRAG}% TOMORROW (prediction)", sel_maine)
cc3.metric("Extra vs the rule", sel_maine - sel_azi,
           help="Containers below the threshold today, but predicted to exceed it tomorrow.")

with st.expander("View predictions for the route", expanded=False):
    afis = pred_ruta.rename(columns={
        "fill_curent": "Fill today (%)", "fill_prezis": "Predicted fill tomorrow (%)"})
    st.dataframe(afis[["Id", "Capacity", "Fill today (%)", "Predicted fill tomorrow (%)"]],
                 use_container_width=True, hide_index=True)

st.caption(
    "The difference from a fixed rule ('if fill >= 80%, collect'): the model catches the "
    "containers that are below the threshold TODAY but WILL exceed it tomorrow - so they can "
    "be collected proactively, in a single pass. Note: the history is simulated, not real, so "
    "the accuracy depends on how realistic the assumed fill rates are."
)
