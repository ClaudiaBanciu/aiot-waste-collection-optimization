"""
Interfata web interactiva (Streamlit)
-------------------------------------
Rulare:
    streamlit run app/main.py   (sau: python -m streamlit run app/main.py)

Foloseste clasele din app/components/:
  - StandardDistanceCalculator  (distanta standard, app/components/distance_calculator.py)
  - FillLevelPredictor  (Random Forest / arbori de decizie, app/components/fill_predictor.py)
"""

import os
import sys

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))            # ca sa gasim pachetul "components"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from components import (
    StandardDistanceCalculator,
    OSRMDistanceCalculator,
    FillLevelPredictor,
)

st.set_page_config(page_title="Gestiune deseuri - Sibiu", layout="wide")

INPUT_FILE = "data/processed/data_geocoded.csv"
PRAG = 80  # prag de umplere (%) peste care un container "necesita colectare"


# =====================================================================
# Incarcarea si pregatirea datelor
# =====================================================================
def citeste_csv_robust(cale):
    """Citeste CSV-ul indiferent cum a fost salvat (UTF-8, cp1252, separator
    ',' sau ';', sau chiar daca e de fapt un .xlsx redenumit)."""
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
    """Fill level poate fi text ('68%') sau numar zecimal (0.68) -> procent 0-100."""
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
    """Ordinea reala a traseului: depozitul de plecare (primul rand fara Id),
    apoi opririle in ordine cronologica, apoi depozitul de sosire (ultimul).
    Returneaza (lista_puncte[(lat,lon)], DataFrame ordonat)."""
    depozite = df_ruta[df_ruta["Id"].isna()]
    opriri = df_ruta[df_ruta["Id"].notna()].sort_values("Datetime")
    parti = []
    if len(depozite):
        parti.append(depozite.iloc[[0]])
    parti.append(opriri)
    if len(depozite) > 1:
        parti.append(depozite.iloc[[-1]])
    ordonat = pd.concat(parti).reset_index(drop=True)
    puncte = list(zip(ordonat["Latitude"], ordonat["Longitude"]))
    return puncte, ordonat


# =====================================================================
# Functii cache-uite peste clasele din components
# =====================================================================
@st.cache_data(show_spinner="Calculez distantele (standard + OSRM pe sosea)...")
def comparatie_distanta(puncte_tuple):
    """Compara cele doua metode de distanta pentru un traseu (neoptim vs optim):
    standard (linie dreapta) si OSRM (pe sosea, serverul public)."""
    puncte = [tuple(p) for p in puncte_tuple]
    standard = StandardDistanceCalculator().compara(puncte)
    osrm_calc = OSRMDistanceCalculator()
    osrm_ok = osrm_calc.disponibil()
    osrm = osrm_calc.compara(puncte) if osrm_ok else None
    return {"standard": standard, "osrm": osrm, "osrm_ok": osrm_ok}


@st.cache_resource(show_spinner=False)
def antreneaza_predictor():
    """Antreneaza o singura data modelul Random Forest pe toate containerele."""
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
    """Construieste o harta folium cu traseul dat (polilinie + puncte numerotate).
    Daca 'geometrie' e dat (conturul pe sosea de la OSRM), il deseneaza urmarind
    strazile; altfel uneste opririle in linie dreapta."""
    harta = folium.Map(
        location=[sum(p[0] for p in puncte) / len(puncte),
                  sum(p[1] for p in puncte) / len(puncte)],
        zoom_start=13,
    )
    if geometrie:
        # traseul real pe sosea (urmareste strazile)
        folium.PolyLine(geometrie, color=culoare, weight=3.5, opacity=0.8).add_to(harta)
    else:
        # linie dreapta intre opriri
        folium.PolyLine(puncte, color=culoare, weight=2.5, opacity=0.7).add_to(harta)
    for i, (lat, lon) in enumerate(puncte):
        rand = ordonat.iloc[i]
        detalii = detalii_punct(rand, i + 1, titlu_popup)
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=culoare, fill=True, fill_opacity=0.9,
            # tooltip = apare la HOVER (cursor pe punct); popup = ramane la click
            tooltip=folium.Tooltip(detalii, sticky=True),
            popup=folium.Popup(detalii, max_width=300),
        ).add_to(harta)
    return harta


def detalii_punct(rand, nr_ordine, titlu):
    """Construieste textul HTML cu detaliile unui punct (afisat la hover/click)."""
    if pd.isna(rand["Id"]):  # depozit (plecare / sosire)
        return (f"<b>#{nr_ordine} &middot; depozit</b><br>"
                f"{rand['Address']}<br><i>{titlu}</i>")

    umplere = f"{rand['Fill_num']:.0f}%" if pd.notna(rand["Fill_num"]) else "-"
    ora = rand["ora"] if pd.notna(rand["ora"]) else "-"
    return (
        f"<b>#{nr_ordine} &middot; {rand['Id']}</b><br>"
        f"{rand['Address']}<br>"
        f"Capacitate: {rand['Capacity']}<br>"
        f"Umplere: {umplere}<br>"
        f"Ora: {ora}<br>"
        f"Masina: {rand['Car']}"
    )


# =====================================================================
# UI
# =====================================================================
df = incarca_date()
st.title("Gestiune deseuri - Sibiu")

st.sidebar.header("Filtre")
rute_disponibile = sorted(df["route_id"].unique())
ruta_selectata = st.sidebar.selectbox("Ruta", rute_disponibile)
nivel_min, nivel_max = st.sidebar.slider("Nivel de umplere (%) - interval", 0, 100, (0, 100))

df_ruta = df[df["route_id"] == ruta_selectata].copy()
masini_ruta = sorted(df_ruta["Car"].dropna().unique())
df_filtrat = df_ruta[df_ruta["Fill_num"].between(nivel_min, nivel_max)].copy()

st.caption(f"Ruta {ruta_selectata} este deservita de: {', '.join(masini_ruta)}")

col1, col2, col3 = st.columns(3)
col1.metric("Containere afisate", int(df_filtrat["Id"].notna().sum()))
col2.metric("Nivel mediu umplere",
            f"{df_filtrat['Fill_num'].mean():.1f}%" if df_filtrat["Fill_num"].notna().any() else "-")
col3.metric("Masini pe ruta", len(masini_ruta))

# ---------------------------------------------------------------------
# Tabel containere
# ---------------------------------------------------------------------
st.subheader("Containere pe ruta")
with st.expander("Vezi datele tabelar", expanded=False):
    tabel = df_filtrat[df_filtrat["Id"].notna()][
        ["Id", "Car", "Address", "ora", "Fill_num", "Capacity"]
    ].copy()
    tabel = tabel.rename(columns={"Fill_num": "Fill Level (%)", "ora": "Ora"}).sort_values("Ora")
    st.dataframe(tabel, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# COMPARATIE TRASEU: neoptimizat vs optimizat
# ---------------------------------------------------------------------
st.subheader("Comparatie traseu: neoptimizat vs optimizat")

puncte, ordonat = puncte_rutei(df_ruta)

if len(puncte) < 2:
    st.warning("Ruta nu are suficiente puncte pentru calculul distantei.")
else:
    rez = comparatie_distanta(tuple(puncte))
    std, osrm, osrm_ok = rez["standard"], rez["osrm"], rez["osrm_ok"]

    # --- metrici: neoptim vs optim (distanta standard, linie dreapta) ---
    st.markdown("**Distanta standard (linie dreapta):**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Neoptimizat", f"{std['neoptimizat_km']:.2f} km")
    c2.metric("Optimizat", f"{std['optimizat_km']:.2f} km")
    c3.metric("Economie", f"{std['economie_%']:.1f}%")

    grafic_randuri = [
        {"metoda": "linie dreapta", "traseu": "neoptimizat", "km": std["neoptimizat_km"]},
        {"metoda": "linie dreapta", "traseu": "optimizat", "km": std["optimizat_km"]},
    ]

    # --- a doua metoda: distanta reala pe sosea (OSRM) ---
    if not osrm_ok:
        st.info("Serverul public OSRM nu raspunde (fara internet?) - se afiseaza doar "
                "distanta standard. Cand revine, apare si distanta reala pe sosea.")
    else:
        st.markdown("**Distanta pe sosea (OSRM):**")
        o1, o2, o3 = st.columns(3)
        o1.metric("Neoptimizat", f"{osrm['neoptimizat_km']:.2f} km")
        o2.metric("Optimizat", f"{osrm['optimizat_km']:.2f} km")
        factor = osrm["neoptimizat_km"] / std["neoptimizat_km"] if std["neoptimizat_km"] else 0
        o3.metric("Factor sosea / linie dreapta", f"{factor:.2f}x")
        grafic_randuri += [
            {"metoda": "pe sosea (OSRM)", "traseu": "neoptimizat", "km": osrm["neoptimizat_km"]},
            {"metoda": "pe sosea (OSRM)", "traseu": "optimizat", "km": osrm["optimizat_km"]},
        ]

    # --- grafic comparativ: cele doua metode x cele doua trasee ---
    grafic = alt.Chart(pd.DataFrame(grafic_randuri)).mark_bar().encode(
        x=alt.X("traseu:N", title=None),
        y=alt.Y("km:Q", title="Distanta (km)"),
        color=alt.Color("traseu:N", legend=None),
        column=alt.Column("metoda:N", title=None),
        tooltip=["metoda", "traseu", alt.Tooltip("km:Q", format=".2f")],
    ).properties(width=180, height=280)
    st.altair_chart(grafic, use_container_width=False)

    if osrm_ok:
        st.caption(
            "Distanta pe sosea (OSRM) e mai mare decat linia dreapta, pentru ca drumurile "
            "nu sunt drepte. Atentie: optimizarea nearest-neighbour minimizeaza distanta in "
            "**linie dreapta**; masurata pe sosea, ordinea optimizata nu e mereu mai scurta "
            "decat cea reala - un exemplu ca 'optim geometric' nu inseamna automat 'optim pe drum'."
        )

    # --- harti alaturate: traseul in ordine cronologica vs optimizat ---
    st.markdown("**Vizual pe harta** (stanga: ordinea reala / dreapta: ordinea optimizata)")
    ordine_opt = std["ordine_optima"]
    puncte_opt = [puncte[i] for i in ordine_opt]
    ordonat_opt = ordonat.iloc[ordine_opt].reset_index(drop=True)

    # daca OSRM e disponibil, putem desena traseul REAL pe sosea (urmareste strazile)
    geom_neopt = geom_opt = None
    if osrm_ok and osrm.get("geom_neopt"):
        pe_sosea = st.toggle("Deseneaza traseul pe sosea (OSRM)", value=True,
                             help="Bifat: linia urmareste strazile. Debifat: linie dreapta intre opriri.")
        if pe_sosea:
            geom_neopt = osrm["geom_neopt"]
            geom_opt = osrm["geom_opt"]

    m_stanga, m_dreapta = st.columns(2)
    with m_stanga:
        eticheta = f"{osrm['neoptimizat_km']:.2f} km pe sosea" if geom_neopt else f"{std['neoptimizat_km']:.2f} km linie dreapta"
        st.caption(f"Neoptimizat - {eticheta}")
        st_folium(deseneaza_traseu(puncte, ordonat, "red", "traseu neoptimizat", geom_neopt),
                  height=430, use_container_width=True, key="harta_neopt")
    with m_dreapta:
        eticheta = f"{osrm['optimizat_km']:.2f} km pe sosea" if geom_opt else f"{std['optimizat_km']:.2f} km linie dreapta"
        st.caption(f"Optimizat (nearest-neighbour) - {eticheta}")
        st_folium(deseneaza_traseu(puncte_opt, ordonat_opt, "green", "traseu optimizat", geom_opt),
                  height=430, use_container_width=True, key="harta_opt")

    st.caption(
        "Optimizarea foloseste euristica 'cel mai apropiat vecin' (nearest neighbour): "
        "pornind din depozit, se alege mereu cea mai apropiata oprire nevizitata. "
        "Nu garanteaza traseul optim absolut, dar e rapida si des folosita ca punct de plecare."
    )

# ---------------------------------------------------------------------
# PREDICTIE fill_level cu Random Forest (arbori de decizie)
# ---------------------------------------------------------------------
st.subheader("Predictie nivel de umplere - Random Forest (arbori de decizie)")

mae, predictii = antreneaza_predictor()
st.write(
    "Modelul prezice nivelul de umplere de **maine** pentru fiecare container, pornind "
    "de la nivelul curent, cel din ziua anterioara si rata de crestere (istoric simulat "
    "pe 7 zile). Model: **RandomForestRegressor**."
)
st.metric("Eroare medie absoluta (MAE) pe setul de test", f"{mae:.2f} puncte procentuale")

pred_ruta = predictii[predictii["route_id"] == ruta_selectata].copy()
sel_azi = int((pred_ruta["fill_curent"] >= PRAG).sum())
sel_maine = int((pred_ruta["fill_prezis"] >= PRAG).sum())

cc1, cc2, cc3 = st.columns(3)
cc1.metric(f"Containere >= {PRAG}% AZI (regula)", sel_azi)
cc2.metric(f"Containere >= {PRAG}% MAINE (predictie)", sel_maine)
cc3.metric("In plus fata de regula", sel_maine - sel_azi,
           help="Containere sub prag azi, dar prezise sa depaseasca pragul maine.")

with st.expander("Vezi predictiile pe ruta", expanded=False):
    afis = pred_ruta.rename(columns={
        "fill_curent": "Fill azi (%)", "fill_prezis": "Fill prezis maine (%)"})
    st.dataframe(afis[["Id", "Capacity", "Fill azi (%)", "Fill prezis maine (%)"]],
                 use_container_width=True, hide_index=True)

st.caption(
    "Diferenta fata de o regula fixa ('daca fill >= 80%, colectam'): modelul prinde "
    "containerele care AZI sunt sub prag, dar VOR depasi pragul maine - deci pot fi "
    "colectate proactiv, intr-o singura trecere. Nota: istoricul e simulat, nu real, "
    "asa ca acuratetea depinde de cat de realiste sunt ratele de umplere presupuse."
)
