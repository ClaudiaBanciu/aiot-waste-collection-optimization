"""Our full Streamlit interface, moved out of main.py into a component.

main.py stays thin and only calls run(df) from here, so it does not conflict
with the rest of the team's work (webscript.py is left untouched).

What this interface shows:
  - route filter + fill-level range filter;
  - quick metrics and the container table;
  - route comparison non-optimized vs optimized, with two distance methods:
    standard (straight line) and OSRM (real road distance);
  - two maps with numbered pins (the stop order) and a black dotted route line;
  - next-day fill-level prediction with a Random Forest.
"""
import os
import sys

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.distance_calculator import (
    StandardDistanceCalculator,
    OSRMDistanceCalculator,
)
from app.components.fill_predictor import FillLevelPredictor

PRAG = 80  # fill threshold (%) above which a container "needs collection"


# =====================================================================
# Helpers
# =====================================================================
def puncte_rutei(df_ruta):
    """The real order of the trip: the departure depot (first row without Id),
    then the stops in chronological order. The vehicle does NOT return to a depot,
    so the arrival depot is dropped and the last stop is the last bin.
    Returns (list_of_points[(lat,lon)], ordered DataFrame)."""
    depozite = df_ruta[df_ruta["Id"].isna()]
    opriri = df_ruta[df_ruta["Id"].notna()].sort_values("Datetime")
    parti = []
    if len(depozite):
        parti.append(depozite.iloc[[0]])   # departure depot only (no return)
    parti.append(opriri)
    ordonat = pd.concat(parti).reset_index(drop=True)

    # --- DISABLED: bin numbering per route, starting at route_id * 1000 -------
    # Kept for reference only. The map now shows the stop order instead, so this
    # number is no longer computed and no longer appears on hover.
    # base = int(df_ruta["route_id"].iloc[0]) * 1000
    # numere, k = [], 0
    # for _, r in ordonat.iterrows():
    #     if pd.isna(r["Id"]):
    #         numere.append(None)          # depots are not bins
    #     else:
    #         numere.append(base + k)
    #         k += 1
    # ordonat["nr_pubela"] = numere
    # -------------------------------------------------------------------------

    puncte = list(zip(ordonat["Latitude"], ordonat["Longitude"]))
    return puncte, ordonat


@st.cache_data(show_spinner="Calculez distanțele (standard + OSRM pe șosea)...")
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
def antreneaza_predictor(_df):
    """Train the Random Forest model once on all containers.
    The leading underscore tells Streamlit not to hash the DataFrame."""
    containere = _df[_df["Id"].notna() & _df["Capacity"].notna() & _df["Fill_num"].notna()][
        ["Id", "Capacity", "route_id"]
    ].copy()
    containere["fill_level"] = _df.loc[containere.index, "Fill_num"].astype(float)
    predictor = FillLevelPredictor(n_zile=7)
    predictor.antreneaza(containere)
    predictii = predictor.prezice_ziua_urmatoare()
    return predictor.mae_, predictii


def detalii_punct(rand, titlu, ordine_parcurs):
    """Build the HTML text with a point's details (shown on hover/click).
    ordine_parcurs = the bin's position along the route (1, 2, 3, ...)."""
    if pd.isna(rand["Id"]):  # depot (departure)
        return (f"<b>Depozit</b><br>"
                f"{rand['Address']}<br><i>{titlu}</i>")

    umplere = f"{rand['Fill_num']:.0f}%" if pd.notna(rand["Fill_num"]) else "-"
    ora = rand["ora"] if pd.notna(rand["ora"]) else "-"
    # --- DISABLED: per-route bin number (route_id * 1000) --------------------
    # nr = int(rand["nr_pubela"]) if pd.notna(rand["nr_pubela"]) else "-"
    # ------------------------------------------------------------------------
    return (
        # f"<b>Oprirea {ordine_parcurs} &middot; Pubela #{nr} &middot; {rand['Id']}</b><br>"
        f"<b>Oprirea {ordine_parcurs} &middot; {rand['Id']}</b><br>"
        f"{rand['Address']}<br>"
        f"Capacitate: {rand['Capacity']}<br>"
        f"Umplere: {umplere}<br>"
        f"Ora: {ora}<br>"
        f"Mașina: {rand['Car']}"
    )


def pin_numerotat(numar, culoare):
    """A teardrop map pin with the stop order number written inside it.
    Built with inline HTML/CSS (DivIcon), so it needs no extra icon library."""
    html = (
        f'<div style="position:relative;width:28px;height:28px;">'
        f'<div style="width:24px;height:24px;margin:0 2px;background:{culoare};'
        f'border:2px solid #fff;border-radius:50% 50% 50% 0;'
        f'transform:rotate(-45deg);box-shadow:0 1px 4px rgba(0,0,0,.45);"></div>'
        f'<div style="position:absolute;left:0;top:0;width:28px;height:26px;'
        f'display:flex;align-items:center;justify-content:center;color:#fff;'
        f'font-family:sans-serif;font-size:11px;font-weight:700;">{numar}</div>'
        f'</div>'
    )
    return folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 28))


def deseneaza_traseu(puncte, ordonat, culoare, titlu_popup, geometrie=None):
    """Build a folium map with the given trip. The road is drawn as a black
    dotted line; each bin is a pin showing its stop order number. If 'geometrie'
    is provided (the road outline from OSRM), the dotted line follows the
    streets; otherwise it connects the stops in a straight line."""
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
            ordine_parcurs += 1                                # order along the route
            detalii = detalii_punct(rand, titlu_popup, ordine_parcurs)
            icon = pin_numerotat(ordine_parcurs, culoare)      # pin with the stop number
        folium.Marker(
            location=[lat, lon],
            # tooltip = shown on HOVER (cursor over the pin); popup = stays on click
            tooltip=folium.Tooltip(detalii, sticky=True),
            popup=folium.Popup(detalii, max_width=300),
            icon=icon,
        ).add_to(harta)
    return harta


# =====================================================================
# Entry point, called from main.py
# =====================================================================
def run(df: pd.DataFrame):
    """Render the whole interface."""
    st.title("🚛 Gestiune deșeuri - Sibiu")

    # --- Sidebar: route + fill-level range ---
    st.sidebar.header("Filtre")
    rute_disponibile = sorted(df["route_id"].unique())
    ruta_selectata = st.sidebar.selectbox("Rută", rute_disponibile)
    nivel_min, nivel_max = st.sidebar.slider(
        "Nivel de umplere (%) — interval", 0, 100, (0, 100)
    )

    df_ruta = df[df["route_id"] == ruta_selectata].copy()
    masini_ruta = sorted(df_ruta["Car"].dropna().unique())
    df_filtrat = df_ruta[df_ruta["Fill_num"].between(nivel_min, nivel_max)].copy()

    st.caption(f"Ruta {ruta_selectata} este deservită de: {', '.join(masini_ruta)}")

    # --- Quick metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Containere afișate", int(df_filtrat["Id"].notna().sum()))
    col2.metric("Nivel mediu umplere",
                f"{df_filtrat['Fill_num'].mean():.1f}%"
                if df_filtrat["Fill_num"].notna().any() else "-")
    col3.metric("Mașini pe rută", len(masini_ruta))

    # --- Container table ---
    st.subheader("Containere pe rută")
    with st.expander("Vezi datele tabelar", expanded=False):
        tabel = df_filtrat[df_filtrat["Id"].notna()][
            ["Id", "Car", "Address", "ora", "Fill_num", "Capacity"]
        ].copy()
        tabel = tabel.rename(columns={"Fill_num": "Nivel umplere (%)", "ora": "Oră",
                                      "Car": "Mașina", "Address": "Adresă",
                                      "Capacity": "Capacitate"}).sort_values("Oră")
        st.dataframe(tabel, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # ROUTE COMPARISON: non-optimized vs optimized
    # -----------------------------------------------------------------
    st.subheader("Comparație traseu: neoptimizat vs optimizat")

    puncte, ordonat = puncte_rutei(df_ruta)

    if len(puncte) < 2:
        st.warning("Ruta nu are suficiente puncte pentru calculul distanței.")
    else:
        rez = comparatie_distanta(tuple(puncte))
        std, osrm, osrm_ok = rez["standard"], rez["osrm"], rez["osrm_ok"]

        # --- standard distance (straight line) ---
        st.markdown("**Distanța standard (linie dreaptă):**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Neoptimizat", f"{std['neoptimizat_km']:.2f} km")
        c2.metric("Optimizat", f"{std['optimizat_km']:.2f} km")
        c3.metric("Economie", f"{std['economie_%']:.1f}%")

        grafic_randuri = [
            {"metoda": "linie dreaptă", "traseu": "neoptimizat", "km": std["neoptimizat_km"]},
            {"metoda": "linie dreaptă", "traseu": "optimizat", "km": std["optimizat_km"]},
        ]

        # --- second method: real road distance (OSRM) ---
        if not osrm_ok:
            st.info("Serverul public OSRM nu răspunde (fără internet?) - se afișează doar "
                    "distanța standard. Când revine, apare și distanța reală pe șosea.")
        else:
            st.markdown("**Distanța pe șosea (OSRM):**")
            o1, o2, o3 = st.columns(3)
            o1.metric("Neoptimizat", f"{osrm['neoptimizat_km']:.2f} km")
            o2.metric("Optimizat", f"{osrm['optimizat_km']:.2f} km")
            factor = osrm["neoptimizat_km"] / std["neoptimizat_km"] if std["neoptimizat_km"] else 0
            o3.metric("Factor șosea / linie dreaptă", f"{factor:.2f}x")
            grafic_randuri += [
                {"metoda": "pe șosea (OSRM)", "traseu": "neoptimizat", "km": osrm["neoptimizat_km"]},
                {"metoda": "pe șosea (OSRM)", "traseu": "optimizat", "km": osrm["optimizat_km"]},
            ]

        # --- comparison chart ---
        grafic = alt.Chart(pd.DataFrame(grafic_randuri)).mark_bar().encode(
            x=alt.X("traseu:N", title=None),
            y=alt.Y("km:Q", title="Distanța (km)"),
            color=alt.Color("traseu:N", legend=None),
            column=alt.Column("metoda:N", title=None),
            tooltip=["metoda", "traseu", alt.Tooltip("km:Q", format=".2f")],
        ).properties(width=180, height=280)
        st.altair_chart(grafic, use_container_width=False)

        if osrm_ok:
            st.caption(
                "Distanța pe șosea (OSRM) e mai mare decât linia dreaptă, pentru că drumurile "
                "nu sunt drepte. Atenție: optimizarea nearest-neighbour minimizează distanța în "
                "**linie dreaptă**; măsurată pe șosea, ordinea optimizată nu e mereu mai scurtă "
                "decât cea reală - un exemplu că 'optim geometric' nu înseamnă automat 'optim pe drum'."
            )

        # --- side-by-side maps ---
        st.markdown("**Vizual pe hartă** (stânga: ordinea reală / dreapta: ordinea optimizată)")
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
            eticheta = (f"{osrm['neoptimizat_km']:.2f} km pe șosea" if geom_neopt
                        else f"{std['neoptimizat_km']:.2f} km linie dreaptă")
            st.caption(f"Neoptimizat - {eticheta}")
            st_folium(deseneaza_traseu(puncte, ordonat, "red", "traseu neoptimizat", geom_neopt),
                      height=430, use_container_width=True, key="harta_neopt")
        with m_dreapta:
            eticheta = (f"{osrm['optimizat_km']:.2f} km pe șosea" if geom_opt
                        else f"{std['optimizat_km']:.2f} km linie dreaptă")
            st.caption(f"Optimizat (nearest-neighbour) - {eticheta}")
            st_folium(deseneaza_traseu(puncte_opt, ordonat_opt, "green", "traseu optimizat", geom_opt),
                      height=430, use_container_width=True, key="harta_opt")

        st.caption(
            "Optimizarea folosește euristica 'cel mai apropiat vecin' (nearest neighbour): "
            "pornind din depozit, se alege mereu cea mai apropiată oprire nevizitată. "
            "Nu garantează traseul optim absolut, dar e rapidă și des folosită ca punct de plecare."
        )

    # -----------------------------------------------------------------
    # fill_level PREDICTION with Random Forest (decision trees)
    # -----------------------------------------------------------------
    st.subheader("Predicție nivel de umplere - Random Forest (arbori de decizie)")

    mae, predictii = antreneaza_predictor(df)
    st.write(
        "Modelul prezice nivelul de umplere de **mâine** pentru fiecare container, pornind "
        "de la nivelul curent, cel din ziua anterioară și rata de creștere (istoric simulat "
        "pe 7 zile). Model: **RandomForestRegressor**."
    )
    st.metric("Eroare medie absolută (MAE) pe setul de test", f"{mae:.2f} puncte procentuale")

    pred_ruta = predictii[predictii["route_id"] == ruta_selectata].copy()
    sel_azi = int((pred_ruta["fill_curent"] >= PRAG).sum())
    sel_maine = int((pred_ruta["fill_prezis"] >= PRAG).sum())

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(f"Containere >= {PRAG}% AZI (regulă)", sel_azi)
    cc2.metric(f"Containere >= {PRAG}% MÂINE (predicție)", sel_maine)
    cc3.metric("În plus față de regulă", sel_maine - sel_azi,
               help="Containere sub prag azi, dar prezise să depășească pragul mâine.")

    with st.expander("Vezi predicțiile pe rută", expanded=False):
        afis = pred_ruta.rename(columns={
            "fill_curent": "Umplere azi (%)", "fill_prezis": "Umplere prezisă mâine (%)"})
        st.dataframe(afis[["Id", "Capacity", "Umplere azi (%)", "Umplere prezisă mâine (%)"]],
                     use_container_width=True, hide_index=True)

    st.caption(
        "Diferența față de o regulă fixă ('dacă umplerea >= 80%, colectăm'): modelul prinde "
        "containerele care AZI sunt sub prag, dar VOR depăși pragul mâine - deci pot fi "
        "colectate proactiv, într-o singură trecere. Notă: istoricul e simulat, nu real, "
        "așa că acuratețea depinde de cât de realiste sunt ratele de umplere presupuse."
    )
