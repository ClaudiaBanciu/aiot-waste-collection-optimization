import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.components.distante import Distante

def run_app(df: pd.DataFrame):
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

        dist_neoptim = Distante.distanta_traseu(puncte)
        _, dist_optim = Distante.optimizeaza_nearest_neighbor(puncte)

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
        fit_data = df_ruta[["ora_numerica", "Fill_num"]].dropna()
        x = fit_data["ora_numerica"].values
        y = fit_data["Fill_num"].values

        if len(set(x)) < 2:
            st.info("Not enough time variation on this route for a prediction.")
            st.stop()
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