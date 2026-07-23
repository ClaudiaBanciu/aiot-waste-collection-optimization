import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
import streamlit as st
import sys, os

class IntervalCitiri:
    #def __init__(self, df_ruta):
    #    self.df_ruta = df_ruta


    # --- Predicție per container individual (dacă are >=2 citiri) ---
    PRAG_MINIM_ORE = 0.5  # sub 30 de minute între citiri -> extrapolarea nu e sigură
    @staticmethod
    def interval_citiri_ore(id_container, df_ruta: pd.DataFrame):
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
