import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SOM · Fleet Report", layout="wide", page_icon="◆")

# ============================================================
# STIL CUSTOM — editorial, alb, tipografie mare
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    /* Ascunde toolbar-ul default de Streamlit (Deploy, iconițe) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; background: #ffffff; }

    .stApp { background: #fbfaf8; }

    /* Masthead editorial */
    .masthead {
        border-bottom: 3px solid #14171c;
        padding-bottom: 1.2rem;
        margin-bottom: 2rem;
    }
    .masthead .kicker {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 2px;
        text-transform: uppercase; color: #a35b2c;
        margin-bottom: 0.6rem;
    }
    .masthead h1 {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem; font-weight: 600; color: #14171c;
        margin: 0; line-height: 1.1; letter-spacing: -0.5px;
    }
    .masthead .dek {
        font-size: 1rem; color: #6b6459; margin-top: 0.6rem; max-width: 640px;
    }

    /* Statistici mari, stil editorial (nu carduri) */
    .stat-row { display: flex; gap: 2.5rem; margin: 1.8rem 0 2.2rem 0; flex-wrap: wrap; }
    .stat { border-left: 2px solid #e4ded3; padding-left: 1rem; }
    .stat .num {
        font-family: 'Fraunces', serif;
        font-size: 2.2rem; font-weight: 600; color: #14171c; line-height: 1;
    }
    .stat .num small { font-size: 1.1rem; font-weight: 500; color: #a39a89; }
    .stat .lbl { font-size: 0.78rem; color: #8a8175; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 0.35rem; }
    .stat .chg { font-size: 0.82rem; color: #3f7a5c; font-weight: 600; margin-top: 0.3rem; }

    .section-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; color: #a35b2c;
        margin: 2rem 0 0.8rem 0;
        border-top: 1px solid #e4ded3; padding-top: 1.4rem;
    }

    div[data-testid="stDataFrame"] { border: 1px solid #e4ded3; border-radius: 4px; }

    /* Sidebar - fix pentru selectbox funcțional */
    section[data-testid="stSidebar"] {
        background: #14171c;
        border-right: none;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cfcabf !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background: #1e2330;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #1e2330 !important;
        border-color: #33394a !important;
        color: #f1f4fa !important;
    }
    section[data-testid="stSidebar"] .kicker-side {
        font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase;
        color: #a35b2c !important; font-weight: 700;
    }

    div[data-baseweb="select"] > div { background: #ffffff; border-color: #e4ded3; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Date
# ============================================================
@st.cache_data
def incarca_date():
    df = pd.read_csv("data/processed/dataset_geocoded.csv", encoding="utf-8")
    df = df.dropna(subset=["lat", "lon"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

df = incarca_date()
CULORI_RUTA = {1: "#c1543a", 2: "#3f7a5c", 3: "#2c5a7d"}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="kicker-side">ROUTE CONTROL</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    ruta_selectata = st.selectbox("Rută activă", sorted(df["route_id"].unique()),
                                    format_func=lambda r: f"Ruta {r:02d}", label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="kicker-side">FLOTĂ</div>', unsafe_allow_html=True)
    for r, culoare in CULORI_RUTA.items():
        n = len(df[df["route_id"] == r])
        st.markdown(f"<span style='color:{culoare}; font-size:1rem;'>●</span> &nbsp;Ruta {r:02d} <span style='color:#8a8175'>· {n} containere</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"{len(df)} containere active · {df['route_id'].nunique()} rute")

df_ruta = df[df["route_id"] == ruta_selectata].sort_values("Datetime").reset_index(drop=True)
coords = list(zip(df_ruta["lat"], df_ruta["lon"]))
culoare_ruta = CULORI_RUTA.get(ruta_selectata, "#2c5a7d")

# ============================================================
# Calcule
# ============================================================
def distanta_totala(lista_coords):
    total = 0.0
    for i in range(len(lista_coords) - 1):
        total += geodesic(lista_coords[i], lista_coords[i + 1]).km
    return total

def nearest_neighbor(lista_coords):
    ramase = lista_coords.copy()
    ruta = [ramase.pop(0)]
    while ramase:
        ultimul = ruta[-1]
        urmatorul = min(ramase, key=lambda c: geodesic(ultimul, c).km)
        ruta.append(urmatorul)
        ramase.remove(urmatorul)
    return ruta

distanta_neoptimizata = distanta_totala(coords)
ruta_optimizata = nearest_neighbor(coords.copy())
distanta_optimizata = distanta_totala(ruta_optimizata)
economie = distanta_neoptimizata - distanta_optimizata
procent_economie = (economie / distanta_neoptimizata * 100) if distanta_neoptimizata > 0 else 0

df["ora"] = df["Datetime"].dt.hour
model = LinearRegression()
model.fit(df[["ora"]], df["fill_level"])
ora_curenta = pd.Timestamp.now().hour
predictie_curenta = model.predict(pd.DataFrame({"ora": [ora_curenta]}))[0]

# ============================================================
# MASTHEAD
# ============================================================
st.markdown(f"""
<div class="masthead">
    <div class="kicker">SOM · Waste Management Division</div>
    <h1>Raport operațional — Ruta {ruta_selectata:02d}</h1>
    <div class="dek">Analiză traseu, distanțe parcurse și estimare grad de umplere pentru rețeaua de colectare Sibiu.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STAT ROW (editorial, nu carduri)
# ============================================================
st.markdown(f"""
<div class="stat-row">
    <div class="stat">
        <div class="num">{distanta_neoptimizata:.1f}<small> km</small></div>
        <div class="lbl">Distanță neoptimizată</div>
    </div>
    <div class="stat">
        <div class="num">{distanta_optimizata:.1f}<small> km</small></div>
        <div class="lbl">Distanță optimizată</div>
        <div class="chg">↓ {economie:.1f} km economisiți</div>
    </div>
    <div class="stat">
        <div class="num">{procent_economie:.0f}<small>%</small></div>
        <div class="lbl">Reducere traseu</div>
    </div>
    <div class="stat">
        <div class="num">{predictie_curenta:.0f}<small>%</small></div>
        <div class="lbl">Umplere estimată, ora {ora_curenta}:00</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAP + TABEL
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-label">Traseu pe hartă</div>', unsafe_allow_html=True)
    harta = folium.Map(
        location=[df_ruta["lat"].mean(), df_ruta["lon"].mean()],
        zoom_start=14,
        tiles="CartoDB positron"
    )
    for i, row in df_ruta.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="#14171c", weight=1,
            fill=True, fill_color=culoare_ruta, fill_opacity=0.9,
            popup=f"<b>{row['Id']}</b><br>{row['Capacity']}<br>Umplere: {row['fill_level']}%"
        ).add_to(harta)
    folium.PolyLine(coords, color=culoare_ruta, weight=2, opacity=0.55).add_to(harta)
    st_folium(harta, width=700, height=430)

with col2:
    st.markdown('<div class="section-label">Containere pe rută</div>', unsafe_allow_html=True)
    st.dataframe(
        df_ruta[["Id", "Capacity", "fill_level", "Datetime"]],
        height=430,
        width="stretch",
        hide_index=True
    )

# ============================================================
# Predicție interactivă
# ============================================================
st.markdown('<div class="section-label">Simulare predicție umplere</div>', unsafe_allow_html=True)
pc1, pc2 = st.columns([3, 1])
with pc1:
    ora_input = st.slider("Ora simulată", 0, 23, ora_curenta, label_visibility="collapsed")
with pc2:
    predictie = model.predict(pd.DataFrame({"ora": [ora_input]}))[0]
    st.markdown(f"""<div class="stat"><div class="num">{predictie:.0f}<small>%</small></div>
    <div class="lbl">Estimare la ora {ora_input}:00</div></div>""", unsafe_allow_html=True)

st.caption("Model liniar preliminar, calibrat pe date generate aleator (Ziua 2) — recalibrare necesară pe date reale de senzori.")