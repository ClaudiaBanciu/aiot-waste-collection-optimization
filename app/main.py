import sys, os
import streamlit as st

# Setăm calea principală pentru ca Python să vadă folderele noastre
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Importăm componentele noastre modulare
from app.components.data_manager import incarca_date
from app.components.ui_elements import (
    render_sidebar,
    render_quick_stats,
    render_map,
    render_dataframe,
    render_distance_comparison,
    render_predictions
)

# Configurare generală a paginii
st.set_page_config(page_title="Gestiune deșeuri - Sibiu", layout="wide")

def main():
    # 1. Încărcarea datelor backend
    df = incarca_date()
    
    st.title("🚛 Gestiune deșeuri - Sibiu")

    # 2. Afișarea filtrelor și preluarea selecțiilor utilizatorului
    ruta_selectata, masini_ruta, df_ruta, df_filtrat = render_sidebar(df)

    # 3. Construirea interfeței pas cu pas
    render_quick_stats(df_filtrat, masini_ruta)
    
    render_map(df_filtrat, masini_ruta)
    
    render_dataframe(df_filtrat)
    
    render_distance_comparison(df_ruta, masini_ruta, ruta_selectata)
    
    render_predictions(df_ruta, ruta_selectata)

if __name__ == "__main__":
    main()