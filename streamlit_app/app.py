# app.py
import streamlit as st
from utils import load_data, load_mongo_data
from map_pages import page_price_area_map_selectable
from analysis import (
    page_stl_spectrogram,
    page_outlier_anomaly,
    page_meteorology_correlation,
    page_mongo_dashboard
)
from snow_drift import page_snow_drift
from sarimax_page import page_sarimax
from viz import safe_set_page_config

# Set layout & theme quickly (helper)
safe_set_page_config()

def page_home():
    st.title("IND320 — Dashboard (Parts 1–4)")
    st.markdown("Project work for Assignment 4 — integrated dashboard.")
    st.markdown("""
    **Parts included**
    - Part 1: Local weather CSV (Open-Meteo subset)  
    - Part 2: Elhub production data (MongoDB)  
    - Part 3: Analysis — STL, Spectrogram, Outliers, Sliding Correlation  
    - Part 4: Map (Price areas) + Snow drift + Forecasting (SARIMAX)
    """)

def page_table(df):
    st.header("Preview — Weather data (first rows)")
    if df is None or df.empty:
        st.warning("No weather data loaded.")
        return
    st.dataframe(df.head(200))

def main():
    safe_set_page_config()  # again in case
    DATA_PATH = "data/bergen_2021_era5.csv"
    df_weather = load_data(DATA_PATH)

    # Try load production data from Mongo — cached in utils
    df_elhub = None
    try:
        mongo_password = st.secrets["mongo"]["password"]
        df_elhub = load_mongo_data(mongo_password)
    except Exception:
        # We don't crash the app — page functions will handle empty/missing data
        df_elhub = None

    st.sidebar.title("Navigation — Assignment 4")
    # Grouped sections in menu (logical order)
    pages = {
        "Home": page_home,
        "Data (Weather)": lambda: page_table(df_weather),
        "Exploratory Plots": lambda: page_stl_spectrogram(df_elhub),
        "Anomaly Detection": lambda: page_outlier_anomaly(df_weather),
        "Price Area Map": lambda: page_price_area_map_selectable(df_elhub),
        "Meteo ↔ Production Corr": lambda: page_meteorology_correlation(df_weather, df_elhub),
        "Snow drift": lambda: page_snow_drift(df_weather, df_elhub),
        "Forecasting (SARIMAX)": lambda: page_sarimax(df_elhub, df_weather),
        "Mongo Dashboard": lambda: page_mongo_dashboard(df_elhub),
        "About": lambda: st.info("See repository: https://github.com/Srivatsav1298/WeatherAndEnergyInsights")
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("Assignment 4 — IND320")

    pages[choice]()

if __name__ == "__main__":
    main()
