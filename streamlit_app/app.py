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

# ----------------------------------------------------
# Missing About page (ADDED)
# ----------------------------------------------------
def page_about():
    st.title("ℹ️ About This Dashboard")
    st.markdown("""
    This dashboard was developed as part of **IND320** at NMBU.

    It unifies:
    - Weather data (ERA5/Open-Meteo)
    - Electricity production & consumption (Elhub)
    - Anomaly detection
    - Spectrograms & STL decomposition
    - Geographic price-area maps
    - Snow drift modelling
    - SARIMAX forecasting

    All components are interactive and modular.
    """)

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------
def page_home():
    st.title("🌦️⚡ IND320 — Weather and Energy Dashboard")
    st.markdown("""
    Welcome to the **Integrated Dashboard** combining weather data, electricity production & consumption, geographic analysis, anomaly detection, correlations, and forecasting.

    ---
    ## 📘 Project Overview (Parts 1–4)

    ### Part 1 — Weather Data
    - ERA5/Open-Meteo hourly dataset  
    - Cleaned, parsed, visualised  
    - Used later for anomalies, snow drift, and forecasting inputs  

    ### Part 2 — Elhub Data Integration (Production & Consumption)
    - API ingestion  
    - Stored in Cassandra + MongoDB  
    - Production & consumption (2021–2024)

    ### Part 3 — Time Series Analysis & Anomalies
    - STL decomposition  
    - Spectrogram frequency analysis  
    - Temperature SPC outlier detection  
    - Precipitation anomalies (LOF)  
    - Sliding-window correlation

    ### Part 4 — Geographic Integration & Forecasting
    - GeoJSON price area map  
    - Snow drift estimation  
    - Wind rose  
    - Full SARIMAX forecasting interface  
    ---
    Use the **sidebar navigation** to explore.
    """)

# ----------------------------------------------------
# DATA LOADERS
# ----------------------------------------------------
@st.cache_data
def get_weather_data():
    return load_data("data/bergen_2021_era5.csv")

@st.cache_data
def get_elhub_data():
    try:
        mongo_password = st.secrets["mongo"]["password"]
        return load_mongo_data(mongo_password)
    except Exception:
        return None

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    safe_set_page_config()

    st.sidebar.title("Navigation")

    selection = st.sidebar.radio("Go to", [
        "1. Home",
        "2. Price Area Map",
        "3. Exploratory Plots (STL/Spec)",
        "4. Weather Data",
        "5. Anomalies (SPC/LOF)",
        "6. Correlations (Meteo/Prod)",
        "7. Snow Drift",
        "8. Forecasting (SARIMAX)",
        "9. Mongo Dashboard",
        "10. About"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Assignment 4 — IND320")

    # ------------------ ROUTING ------------------

    if selection == "1. Home":
        page_home()

    elif selection == "2. Price Area Map":
        df = get_elhub_data()
        page_price_area_map_selectable(df)

    elif selection == "3. Exploratory Plots (STL/Spec)":
        df = get_elhub_data()
        page_stl_spectrogram(df)

    elif selection == "4. Weather Data":
        df = get_weather_data()
        st.header("Weather Data")
        if df is None or df.empty:
            st.error("Weather dataset is empty.")
        else:
            st.dataframe(df)

    elif selection == "5. Anomalies (SPC/LOF)":
        df = get_weather_data()
        page_outlier_anomaly(df)

    elif selection == "6. Correlations (Meteo/Prod)":
        df_weather = get_weather_data()
        df_elhub = get_elhub_data()
        page_meteorology_correlation(df_weather, df_elhub)

    elif selection == "7. Snow Drift":
        df_weather = get_weather_data()
        df_elhub = get_elhub_data()
        page_snow_drift(df_weather, df_elhub)

    elif selection == "8. Forecasting (SARIMAX)":
        df_weather = get_weather_data()
        df_elhub = get_elhub_data()
        page_sarimax(df_elhub, df_weather)

    elif selection == "9. Mongo Dashboard":
        df = get_elhub_data()
        page_mongo_dashboard(df)

    elif selection == "10. About":
        page_about()

# ----------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------
if __name__ == "__main__":
    main()
