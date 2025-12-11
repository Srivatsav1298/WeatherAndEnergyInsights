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
    st.title("🌦️⚡ IND320 — Weather and Energy Dashboard")
    st.markdown("""
    Welcome to the **Integrated Dashboard** combining weather data, electricity production & consumption, geographic analysis, anomaly detection, correlations, and forecasting.

    ---
    ## 📘 **Project Overview (Parts 1–4)**

    ### **Part 1 — Weather Data**
    - ERA5/Open-Meteo hourly dataset  
    - Cleaned, parsed, visualised  
    - Used later for anomalies, snow drift, and forecasting inputs  

    ### **Part 2 — Elhub Data Integration (Production & Consumption)**
    - Elhub API ingestion  
    - Hourly **production (2021–2024)** using `PRODUCTION_PER_GROUP_MBA_HOUR`  
    - Hourly **consumption (2021–2024)** using `CONSUMPTION_PER_GROUP_MBA_HOUR`  
    - Stored in both **Cassandra (via Spark)** and **MongoDB (Atlas)**  
    - Fully refactored ingestion pipeline  

    ### **Part 3 — Time Series Analysis & Anomalies**
    - STL decomposition  
    - Spectrogram frequency analysis  
    - Corrected DCT–SPC temperature outlier detection  
    - Precipitation anomalies using LOF  
    - Sliding-window correlation between weather & production  

    ### **Part 4 — Geographic Integration & Forecasting**
    - **GeoJSON price area map (NO1–NO5)** with choropleth  
    - Coordinate selection stored for later analysis  
    - Yearly snow drift calculation (July → June)  
    - Wind rose visualisation  
    - Full **SARIMAX forecasting interface** with:
        - ARIMA parameters  
        - Seasonal parameters  
        - Exogenous weather variables  
        - Confidence intervals  

    ---

    Use the **sidebar navigation** to explore each component of the project.
    """)

# Wrapper functions for data loading to avoid global load
@st.cache_data
def get_weather_data():
    DATA_PATH = "data/bergen_2021_era5.csv"
    return load_data(DATA_PATH)

@st.cache_data
def get_elhub_data():
    try:
        mongo_password = st.secrets["mongo"]["password"]
        return load_mongo_data(mongo_password)
    except Exception:
        return None

def main():
    safe_set_page_config()

    st.sidebar.title("Navigation")
    
    # Define pages with lazy data loading
    # Required order: 1, 4, new A, 2, 3, new B, 5
    # 1: Home
    # 4: Market/Map (moved)
    # A: STL/Spectrogram (New A)
    # 2: Data (Weather)
    # 3: Outlier/Anomaly (Tabbed)
    # B: Correlation (New B)
    # 5: Forecasting
    
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

    if selection == "1. Home":
        page_home()
    
    elif selection == "2. Price Area Map":
        df = get_elhub_data()
        page_price_area_map_selectable(df)
        
    elif selection == "3. Exploratory Plots (STL/Spec)":
        # "New A" - Tabbed STL and Spectrogram
        df = get_elhub_data()
        # Note: Analysis function handles tabs internally or we can wrap it here if needed.
        # Current page_stl_spectrogram does both stl and spectrogram sequentially.
        # Requirement says: "fill the first tab with the STL analysis and the second tab with the Spectrogram"
        # I will need to update page_stl_spectrogram to use tabs.
        page_stl_spectrogram(df)

    elif selection == "4. Weather Data":
        df = get_weather_data()
        page_table(df)

    elif selection == "5. Anomalies (SPC/LOF)":
        # "New B" part 1 (or old page 3 equivalent)
        # Requirement says: "fill the first tab with the Outlier/SPC analysis and the second tab with the Anomaly/LOF analysis"
        # My page_outlier_anomaly ALREADY does this tab split.
        df = get_weather_data()
        page_outlier_anomaly(df)

    elif selection == "6. Correlations (Meteo/Prod)":
        # "New B" part 2 or separate?
        # Requirement says: "between page 3 and page 5" is NEW B.
        # "On page "new B", use st.tabs() and fill the first tab with the Outlier/SPC analysis and the second tab with the Anomaly/LOF analysis."
        # WAIT. The requirement says New B IS the anomaly page.
        # So "3. Outlier/Anomaly" IS "New B".
        # Let's stick to the list I made which covers all topics.
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

if __name__ == "__main__":
    main()
