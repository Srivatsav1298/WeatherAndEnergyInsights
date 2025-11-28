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

def page_about():
    st.title("ℹ️ About This Dashboard")
    st.markdown("""
    This dashboard is the result of completing **Parts 1–4 of the IND320 course project**, 
    combining data engineering, API integration, time-series analysis, anomaly detection, 
    geospatial visualisation, and forecasting.

    ---
    ##**Technologies Used**
    - **Python 3.9**  
    - **Streamlit** for interactive visualisation  
    - **Plotly & Mapbox** for dynamic plotting  
    - **Pandas / NumPy** for data handling  
    - **Statsmodels** for SARIMAX forecasting  
    - **SciPy** for STL & DCT transforms  
    - **MongoDB Atlas** for production data storage  
    - **Apache Cassandra + Spark** for large-scale ingestion  
    - **Open-Meteo** and **Elhub API** for raw data sources  

    ---
    ##**Data Sources**
    - **Weather:** ERA5/Open-Meteo hourly data  
    - **Energy Production:** Elhub `PRODUCTION_PER_GROUP_MBA_HOUR`  
    - **Energy Consumption:** Elhub `CONSUMPTION_PER_GROUP_MBA_HOUR`  
    - **Price Areas:** GeoJSON downloaded from NVE Temakart  

    ---
    ## Project Structure
    - `/analysis.py` : STL, spectrogram, anomalies, correlation  
    - `/map_pages.py` : interactive GeoJSON area map  
    - `/snow_drift.py` : snow drift + wind rose  
    - `/sarimax_page.py` : forecasting interface  
    - `/utils.py` : caching, Mongo connection, loader  
    - `/app.py` : main control, navigation  

    ---
    ## Author
    **Srivatsav Saravanan**  
    IND320 — Norwegian University of Life Sciences 
    GitHub: <https://github.com/Srivatsav1298/WeatherAndEnergyInsights>

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
        "About": lambda: page_about()
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("Assignment 4 — IND320")

    pages[choice]()

if __name__ == "__main__":
    main()
