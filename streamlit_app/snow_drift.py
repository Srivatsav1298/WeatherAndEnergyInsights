# snow_drift.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, cos, sin
from datetime import datetime

def compute_snow_drift_for_years(df_weather, coord, year_start, year_end):
    """
    Minimal snow drift calculation:
      - Year defined: 1 July YEAR -> 30 June YEAR+1
      - For each year in [year_start, year_end] compute a proxy 'drift index' using wind speed & direction.
    Uses wind speed (m/s) and direction to compute a transport proxy.
    """
    if coord is None:
        raise ValueError("No coordinate specified for snow drift calculation.")
    lat, lon = coord

    results = []
    for y in range(year_start, year_end+1):
        period_start = pd.Timestamp(f"{y}-07-01 00:00:00")
        period_end = pd.Timestamp(f"{y+1}-06-30 23:59:59")
        mask = (df_weather.index >= period_start) & (df_weather.index <= period_end)
        dfp = df_weather.loc[mask].copy()
        if dfp.empty:
            results.append({"year": y, "drift_index": np.nan, "n": 0})
            continue
        # Expect columns windspeed_10m (m/s) and winddirection_10m (deg) or similar names
        # Try common names
        ws_col = None
        wd_col = None
        for c in dfp.columns:
            if "wind" in c and ("speed" in c or "speed_10m" in c or "windspeed" in c):
                ws_col = c; break
        for c in dfp.columns:
            if "wind" in c and ("dir" in c or "direction" in c or "winddirection" in c):
                wd_col = c; break
        if ws_col is None or wd_col is None:
            # fallback: attempt 'windspeed_10m' and 'winddirection_10m'
            ws_col = ws_col or ("windspeed_10m" if "windspeed_10m" in dfp.columns else None)
            wd_col = wd_col or ("winddirection_10m" if "winddirection_10m" in dfp.columns else None)
        if ws_col is None or wd_col is None:
            results.append({"year": y, "drift_index": np.nan, "n": len(dfp)})
            continue
        # compute vector transport magnitude: u = ws * cos(rad(wd)), v = ws * sin(rad(wd))
        wd_rad = np.deg2rad(dfp[wd_col].astype(float).fillna(0).values)
        ws = dfp[ws_col].astype(float).fillna(0).values
        # proxy transport magnitude = mean of ws * |cos(direction toward upstream)| etc.
        transport = np.abs(ws * np.cos(wd_rad)) + np.abs(ws * np.sin(wd_rad))
        drift_index = np.nanmean(transport)
        results.append({"year": y, "drift_index": float(drift_index), "n": int(len(dfp))})
    return pd.DataFrame(results)

def page_snow_drift(df_weather, df_elhub):
    st.header("Snow drift — yearly, using selected centroid")
    # require both datasets present to use selected centroid
    coord = st.session_state.get("selected_coord", None)
    if coord is None:
        st.error("No selected coordinate: go to Price Area Map and pick a price area first.")
        return
    if df_weather is None or df_weather.empty:
        st.error("Weather data not loaded.")
        return

    st.write(f"Using coordinate: {coord}")

    # allow year range (min/max based on weather data)
    min_year = df_weather.index.min().year
    max_year = df_weather.index.max().year
    # because years are 1 July -> 30 June, we allow selection across these years
    start_year = st.number_input("Start year (y): choose starting YEAR for 1 July YEAR", min_value=int(min_year), max_value=int(max_year), value=int(min_year))
    end_year = st.number_input("End year (y)", min_value=int(start_year), max_value=int(max_year), value=int(min_year))
    if start_year > end_year:
        st.error("Start year must be <= end year")
        return

    st.info("Calculating yearly snow drift index (proxy)")
    df_res = compute_snow_drift_for_years(df_weather, coord, int(start_year), int(end_year))
    st.dataframe(df_res)

    # Plot bar chart
    fig = px.bar(df_res, x="year", y="drift_index", title="Yearly Snow Drift Index (proxy)")
    st.plotly_chart(fig, use_container_width=True)

    # Simple wind rose for the most recent chosen year
    yr = int(start_year)
    st.markdown(f"Wind rose for year starting 1 Jul {yr}")
    # build wind rose for first selected year (if data exists)
    try:
        df_w = df_weather.loc[(df_weather.index >= pd.Timestamp(f"{yr}-07-01")) & (df_weather.index <= pd.Timestamp(f"{yr+1}-06-30"))]
        if df_w.empty:
            st.warning("No weather records for that year to build wind rose.")
            return
        # detect columns again
        ws_col, wd_col = None, None
        for c in df_w.columns:
            if "wind" in c and ("speed" in c or "windspeed" in c):
                ws_col = c; break
        for c in df_w.columns:
            if "wind" in c and ("dir" in c or "direction" in c):
                wd_col = c; break
        if ws_col is None or wd_col is None:
            st.warning("Could not find wind speed/direction columns in weather data to create wind rose.")
            return
        # create rose bins
        wd = df_w[wd_col].dropna().astype(float)
        ws = df_w[ws_col].fillna(0).astype(float)
        # create direction bins 0-360 in 16 sectors
        bins = np.linspace(0,360,17)
        df_w["wd_bin"] = pd.cut(wd, bins=bins, include_lowest=True, right=False, labels=False)
        grouped = df_w.groupby("wd_bin")[ws_col].mean().reset_index().rename(columns={ws_col:"mean_ws"})
        # build polar plot
        theta = (grouped["wd_bin"].astype(float) + 0.5) * (360/16)
        fig = px.bar_polar(grouped, r="mean_ws", theta=theta, title=f"Wind rose — Jul {yr} → Jun {yr+1}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute wind rose: {e}")
