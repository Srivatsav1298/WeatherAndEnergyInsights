# analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import dct, idct
from datetime import datetime

# Mongo dashboard simple viewer
def page_mongo_dashboard(df_elhub):
    st.header("Mongo: Elhub production viewer")
    if df_elhub is None or df_elhub.empty:
        st.warning("No data loaded from Mongo.")
        return
    st.write("Show sample and basic plots")
    st.dataframe(df_elhub.head(200))
    # quick time-series for a choosable area + productionGroup
    price_areas = sorted(df_elhub["priceArea"].dropna().unique())
    pa = st.selectbox("Price area", price_areas, index=0)
    groups = sorted(df_elhub["productionGroup"].dropna().unique())
    pg = st.selectbox("Production group", groups, index=0)
    df_sel = df_elhub[(df_elhub["priceArea"]==pa) & (df_elhub["productionGroup"]==pg)]
    if df_sel.empty:
        st.warning("No records for selection")
        return
    df_ts = df_sel.set_index("startTime").sort_index().resample("D").sum()
    st.line_chart(df_ts["quantityKwh"])

# STL & Spectrogram
def page_stl_spectrogram(df_elhub):
    st.header("STL decomposition & spectrogram")
    if df_elhub is None or df_elhub.empty:
        st.warning("No Elhub data loaded")
        return
    pa = st.selectbox("Price area (STL)", sorted(df_elhub["priceArea"].unique()))
    pg = st.selectbox("Production group (STL)", sorted(df_elhub["productionGroup"].unique()))
    df_sel = df_elhub[(df_elhub["priceArea"]==pa) & (df_elhub["productionGroup"]==pg)].copy()
    if df_sel.empty:
        st.warning("No data for selection")
        return
    df_daily = df_sel.groupby(df_sel["startTime"].dt.date)["quantityKwh"].sum().reset_index()
    df_daily["startTime"] = pd.to_datetime(df_daily["startTime"])
    if len(df_daily) < 10:
        st.warning("Too few daily points for STL")
        return
    period = st.slider("STL seasonal period (days)", 7, 90, 30)
    stl = STL(df_daily["quantityKwh"], period=period, robust=True).fit()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_daily["startTime"], y=df_daily["quantityKwh"], name="Observed"))
    fig.add_trace(go.Scatter(x=df_daily["startTime"], y=stl.trend, name="Trend"))
    fig.add_trace(go.Scatter(x=df_daily["startTime"], y=stl.seasonal, name="Seasonal"))
    fig.add_trace(go.Scatter(x=df_daily["startTime"], y=stl.resid, name="Residual"))
    fig.update_layout(title=f"STL — {pg} ({pa})", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    # spectrogram
    win = st.slider("Spectrogram window (nperseg)", 15, 90, 30)
    f, t, Sxx = spectrogram(df_daily["quantityKwh"].values, fs=1, nperseg=win)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    spec_fig = go.Figure(data=go.Heatmap(z=Sxx_db, x=t, y=f, colorscale="Viridis"))
    spec_fig.update_layout(title="Spectrogram (daily series)", template="plotly_white")
    st.plotly_chart(spec_fig, use_container_width=True)

def page_outlier_anomaly(df_weather):
    st.header("⚡ Outlier & Anomaly Detection (Part 3B)")
    if df_weather is None or df_weather.empty:
        st.error("❌ Weather dataset is empty.")
        st.stop()

    tab1, tab2 = st.tabs(["🌡️ Temperature SPC", "🌧️ Precipitation Anomalies (LOF)"])

    # --- TAB 1 ---
    with tab1:
        st.subheader("Temperature Outlier Detection (SPC – DCT Method)")
        freq_cutoff = st.slider("DCT Frequency Cutoff", 5, 200, 50)
        n_std = st.slider("SPC Sigma Threshold (σ)", 1.0, 5.0, 2.0)
        if "temperature_2m" in df_weather.columns:  # Updated column name here
            df = df_weather.copy()
            df["temperature_2m"] = df["temperature_2m"].interpolate().fillna(method='bfill')  # Updated column name here
            temp = df["temperature_2m"].values
            coeff = dct(temp, norm='ortho')
            coeff[:int(freq_cutoff)] = 0
            satv = idct(coeff, norm='ortho')
            mean, std = np.mean(satv), np.std(satv)
            upper, lower = mean + n_std * std, mean - n_std * std
            df["outlier"] = (satv > upper) | (satv < lower)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["temperature_2m"],
                                     mode="lines", name="Temperature (°C)",
                                     line=dict(color="royalblue")))
            fig.add_trace(go.Scatter(x=df.index, y=[upper]*len(df),
                                     mode="lines", name="Upper Bound (+σ)",
                                     line=dict(dash="dash", color="orange")))
            fig.add_trace(go.Scatter(x=df.index, y=[lower]*len(df),
                                     mode="lines", name="Lower Bound (-σ)",
                                     line=dict(dash="dash", color="orange")))
            fig.add_trace(go.Scatter(x=df.index[df["outlier"]],
                                     y=df["temperature_2m"][df["outlier"]],
                                     mode="markers", name="Outliers",
                                     marker=dict(color="red", size=9, symbol="diamond")))
            fig.update_layout(title="Temperature SPC Outlier Detection (DCT-based)",
                              xaxis_title="Date", yaxis_title="Temperature (°C)",
                              template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Detected {df['outlier'].sum()} outliers from {len(df)} records.")
        else:
            st.error("⚠️ Missing column: 'temperature_2m'")  # Updated column name here

    # --- TAB 2 ---
    with tab2:
        st.subheader("Precipitation Anomaly Detection (LOF)")
        contamination = st.slider("LOF Contamination Ratio", 0.001, 0.05, 0.01)
        if "precipitation" in df_weather.columns:  # Updated column name here
            df = df_weather.copy()
            df["precipitation"] = df["precipitation"].fillna(0)  # Updated column name here
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            df["anomaly"] = lof.fit_predict(df[["precipitation"]]) == -1  # Updated column name here
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["precipitation"],
                                     mode="lines", name="Normal Observations",
                                     line=dict(color="royalblue")))
            fig.add_trace(go.Scatter(x=df.index[df["anomaly"]],
                                     y=df["precipitation"][df["anomaly"]],
                                     mode="markers", name="Anomalies (LOF)",
                                     marker=dict(color="red", size=9, symbol="diamond")))
            fig.update_layout(title="Precipitation Anomalies — Local Outlier Factor (LOF)",
                              xaxis_title="Date", yaxis_title="Precipitation (mm)",
                              template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Detected {df['anomaly'].sum()} anomalies from {len(df)} observations.")
        else:
            st.error("⚠️ Missing column: 'precipitation'")  # Updated column name here


# ---------------------------------------------------------------
#  NEW PAGE: Meteorology <-> Production correlation (YOUR FUNCTION)
# ---------------------------------------------------------------
def page_meteorology_correlation(df_weather, df_elhub):
    """
    Interactive sliding-window correlation between a meteorological variable (df_weather)
    and a production timeseries from df_elhub.
    """
    st.header("🔁 Meteorology ↔ Energy Production — Sliding-window correlation")

    # ----------------------------------------------------------
    # Basic checks
    # ----------------------------------------------------------
    if df_weather is None or df_weather.empty:
        st.error("Weather dataset not loaded.")
        return

    if df_elhub is None or df_elhub.empty:
        st.error("Elhub production data not loaded.")
        return

    # ----------------------------------------------------------
    # ROOT CAUSE DIAGNOSTIC BLOCK (NEW)
    # ----------------------------------------------------------
    try:
        w_min = df_weather.index.min()
        w_max = df_weather.index.max()
    except Exception:
        st.error("Weather index is not datetime. Please ensure the CSV is loaded with parse_dates.")
        return

    try:
        e_min = df_elhub["startTime"].min()
        e_max = df_elhub["startTime"].max()
    except Exception:
        st.error("Elhub 'startTime' column is missing or not datetime.")
        return

    st.markdown("### 🕵️ Diagnostics — Date Ranges")
    st.write("- **Weather min/max:**", w_min, "→", w_max)
    st.write("- **Elhub min/max:**", e_min, "→", e_max)

    # ----------------------------------------------------------
    # Compute allowed range
    # ----------------------------------------------------------
    min_dt = max(w_min.date(), e_min.date())
    max_dt = min(w_max.date(), e_max.date())

    # ----------------------------------------------------------
    # NEW: Check for inverted range
    # ----------------------------------------------------------
    if min_dt > max_dt:
        st.error(
            f"""
            ❌ **No overlapping date range found.**
            
            This caused the Streamlit crash (`min_value > max_value`).  
            
            **Weather range:** {w_min.date()} → {w_max.date()}  
            **Elhub range:** {e_min.date()} → {e_max.date()}  

            ✔️ Fix your input datasets so that their date ranges overlap.
            """
        )
        return

    # ----------------------------------------------------------
    # Variable selection
    # ----------------------------------------------------------
    weather_vars = [c for c in df_weather.columns if pd.api.types.is_numeric_dtype(df_weather[c])]
    if not weather_vars:
        st.error("No numeric weather variables available.")
        return

    wvar = st.selectbox("Meteorological variable (X)", weather_vars, index=0)

    pa_options = sorted(df_elhub["priceArea"].dropna().unique())
    pa_default = st.session_state.get("selected_price_area", None)
    pa = st.selectbox("Price area (Y)", pa_options,
                      index=pa_options.index(pa_default) if pa_default in pa_options else 0)

    pgroups = sorted(df_elhub["productionGroup"].dropna().unique())
    pg = st.selectbox("Production group (Y)", pgroups, index=0)

    # ----------------------------------------------------------
    # User date inputs (now safe)
    # ----------------------------------------------------------
    start_dt = st.date_input("Start date", min_dt, min_value=min_dt, max_value=max_dt)
    end_dt = st.date_input("End date", max_dt, min_value=min_dt, max_value=max_dt)

    if start_dt > end_dt:
        st.error("Start date must be before end date.")
        return

    # ----------------------------------------------------------
    # Numeric configuration
    # ----------------------------------------------------------
    agg_freq = st.radio("Aggregate frequency", ["H", "D"], index=1)
    window_days = st.slider("Window length (days)", 1, 90, 30)
    max_lag_hours = st.slider("Max lag (hours)", 0, 168, 72)
    step_lag = st.slider("Lag step (hours)", 1, 24, 6)

    # ----------------------------------------------------------
    # Prepare weather data
    # ----------------------------------------------------------
    w = df_weather[[wvar]].copy()
    if not pd.api.types.is_datetime64_any_dtype(w.index):
        st.error("Weather index is not datetime. Reload CSV with parsed dates.")
        return

    w = w.loc[(w.index.date >= start_dt) & (w.index.date <= end_dt)]
    if w.empty:
        st.warning("No weather data in selected interval.")
        return

    # ----------------------------------------------------------
    # Prepare Elhub data
    # ----------------------------------------------------------
    dfp = df_elhub[(df_elhub["priceArea"] == pa) & (df_elhub["productionGroup"] == pg)].copy()
    dfp = dfp.loc[(dfp["startTime"].dt.date >= start_dt) & (dfp["startTime"].dt.date <= end_dt)]

    if dfp.empty:
        st.warning("No production data for this selection.")
        return

    dfp = dfp.set_index("startTime").sort_index()

    # ----------------------------------------------------------
    # Resample
    # ----------------------------------------------------------
    if agg_freq == "D":
        prod = dfp["quantityKwh"].resample("D").sum()
        w_res = w.resample("D").mean()
    else:
        prod = dfp["quantityKwh"].resample("H").sum()
        w_res = w.resample("H").mean()

    joint = pd.DataFrame({"X": w_res[wvar], "Y": prod}).dropna()
    if joint.empty:
        st.warning("No overlapping data after resampling.")
        return

    st.write(f"Prepared {len(joint)} rows after resampling to frequency '{agg_freq}'.")

    # ----------------------------------------------------------
    # Lag correlation
    # ----------------------------------------------------------
    window_len = max(1, int(window_days * (24 if agg_freq == "H" else 1)))
    lags = list(range(-max_lag_hours, max_lag_hours + 1, step_lag))

    if agg_freq == "D":
        lags_unique = sorted(set(int(round(l / 24)) for l in lags))
    else:
        lags_unique = lags

    corr_df = pd.DataFrame(index=joint.index)

    for lag in lags_unique:
        X_shifted = joint["X"].shift(-lag)
        rolling_corr = X_shifted.rolling(
            window=window_len,
            min_periods=max(3, int(window_len * 0.5))
        ).corr(joint["Y"])
        corr_df[f"lag_{lag}"] = rolling_corr

    # ----------------------------------------------------------
    # Heatmap
    # ----------------------------------------------------------
    heat_data = corr_df.T
    if heat_data.shape[1] > 1200:
        step = int(np.ceil(heat_data.shape[1] / 1200))
        heat_data = heat_data.iloc[:, ::step]

    fig = go.Figure(data=go.Heatmap(
        z=heat_data.values,
        x=[d.strftime("%Y-%m-%d") for d in heat_data.columns],
        y=heat_data.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        colorbar=dict(title="corr")
    ))
    fig.update_layout(
        title=f"Rolling correlation X={wvar} vs Y={pa}/{pg} — window={window_days}d, agg={agg_freq}",
        xaxis_nticks=10,
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------
    # Example lag view
    # ----------------------------------------------------------
    st.markdown("### Example: time series + one lagged rolling correlation")
    example_lag = st.selectbox(
        "Choose lag to inspect (hours; negative ⇒ X lagging)",
        lags,
        index=len(lags) // 2
    )
    example_col = f"lag_{example_lag if agg_freq=='H' else int(round(example_lag/24))}"

    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=joint.index, y=joint["X"], mode="lines", name=f"X={wvar}"))
    ts_fig.add_trace(go.Scatter(x=joint.index, y=joint["Y"], mode="lines",
                                name=f"Y={pa}/{pg}", yaxis="y2"))
    ts_fig.update_layout(
        title=f"X & Y time series (agg={agg_freq})",
        yaxis2=dict(overlaying="y", side="right")
    )
    st.plotly_chart(ts_fig, use_container_width=True)

    latest_corr = corr_df[example_col].iloc[-1] if example_col in corr_df.columns else None
    st.metric("Latest rolling correlation", f"{latest_corr:.3f}" if pd.notnull(latest_corr) else "n/a")
