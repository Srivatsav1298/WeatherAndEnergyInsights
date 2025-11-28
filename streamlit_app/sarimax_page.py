# sarimax_page.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

def page_sarimax(df_elhub, df_weather):
    st.header("SARIMAX forecasting (production)")

    if df_elhub is None or df_elhub.empty:
        st.warning("Elhub production data missing.")
        return
    # pick a price area + production group and series to forecast
    pa = st.selectbox("Price area (forecast)", sorted(df_elhub["priceArea"].dropna().unique()))
    pg = st.selectbox("Production group (forecast)", sorted(df_elhub["productionGroup"].dropna().unique()))
    dfp = df_elhub[(df_elhub["priceArea"]==pa) & (df_elhub["productionGroup"]==pg)].copy()
    if dfp.empty:
        st.warning("No production records for selection.")
        return
    # set index and resample daily (user option)
    freq = st.selectbox("Resample freq", ["D","H"], index=0)
    dfp = dfp.set_index("startTime").sort_index()
    if freq=="D":
        series = dfp["quantityKwh"].resample("D").sum()
    else:
        series = dfp["quantityKwh"].resample("H").sum()
    series = series.dropna()
    if len(series) < 30:
        st.warning("Too few points for meaningful forecasting. Need >=30.")
        return

    # exogenous variables selection (assignment requires exogenous options)
    exog_available = []
    if df_weather is not None and not df_weather.empty:
        # resample weather to same freq and align on index using mean/sum
        if freq=="D":
            wdf = df_weather.resample("D").mean()
        else:
            wdf = df_weather.resample("H").mean()
        # choose numeric columns
        exog_available = [c for c in wdf.columns if pd.api.types.is_numeric_dtype(wdf[c])]
    else:
        wdf = None

    st.markdown("**Exogenous variables** (optional). You can pick multiple. They will be resampled to model freq.")
    exog_choice = st.multiselect("Exogenous columns", exog_available)

    # choose training horizon and forecast horizon
    train_end = st.date_input("Training end date", value=series.index.max().date())
    forecast_steps = st.number_input("Forecast steps (periods)", min_value=1, max_value=365, value=30)

    # SARIMAX params (kept simple in UI)
    p = st.slider("p (AR)", 0, 3, 1)
    d = st.slider("d (I)", 0, 2, 0)
    q = st.slider("q (MA)", 0, 3, 1)
    P = st.slider("P (seasonal AR)", 0, 2, 0)
    D = st.slider("D (seasonal I)", 0, 1, 0)
    Q = st.slider("Q (seasonal MA)", 0, 2, 0)
    s = st.selectbox("Seasonal period (s)", [0,7,12,24,365], index=1, help="0 = no seasonality")

    # prepare training data
    train_mask = series.index.date <= train_end
    y_train = series.loc[train_mask].copy()
    if y_train.empty:
        st.error("No training data for chosen end date.")
        return

    # prepare exog aligned to y_train if any
    exog_train = None
    if exog_choice and wdf is not None:
        exog_df = wdf[exog_choice].copy()
        exog_df = exog_df.reindex(series.index).interpolate().ffill().bfill()
        exog_train = exog_df.loc[train_mask]
    try:
        st.info("Training SARIMAX — this may take some seconds.")
        order = (p,d,q)
        seasonal_order = (P,D,Q,s) if s>0 else (0,0,0,0)
        model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        st.success("Model trained.")
    except Exception as e:
        st.error(f"SARIMAX training failed: {e}")
        return

    # Forecast
    # prepare exog for forecast horizon if needed
    exog_fore = None
    if exog_choice and wdf is not None:
        exog_df = wdf[exog_choice].reindex(series.index).interpolate().ffill().bfill()
        last_idx = series.index.max()
        # build future index based on freq
        if freq=="D":
            future_index = pd.date_range(start=last_idx + pd.Timedelta(1,unit="D"), periods=forecast_steps, freq="D")
        else:
            future_index = pd.date_range(start=last_idx + pd.Timedelta(1,unit="h"), periods=forecast_steps, freq="H")
        # simple strategy: use last available exog row repeated (realistic approach would fetch forecasted exog)
        last_row = exog_df.iloc[-1:].rename(index={exog_df.index[-1]: future_index[0]})
        # replicate to match length
        exog_fore = pd.DataFrame([last_row.values.flatten()] * forecast_steps, columns=exog_choice, index=future_index)

    try:
        fc = res.get_forecast(steps=forecast_steps, exog=exog_fore)
        mean_fc = fc.predicted_mean
        ci = fc.conf_int()
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed"))
    fig.add_trace(go.Scatter(x=mean_fc.index, y=mean_fc.values, name="Forecast"))
    fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,0], fill=None, mode="lines", name="Lower CI"))
    fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,1], fill='tonexty', mode="lines", name="Upper CI"))
    fig.update_layout(title=f"SARIMAX forecast — {pa} / {pg}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # show model summary collapsible
    with st.expander("Model Summary (text)"):
        st.text(res.summary().as_text())
