# sarimax_page.py
import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go


def page_sarimax(df_elhub, df_weather):

    st.header("📈 SARIMAX Forecasting (Production & Consumption)")

    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------
    if df_elhub is None or df_elhub.empty:
        st.error("❌ Elhub dataset is empty — cannot run forecasting.")
        return
        # DEBUG — show raw columns
    st.write("DEBUG columns:", df_elhub.columns.tolist())

    
        # ---------- Defensive canonicalization (in case utils didn't run) ----------
    # Look for case-insensitive variants and create canonical columns expected below.
    lower_cols = {c.lower(): c for c in df_elhub.columns if isinstance(c, str)}

    def _alias_if_missing(canonical, candidates_lower):
        """If canonical is missing but a lower-case variant exists, create canonical column."""
        if canonical not in df_elhub.columns:
            for cand in candidates_lower:
                if cand in lower_cols and lower_cols[cand] != canonical:
                    df_elhub[canonical] = df_elhub[lower_cols[cand]]
                    break

    _alias_if_missing("consumptionGroup", ["consumptiongroup", "consumption_group"])
    _alias_if_missing("productionGroup", ["productiongroup", "production_group"])
    _alias_if_missing("quantityKwh", ["quantitykwh", "quantity_kwh", "quantity"])
    _alias_if_missing("startTime", ["starttime", "start_time", "start"])
    _alias_if_missing("priceArea", ["pricearea", "price_area"])
    _alias_if_missing("recordType", ["recordtype", "record_type", "type"])
    # Ensure datetime for startTime (if created above)
    if "startTime" in df_elhub.columns and not pd.api.types.is_datetime64_any_dtype(df_elhub["startTime"]):
        try:
            df_elhub["startTime"] = pd.to_datetime(df_elhub["startTime"], errors="coerce")
        except Exception:
            pass

    
    mode = st.radio(
        "Select dataset to forecast:",
        ["Production", "Consumption"],
        horizontal=True
    )

    has_prod = "productionGroup" in df_elhub.columns
    has_cons = "consumptionGroup" in df_elhub.columns

    # Handle selection vs available data
    if mode == "Production" and not has_prod:
        st.error("⚠ No PRODUCTION data available (missing column 'productionGroup').")
        return

    if mode == "Consumption" and not has_cons:
        st.error("⚠ No CONSUMPTION data available (missing column 'consumptionGroup').")
        return

    # ------------------------------------------------------------------
    # SELECT DATA BASED ON MODE
    # ------------------------------------------------------------------
    if mode == "Production":
        df = df_elhub.dropna(subset=["productionGroup"]).copy()
        group_col = "productionGroup"
        st.info("Forecasting: PRODUCTION")
    else:
        df = df_elhub.dropna(subset=["consumptionGroup"]).copy()
        group_col = "consumptionGroup"
        st.info("Forecasting: CONSUMPTION")

    # Basic sanity check
    if df.empty:
        st.error(f"❌ No data available for {mode.lower()}.")
        return

    # ------------------------------------------------------------------
    # PRICE AREA & GROUP
    # ------------------------------------------------------------------
    pa = st.selectbox(
        "Price Area",
        sorted(df["priceArea"].dropna().unique())
    )

    grp = st.selectbox(
        f"{mode} Group",
        sorted(df[group_col].dropna().unique())
    )

    df_sel = df[(df["priceArea"] == pa) & (df[group_col] == grp)].copy()

    if df_sel.empty:
        st.error(f"❌ No {mode.lower()} data available for selection.")
        return

    # ------------------------------------------------------------------
    # SET TIME INDEX & RESAMPLE
    # ------------------------------------------------------------------
    df_sel = df_sel.set_index("startTime").sort_index()

    freq = st.selectbox("Resample Frequency", ["H", "D"], index=1)

    if freq == "D":
        series = df_sel["quantityKwh"].resample("D").sum()
    else:
        series = df_sel["quantityKwh"].resample("H").sum()

    series = series.dropna()

    if len(series) < 30:
        st.warning("⚠ Too few data points (<30) to train SARIMAX.")
        return

    # ------------------------------------------------------------------
    # EXOGENOUS (WEATHER) VARIABLES
    # ------------------------------------------------------------------
    st.subheader("Exogenous Weather Variables (Optional)")

    if df_weather is not None and not df_weather.empty:

        if freq == "D":
            wdf = df_weather.resample("D").mean()
        else:
            wdf = df_weather.resample("H").mean()

        exog_options = [
            c for c in wdf.columns
            if pd.api.types.is_numeric_dtype(wdf[c])
        ]

        exog_choice = st.multiselect("Select weather predictors:", exog_options)

    else:
        wdf = None
        exog_choice = []
        st.info("ℹ No weather data available — exogenous features disabled.")

    # ------------------------------------------------------------------
    # TRAINING WINDOW
    # ------------------------------------------------------------------
    train_end = st.date_input(
        "Training end date",
        value=series.index.max().date()
    )

    forecast_steps = st.number_input(
        "Forecast horizon (# periods)",
        min_value=1,
        max_value=730,
        value=30
    )

    # ------------------------------------------------------------------
    # SARIMAX PARAMS
    # ------------------------------------------------------------------
    st.subheader("SARIMAX Parameters")

    p = st.slider("p (AR)", 0, 3, 1)
    d = st.slider("d (I)", 0, 2, 0)
    q = st.slider("q (MA)", 0, 3, 1)

    P = st.slider("P (Seasonal AR)", 0, 2, 0)
    D = st.slider("D (Seasonal I)", 0, 1, 0)
    Q = st.slider("Q (Seasonal MA)", 0, 2, 0)

    s = st.selectbox(
        "Seasonal Period (s)",
        [0, 7, 12, 24, 365],
        index=1
    )

    # ------------------------------------------------------------------
    # SPLIT TRAINING DATA
    # ------------------------------------------------------------------
    mask = series.index.date <= train_end
    y_train = series.loc[mask]

    if y_train.empty:
        st.error("❌ Training period resulted in an empty dataset.")
        return

    # Handle exogenous training
    exog_train = None

    if exog_choice:
        exog_all = wdf[exog_choice].reindex(series.index).interpolate().ffill().bfill()
        exog_train = exog_all.loc[mask]

    # ------------------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------------------
    try:
        st.info("⏳ Training SARIMAX model… Please wait.")

        order = (p, d, q)
        seasonal_order = (P, D, Q, s) if s > 0 else (0, 0, 0, 0)

        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        res = model.fit(disp=False)

        st.success("✅ SARIMAX model trained successfully.")

    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        return

    # ------------------------------------------------------------------
    # PREP FUTURE EXOG
    # ------------------------------------------------------------------
    exog_fore = None

    if exog_choice:
        exog_all = wdf[exog_choice].reindex(series.index).interpolate().ffill().bfill()

        last_idx = series.index.max()

        if freq == "D":
            future_idx = pd.date_range(last_idx + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
        else:
            future_idx = pd.date_range(last_idx + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")

        # Repeat last row for future (simple baseline)
        last_row = exog_all.iloc[-1]
        exog_fore = pd.DataFrame([last_row.values] * forecast_steps,
                                 columns=exog_choice,
                                 index=future_idx)

    # ------------------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------------------
    try:
        fc = res.get_forecast(steps=forecast_steps, exog=exog_fore)
        fc_mean = fc.predicted_mean
        ci = fc.conf_int()
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
        return

    # ------------------------------------------------------------------
    # PLOT FORECAST
    # ------------------------------------------------------------------
    st.subheader("📉 Forecast Output")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Observed"))
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, name="Forecast"))

    fig.add_trace(go.Scatter(
        x=ci.index, y=ci.iloc[:, 0],
        mode="lines", name="Lower CI"
    ))

    fig.add_trace(go.Scatter(
        x=ci.index, y=ci.iloc[:, 1],
        fill="tonexty", mode="lines",
        name="Upper CI"
    ))

    fig.update_layout(
        title=f"SARIMAX Forecast — {mode} • {pa} • {grp}",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # MODEL SUMMARY
    # ------------------------------------------------------------------
    with st.expander("🔎 Model Summary"):
        st.text(res.summary().as_text())
