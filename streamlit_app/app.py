import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import dct, idct
import numpy as np
import plotly.graph_objects as go


# --------------- CSV loader (Part 1) ----------------
def custom_date_parser(x):
    return datetime.strptime(x, "%Y-%m-%dT%H:%M")

@st.cache_data
def load_data(path: str):
    try:
        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=custom_date_parser
        )
        if df.index.isnull().any():
            st.warning("Warning: Some date values could not be parsed correctly. Check the index.")
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


# --------------- Helper / Header ----------------
def show_header():
    st.title("IND320 — Dashboard (Part 1 + 2 + 3)")
    st.write("Use the sidebar to navigate between pages.")


# --------------- Page 1: Home ----------------
def page_home():
    show_header()
    st.markdown("## Welcome")
    st.write("This dashboard demonstrates the progression of IND320 Parts 1, 2 and 3.")
    st.markdown("""
    **Part 1:** CSV loaded locally — Data summary and plots  
    **Part 2:** MongoDB connection — Interactive Elhub dashboard  
    **Part 3:** Open-Meteo API — STL, Spectrogram & Outlier Detection
    """)


# --------------- Page 2: Data Table ----------------
def page_table(df):
    st.header("📈 Variables summary for the first month (2020-01)")
    first_month = df[df.index.month == 1]
    summary_data = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = first_month[col].dropna()
            summary_data.append({
                "Variable": col,
                "First month sparkline": series.values,
                "Count (first month)": series.count(),
                "Mean (first month)": round(series.mean(), 4),
                "Min (first month)": round(series.min(), 4),
                "Max (first month)": round(series.max(), 4),
                "Std Dev": round(series.std(), 4),
                "Range": round(series.max() - series.min(), 4)
            })
    summary_df = pd.DataFrame(summary_data)
    column_config = {
        "Variable": st.column_config.TextColumn("Variable"),
        "First month sparkline": st.column_config.LineChartColumn("First month sparkline", y_min=None, y_max=None),
        "Count (first month)": st.column_config.NumberColumn("Count (first month)"),
        "Mean (first month)": st.column_config.NumberColumn("Mean (first month)"),
        "Min (first month)": st.column_config.NumberColumn("Min (first month)"),
        "Max (first month)": st.column_config.NumberColumn("Max (first month)"),
        "Std Dev": st.column_config.NumberColumn("Std Dev"),
        "Range": st.column_config.NumberColumn("Range")
    }
    st.dataframe(summary_df, column_config=column_config, use_container_width=True, hide_index=True)
    st.caption("Each sparkline uses its own y-axis scale for better variation visibility.")


# --------------- Page 3: Plots ----------------
def page_plots(df):
    st.header("Interactive plots")
    st.write("Choose a column (or All), and a month to visualize.")
    if df.index.empty:
        st.error("Index appears empty. Check CSV and index parsing.")
        return
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("Index is not datetime. Please check your data.")
        return
    df["month_name"] = df.index.strftime("%B")
    months = df["month_name"].unique().tolist()
    month_choice = st.select_slider("Select Month", options=months, value=months[0])
    df_filtered = df[df["month_name"] == month_choice]
    st.markdown(f"### Showing data for **{month_choice}** ({len(df_filtered)} rows)")
    tab1, tab2 = st.tabs(["📊 Single/All Columns", "🪞 Dual-Axis Plot"])
    with tab1:
        column_options = ["All"] + list(df.columns)
        chosen = st.selectbox("Choose a single column or All", column_options, index=0)
        if chosen == "All":
            df_num = df_filtered.select_dtypes(include='number')
            if df_num.shape[1] == 0:
                st.warning("No numeric columns to plot for 'All'.")
            else:
                df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
                st.line_chart(df_norm)
        else:
            series = pd.to_numeric(df_filtered[chosen], errors='coerce')
            st.line_chart(series)
    with tab2:
        numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Left Y-axis variable", numeric_cols, index=0)
            col2 = st.selectbox("Right Y-axis variable", numeric_cols, index=1)
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_filtered.index, df_filtered[col1], 'b-', label=col1)
            ax2 = ax1.twinx()
            ax2.plot(df_filtered.index, df_filtered[col2], 'g-', label=col2)
            plt.title(f"Dual-axis Plot: {col1} vs {col2}")
            st.pyplot(fig)


# --------------- Page 4: Mongo Dashboard (Part 2) ----------------
def page_mongo_dashboard():
    st.header("Production Dashboard — MongoDB (Part 2)")
    st.write("Visualizing Elhub 2021 production data stored in MongoDB.")
    mongo_password = st.secrets["mongo"]["password"]
    if not mongo_password:
        st.error("MongoDB password is missing. Please set the environment variable MONGO_PASSWORD.")
        return
    try:
        username = "abbuvatsav"
        mongo_uri = f"mongodb+srv://{username}:{mongo_password}@cluster0.klxry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(mongo_uri)
        collection = client["Cluster0"]["elhub_production_data"]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return
    if df.empty:
        st.warning("No data found in MongoDB.")
        return
    df["startTime"] = pd.to_datetime(df["startTime"])
    left, right = st.columns(2)
    with left:
        st.subheader("Total Production (Pie Chart)")
        price_areas = sorted(df["priceArea"].dropna().unique())
        pa = st.radio("Price Area", price_areas, index=0)
        df_pa = df[df["priceArea"] == pa]
        agg = df_pa.groupby("productionGroup")["quantityKwh"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        wedges, _, _ = ax1.pie(agg, autopct="%1.1f%%", startangle=140)
        ax1.legend(wedges, agg.index, title="Production Group", loc="center left", bbox_to_anchor=(1, 0.5))
        st.pyplot(fig1)
    with right:
        st.subheader("Monthly Production (Line Chart)")
        groups = sorted(df["productionGroup"].dropna().unique())
        chosen = st.multiselect("Production Groups", groups, default=groups[:2])
        months = [datetime(2021, m, 1).strftime("%B") for m in range(1, 13)]
        month = st.selectbox("Month", months, index=0)
        mnum = datetime.strptime(month, "%B").month
        df_m = df[(df["priceArea"] == pa) & (df["startTime"].dt.month == mnum)]
        if chosen:
            df_m = df_m[df_m["productionGroup"].isin(chosen)]
        if df_m.empty:
            st.warning("No data for this selection.")
        else:
            pivot = df_m.pivot_table(index="startTime", columns="productionGroup",
                                     values="quantityKwh", aggfunc="sum").fillna(0)
            st.line_chart(pivot)
    with st.expander("Data Source Information"):
        st.markdown("""
        **Source:** [Elhub API](https://api.elhub.no)  
        **Pipeline:** Spark → Cassandra → MongoDB  
        **Year:** 2021  
        **Purpose:** Dashboard for Part 2 of IND320 project.
        """)


# ============================
# ---------- PART 3 -----------
# ============================

def page_stl_spectrogram(df_elhub):
    st.header("📊 STL Decomposition & Spectrogram (Part 3A)")
    if df_elhub.empty:
        st.warning("No Elhub data loaded.")
        return
    pa = st.selectbox("Select Price Area", sorted(df_elhub["priceArea"].unique()))
    pg = st.selectbox("Select Production Group", sorted(df_elhub["productionGroup"].unique()))
    df_sel = df_elhub[(df_elhub["priceArea"] == pa) & (df_elhub["productionGroup"] == pg)]
    df_daily = df_sel.groupby(df_sel["startTime"].dt.date)["quantityKwh"].sum().reset_index()
    df_daily["startTime"] = pd.to_datetime(df_daily["startTime"])
    tab1, tab2 = st.tabs(["📈 STL Decomposition", "🎧 Spectrogram"])
    with tab1:
        period = st.slider("Period (days)", 7, 90, 30)
        stl = STL(df_daily["quantityKwh"], period=period, robust=True).fit()
        fig = go.Figure()
        for n, s in zip(["Observed", "Trend", "Seasonal", "Residual"],
                        [df_daily["quantityKwh"], stl.trend, stl.seasonal, stl.resid]):
            fig.add_trace(go.Scatter(x=df_daily["startTime"], y=s, mode="lines", name=n))
        fig.update_layout(template="plotly_white", title=f"STL Decomposition — {pg} ({pa})")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        win = st.slider("Window Length", 15, 60, 30)
        ov = st.slider("Window Overlap", 5, 30, 15)
        signal = df_daily["quantityKwh"].values
        f, t, Sxx = spectrogram(signal, fs=1, nperseg=win, noverlap=ov)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        fig2 = go.Figure(data=go.Heatmap(z=Sxx_db, x=t, y=f, colorscale="Viridis"))
        fig2.update_layout(template="plotly_white", title=f"Spectrogram — {pg} ({pa})",
                           xaxis_title="Time (days)", yaxis_title="Frequency (cycles/day)")
        st.plotly_chart(fig2, use_container_width=True)


def page_outlier_anomaly(df_weather):
    st.header("⚡ Outlier & Anomaly Detection (Part 3B)")

    # --- Step 1: Safety check ---
    if df_weather is None or df_weather.empty:
        st.error("❌ Weather dataset is empty. Please check that your CSV file path is correct.")
        st.stop()

    # --- Step 2: Display tabs ---
    tab1, tab2 = st.tabs(["🌡️ Temperature SPC", "🌧️ Precipitation Anomalies (LOF)"])

    # ==========================================================
    # 🌡️ TAB 1 — Temperature SPC (Statistical Process Control)
    # ==========================================================
    with tab1:
        st.subheader("Temperature Outlier Detection (SPC – DCT Method)")

        # --- Parameter sliders ---
        freq_cutoff = st.slider("DCT Frequency Cutoff", 5, 200, 50)
        n_std = st.slider("SPC Sigma Threshold (σ)", 1.0, 5.0, 2.0)

        # Ensure the exact column name 'temperature_2m (°C)'
        if "temperature_2m (°C)" in df_weather.columns:
            df = df_weather.copy()
            df["temperature_2m (°C)"] = df["temperature_2m (°C)"].interpolate().fillna(method='bfill')

            # --- DCT transformation and filtering ---
            temp = df["temperature_2m (°C)"].values
            coeff = dct(temp, norm='ortho')
            coeff[:int(freq_cutoff)] = 0
            satv = idct(coeff, norm='ortho')

            # --- SPC bounds ---
            mean, std = np.mean(satv), np.std(satv)
            upper, lower = mean + n_std * std, mean - n_std * std
            df["outlier"] = (satv > upper) | (satv < lower)

            # --- Plotly interactive plot ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["temperature_2m (°C)"],
                                     mode="lines", name="Temperature (°C)",
                                     line=dict(color="royalblue")))
            fig.add_trace(go.Scatter(x=df.index, y=[upper] * len(df),
                                     mode="lines", name="Upper Bound (+σ)",
                                     line=dict(dash="dash", color="orange")))
            fig.add_trace(go.Scatter(x=df.index, y=[lower] * len(df),
                                     mode="lines", name="Lower Bound (-σ)",
                                     line=dict(dash="dash", color="orange")))
            fig.add_trace(go.Scatter(x=df.index[df["outlier"]],
                                     y=df["temperature_2m (°C)"][df["outlier"]],
                                     mode="markers", name="Outliers",
                                     marker=dict(color="red", size=9, symbol="diamond")))

            fig.update_layout(
                title="Temperature SPC Outlier Detection (DCT-based)",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Detected {df['outlier'].sum()} outliers from {len(df)} records.")
        else:
            st.error("⚠️ Missing column: 'temperature_2m (°C)' — please verify your dataset headers.")


    # ==========================================================
    # 🌧️ TAB 2 — Precipitation Anomalies (LOF)
    # ==========================================================
    with tab2:
        st.subheader("Precipitation Anomaly Detection (LOF)")
        contamination = st.slider("LOF Contamination Ratio", 0.001, 0.05, 0.01)

        # Ensure the exact column name 'precipitation (mm)'
        if "precipitation (mm)" in df_weather.columns:
            df = df_weather.copy()
            df["precipitation (mm)"] = df["precipitation (mm)"].fillna(0)

            # --- LOF outlier detection ---
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            df["anomaly"] = lof.fit_predict(df[["precipitation (mm)"]]) == -1

            # --- Plotly plot ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["precipitation (mm)"],
                mode="lines", name="Normal Observations",
                line=dict(color="royalblue")
            ))
            fig.add_trace(go.Scatter(
                x=df.index[df["anomaly"]],
                y=df["precipitation (mm)"][df["anomaly"]],
                mode="markers", name="Anomalies (LOF)",
                marker=dict(color="red", size=9, symbol="diamond")
            ))

            fig.update_layout(
                title="Precipitation Anomalies — Local Outlier Factor (LOF)",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Detected {df['anomaly'].sum()} anomalies from {len(df)} observations.")
        else:
            st.error("⚠️ Missing column: 'precipitation (mm)' — please verify your dataset headers.")




# --------------- Page 6: About ----------------
def page_about():
    st.header("About / Test Page")
    st.write("Project links and credits.")
    st.markdown("**GitHub Repo:** [WeatherAndEnergyInsights](https://github.com/Srivatsav1298/WeatherAndEnergyInsights)") 
    st.markdown("**Streamlit App:** [weatherandenergyinsights.streamlit.app](https://weatherandenergyinsights.streamlit.app//)")


# --------------- Main ----------------
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    DATA_PATH = "data/open-meteo-subset.csv"
    df_weather = load_data(DATA_PATH)

    # Connect Mongo
    try:
        mongo_password = st.secrets["mongo"]["password"]
        uri = f"mongodb+srv://abbuvatsav:{mongo_password}@cluster0.klxry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri)
        df_elhub = pd.DataFrame(list(client["Cluster0"]["elhub_production_data"].find({}, {"_id": 0})))
        df_elhub["startTime"] = pd.to_datetime(df_elhub["startTime"])
    except Exception:
        df_elhub = pd.DataFrame()

    st.sidebar.title("Navigation")
    pages = {
        "Home": page_home,
        "Data Table": page_table,
        "Plots": page_plots,
        "Mongo Dashboard": page_mongo_dashboard,
        "STL / Spectrogram (Part 3A)": lambda: page_stl_spectrogram(df_elhub),
        "Outlier / Anomaly (Part 3B)": lambda: page_outlier_anomaly(df_weather),
        "About": page_about
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("IND320 Dashboard — Parts 1, 2 & 3")
    if choice in ["Data Table", "Plots"]:
        if not df_weather.empty:
            pages[choice](df_weather)
        else:
            st.error("Cannot display page: Data loading failed.")
    else:
        pages[choice]()


if __name__ == "__main__":
    main()
