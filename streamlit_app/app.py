import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient   # ✅ NEW for MongoDB

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
        st.write("Preview of data:")
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# --------------- Helper / Header ----------------
def show_header():
    st.title("IND320 — Dashboard (Part 1 + 2)")
    st.write("Use the sidebar to navigate between pages.")

# --------------- Page 1: Home ----------------
def page_home():
    show_header()
    st.markdown("## Welcome")
    st.write("This dashboard demonstrates the progression of IND320 Parts 1 and 2.")
    st.markdown("""
    **Part 1:**  
    • CSV loaded locally  
    • Data table and exploratory plots  

    **Part 2:**  
    • MongoDB connection  
    • Interactive pie and line charts from API data  
    """)

# --------------- Page 2: Data Table ----------------
def page_table(df):
    st.header("Data Table (CSV)")
    st.dataframe(df.head(200))

    st.markdown("### Small charts for first columns")
    if len(df.columns) > 0:
        n_small = min(4, len(df.columns))
        cols = st.columns(n_small)
        for i, colname in enumerate(df.columns[:n_small]):
            with cols[i]:
                st.subheader(colname)
                try:
                    st.line_chart(df[colname].head(100))
                except Exception as e:
                    st.write("Cannot plot:", e)
    else:
        st.write("No columns to display charts.")

# --------------- Page 3: Plots ----------------
def page_plots(df):
    st.header("Interactive CSV Plots")
    st.write("Choose column(s) and index range to visualize.")
    if df.index.empty:
        st.error("Index appears empty. Check CSV and index parsing.")
        return

    idx_options_full = [str(x) for x in df.index]
    if len(idx_options_full) > 100:
        idx_options = idx_options_full[::len(idx_options_full)//100 + 1]
        if idx_options_full[-1] not in idx_options:
            idx_options.append(idx_options_full[-1])
    else:
        idx_options = idx_options_full
    if len(idx_options) < 2:
        st.error("Index needs at least two points.")
        return

    sel = st.select_slider("Select index range (start → end)", options=idx_options,
                           value=(idx_options[0], idx_options[-1]))
    s_idx, e_idx = idx_options_full.index(sel[0]), idx_options_full.index(sel[1])
    if s_idx > e_idx:
        s_idx, e_idx = e_idx, s_idx
    df_filtered = df.iloc[s_idx:e_idx + 1]
    st.markdown(f"Showing {len(df_filtered)} rows")

    tab1, tab2 = st.tabs(["📊 Single/All Columns", "🪞 Dual-Axis Plot"])
    with tab1:
        column_options = ["All"] + list(df.columns)
        chosen = st.selectbox("Choose column(s)", column_options, index=0)
        if chosen == "All":
            df_num = df_filtered.select_dtypes(include='number')
            if df_num.shape[1] == 0:
                st.warning("No numeric columns.")
            else:
                df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
                st.line_chart(df_norm)
                st.caption("All numeric columns normalized to [0,1]")
        else:
            try:
                series = pd.to_numeric(df_filtered[chosen], errors='coerce')
                st.line_chart(series)
            except Exception as e:
                st.error(f"Could not plot {chosen}: {e}")

    with tab2:
        num_cols = df_filtered.select_dtypes(include='number').columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            c1 = st.selectbox("Left Y-axis", num_cols, index=0)
            c2 = st.selectbox("Right Y-axis", num_cols, index=1)
            if c1 != c2:
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(df_filtered.index, df_filtered[c1], 'b-', label=c1)
                ax1.set_xlabel('Index')
                ax1.set_ylabel(c1, color='b')
                ax2 = ax1.twinx()
                ax2.plot(df_filtered.index, df_filtered[c2], 'g-', label=c2)
                ax2.set_ylabel(c2, color='g')
                plt.title(f"{c1} vs {c2}")
                plt.grid(True)
                st.pyplot(fig)
            else:
                st.info("Select two different columns.")

# --------------- Page 4: Mongo Dashboard (Part 2) ----------------
def page_mongo_dashboard():
    st.header("Production Dashboard — MongoDB (Part 2)")
    st.write("Visualizing Elhub 2021 production data stored in MongoDB.")

    mongo_password = st.secrets["mongo"]["password"]

    # Check if the password is retrieved
    if not mongo_password:
        st.error("MongoDB password is missing. Please set the environment variable MONGO_PASSWORD.")
        return

    # Proceed to connect to MongoDB
    try:
        username = "abbuvatsav"  # Replace with your MongoDB username
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

    # ---- Left: Pie Chart ----
    with left:
        st.subheader("Total Production (Pie Chart)")
        price_areas = sorted(df["priceArea"].dropna().unique())
        pa = st.radio("Price Area", price_areas, index=0)
        df_pa = df[df["priceArea"] == pa]
        agg = df_pa.groupby("productionGroup")["quantityKwh"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        ax1.pie(agg, labels=agg.index, autopct="%1.1f%%", startangle=140)
        ax1.axis("equal")
        ax1.set_title(f"Total Production — {pa}")
        st.pyplot(fig1)

    # ---- Right: Line Chart ----
    with right:
        st.subheader("Monthly Production (Line Chart)")
        groups = sorted(df["productionGroup"].dropna().unique())
        try:
            chosen = st.pills("Production Groups", options=groups, selection_mode="multi")
        except Exception:
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
            st.caption(f"{pa} – {month} 2021")

    # ---- Data Source ----
    with st.expander("Data Source Information"):
        st.markdown("""
        **Source:** [Elhub API](https://api.elhub.no)  
        **Pipeline:** Spark → Cassandra → MongoDB  
        **Year:** 2021  
        **Purpose:** Dashboard for Part 2 of IND320 project.
        """)

# --------------- Page 5: About ----------------
def page_about():
    st.header("About / Test Page")
    st.write("Project links and credits.")
    st.markdown("**GitHub Repo:** [WeatherAndEnergyInsights](https://weatherandenergy.streamlit.app/)") 
    st.markdown("**Streamlit App:** [weatherandenergy.streamlit.app/](https://weatherandenergy.streamlit.app/)")

# --------------- Main ----------------
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    DATA_PATH = "data/open-meteo-subset.csv"
    df = load_data(DATA_PATH)

    st.sidebar.title("Navigation")
    pages = {
        "Home": page_home,
        "Data Table": page_table,
        "Plots": page_plots,
        "Mongo Dashboard": page_mongo_dashboard,   # ✅ NEW PAGE
        "About/Test": page_about
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("IND320 Dashboard — Parts 1 & 2")

    if choice in ["Data Table", "Plots"]:
        if not df.empty:
            pages[choice](df)
        else:
            st.error("Cannot display page: Data loading failed.")
    else:
        pages[choice]()

if __name__ == "__main__":
    main()
