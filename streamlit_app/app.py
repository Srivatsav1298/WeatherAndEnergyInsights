import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Custom date parser to handle ISO format
def custom_date_parser(x):
    return datetime.strptime(x, "%Y-%m-%dT%H:%M")

# Caching read of the CSV for app speed
@st.cache_data
def load_data(path: str):
    try:
        # custom date parser to handle the 'T' in date strings
        # index_col=0 assumes the first column is the datetime index
        df = pd.read_csv(path, index_col=0, parse_dates=True, infer_datetime_format=True, date_parser=custom_date_parser)
        
        # Check if the index was parsed correctly
        if df.index.isnull().any():
            st.warning("Warning: Some date values could not be parsed correctly. Check the index.")
        return df
    
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ---------- helper UI functions ----------
def show_header():
    st.title("IND320 — Dashboard basics (Part 1)")
    st.write("Streamlit demo app for Part 1. Use the sidebar to navigate between pages.")

def page_home():
    show_header()
    st.markdown("## Welcome")
    st.write("This is the home page. The sidebar contains four pages: Home, Data Table, Plots, About/Test.")
    st.markdown("**Quick checklist for Part 1:**")
    st.markdown("""
    - CSV file loaded from `data/open-meteo-subset.csv` (local).
    - Table preview on second page.
    - Row-wise small charts (first column used as sample) displayed on second page.
    - Plotting page: choose column(s) and a month range.
    """)

def page_table(df):
    st.header("📊 Data Table — Interactive Overview")

    st.markdown("""
    Below is an **interactive data table** showing all variables.  
    Each numeric column includes a **sparkline (mini line chart)** for quick trend visualization  
    using `st.column_config.LineChartColumn()`.
    """)

    # Detect numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    # Build column config with different display types
    column_config = {}
    for col in numeric_cols:
        column_config[col] = st.column_config.LineChartColumn(
            col,
            help=f"Trend of {col} over time",
            y_min=float(df[col].min()),
            y_max=float(df[col].max())
        )
    for col in non_numeric_cols:
        column_config[col] = st.column_config.Column(
            col,
            help=f"Non-numeric column: {col}"
        )

    # Display the DataFrame with visual enhancement
    st.dataframe(
        df.head(200),  # limit preview for performance
        column_config=column_config,
        use_container_width=True,
        hide_index=True
    )

    st.caption("ℹ️ Hover over sparkline charts to see trends per column. Only first 200 rows shown.")


def page_plots(df):
    st.header("Interactive plots")
    st.write("Choose a column (or All), and a month to visualize.")

    if df.index.empty:
        st.error("Index appears empty. Check CSV and index parsing.")
        return

    # ---- New Month-Based Slider ----
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("Index is not datetime. Please check your data.")
        return

    df["month_name"] = df.index.strftime("%B")
    months = df["month_name"].unique().tolist()
    month_choice = st.select_slider("Select Month", options=months, value=months[0])
    df_filtered = df[df["month_name"] == month_choice]
    st.markdown(f"### Showing data for **{month_choice}** ({len(df_filtered)} rows)")

    # ---- Existing Plot Tabs ----
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
                st.caption("All numeric columns normalized to [0,1] for comparison.")
        else:
            try:
                series = pd.to_numeric(df_filtered[chosen], errors='coerce')
                st.line_chart(series)
                st.caption(f"Plot for column: {chosen}")
            except Exception as e:
                st.error(f"Could not plot column {chosen}: {e}")

    with tab2:
        numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for dual-axis plot.")
        else:
            col1 = st.selectbox("Left Y-axis variable", numeric_cols, index=0)
            col2 = st.selectbox("Right Y-axis variable", numeric_cols, index=1)

            if col1 != col2:
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(df_filtered.index, df_filtered[col1], 'b-', label=col1)
                ax1.set_xlabel('Index')
                ax1.set_ylabel(col1, color='b')
                ax2 = ax1.twinx()
                ax2.plot(df_filtered.index, df_filtered[col2], 'g-', label=col2)
                ax2.set_ylabel(col2, color='g')
                plt.title(f"Dual-axis Plot: {col1} vs {col2}")
                fig.tight_layout()
                plt.grid(True)
                st.pyplot(fig)
            else:
                st.info("Please select two different columns for dual-axis plotting.")

                
def page_about():
    st.header("About / Test Page")
    st.write("Welcome to the 4th Page")

    st.markdown("### Project Links")
    st.markdown("**GitHub Repository:** [https://github.com/Srivatsav1298/WeatherAndEnergyInsights/tree/main](https://github.com/Srivatsav1298/WeatherAndEnergyInsights/tree/main)")
    st.markdown("**Streamlit App:** [https://weatherandenergyinsightspart1.streamlit.app](https://weatherandenergyinsightspart1.streamlit.app)")

# ----------------- MAIN EXECUTION BLOCK -----------------
def main():
    
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    
   
    DATA_PATH = "data/open-meteo-subset.csv"
    
    # Load the data using the cached function
    df = load_data(DATA_PATH)

    # Sidebar configuration
    st.sidebar.title("Navigation")
    
    pages = {
        "Home": page_home,
        "Data Table": page_table,
        "Plots": page_plots,
        "About/Test": page_about
    }
    
    # Create the radio button selector for navigation
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("This is the sidebar for the IND320 dashboard demo.")
    
    # Page Rendering: Call the function corresponding to the user's selection
    if selection == "Data Table" or selection == "Plots":
        # These pages require the DataFrame
        if not df.empty:
            pages[selection](df)
        else:
            st.error("Cannot display page: Data loading failed.")
    else:
        # Home and About pages do not require the DataFrame
        pages[selection]()

# Standard Python entry point
if __name__ == '__main__':
    main()