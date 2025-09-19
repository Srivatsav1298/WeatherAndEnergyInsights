import streamlit as st
import pandas as pd
from datetime import datetime

# Custom date parser to handle ISO format
def custom_date_parser(x):
    return datetime.strptime(x, "%Y-%m-%dT%H:%M")

# Caching read of the CSV for app speed
@st.cache_data
def load_data(path: str):
    try:
        # custom date parser to handle the 'T' in date strings
        df = pd.read_csv(path, index_col=0, parse_dates=True, infer_datetime_format=True, date_parser=custom_date_parser)
        
        # Check if the index was parsed correctly
        if df.index.isnull().any():
            st.warning("Warning: Some date values could not be parsed correctly. Check the index.")
        
        # Show the first few rows to check the data
        st.write("Preview of data:")
        st.write(df.head())  # Show the first few rows for inspection
        
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
    st.header("Data Table")
    st.write("Interactive table of the imported CSV (first 200 rows shown).")
    st.dataframe(df.head(200))

    st.markdown("### Row-wise small charts for the first few columns (interpreting 'first month')")
    if len(df.columns) > 0:
        # Determine the number of small charts to display
        n_small = min(4, len(df.columns))
        cols = st.columns(n_small)
        for i, colname in enumerate(df.columns[:n_small]):
            with cols[i]:
                st.subheader(colname)
                # line chart for each column
                try:
                    st.line_chart(df[colname])
                except Exception as e:
                    st.write("Cannot plot:", e)
    else:
        st.write("No columns to display charts. Please check the dataset.")

def page_plots(df):
    st.header("Interactive plots")
    st.write("Choose a column (or All) and a subset of the index (months/dates).")
    
    # Ensure the index is properly parsed and not empty
    if df.index.empty:
        st.error("Index appears empty. Check CSV and index parsing.")
        return

    # Prepare string representation of index for slider options
    idx_options = [str(x) for x in df.index]
    if len(idx_options) == 0:
        st.error("Index appears empty. Check CSV and index parsing.")
        return

    # Select column or all
    column_options = ["All"] + list(df.columns)
    chosen = st.selectbox("Choose a single column or All", column_options, index=0)

    # Select a subset range using select_slider (defaults to first index only)
    start = idx_options[0]
    end = idx_options[0]
    sel = st.select_slider("Select index range (start → end)", options=idx_options, value=(start, end))
    start_idx = idx_options.index(sel[0])
    end_idx = idx_options.index(sel[1])
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Filtered dataframe
    df_filtered = df.iloc[start_idx:end_idx+1]

    st.markdown(f"Showing indices from **{idx_options[start_idx]}** to **{idx_options[end_idx]}** ({len(df_filtered)} rows).")

    if chosen == "All":
        # If multiple numeric columns exist, normalize to show them on the same axis
        df_num = df_filtered.select_dtypes(include='number')
        if df_num.shape[1] == 0:
            st.warning("No numeric columns to plot for 'All'. Please choose a single column.")
        else:
            # Normalize to [0,1] for visibility across different scales
            df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
            st.line_chart(df_norm)
            st.caption("All numeric columns normalized to [0,1] to allow visual comparison.")
    else:
        # Single column plot with labels
        try:
            series = pd.to_numeric(df_filtered[chosen], errors='coerce')
            st.line_chart(series)
            st.caption(f"Plot for column: {chosen}")
        except Exception as e:
            st.error(f"Could not plot column {chosen}: {e}")

def page_about():
    st.header("About / Test page")
    st.write("Dummy page for part 1. Replace with further content in later parts of the project.")
    st.write("Include links to GitHub and Streamlit app in the final notebook.")

# ---------- main ----------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Table", "Plots", "About/Test"])

    # Try loading data; show helpful error if missing
    try:
        df = load_data("../data/open-meteo-subset.csv")
    except FileNotFoundError:
        st.sidebar.error("CSV file not found. Put open-meteo-subset.csv into the `data/` folder.")
        df = pd.DataFrame()  # empty placeholder

    if page == "Home":
        page_home()
    elif page == "Data Table":
        page_table(df)
    elif page == "Plots":
        page_plots(df)
    else:
        page_about()

if __name__ == "__main__":
    main()
