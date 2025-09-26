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
        
        # Show the first few rows to check the data (only in initial run/rerun)
        st.write("Preview of data:")
        st.write(df.head()) 
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
                    # Filter data to a manageable size for 'small chart' interpretation, e.g., the first month
                    # Assuming an index frequency that makes the first 700 rows roughly representative of a short period
                    # Adjusting to a small number of rows just for a 'small' chart example:
                    st.line_chart(df[colname].head(100))
                except Exception as e:
                    st.write("Cannot plot:", e)
    else:
        st.write("No columns to display charts. Please check the dataset.")

def page_plots(df):
    st.header("Interactive plots")
    st.write("Choose a column (or All), a subset of the index, or a dual-axis plot.")

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
        st.error("Index needs at least two points for range selection.")
        return

    # Range slider setup
    start = idx_options[0]
    end = idx_options[-1]
    
    # Ensure the slider value is a tuple (start, end)
    sel = st.select_slider("Select index range (start → end)", options=idx_options, value=(start, end))
    
    
    full_start_idx = idx_options_full.index(sel[0])
    full_end_idx = idx_options_full.index(sel[1])

    # Ensure start is before end
    if full_start_idx > full_end_idx:
        full_start_idx, full_end_idx = full_end_idx, full_start_idx

    # Slice the dataframe
    df_filtered = df.iloc[full_start_idx:full_end_idx+1]
    st.markdown(f"Showing data from **{idx_options_full[full_start_idx]}** to **{idx_options_full[full_end_idx]}** ({len(df_filtered)} rows).")

    # Tabbed interface: single/multiple vs dual-axis
    tab1, tab2 = st.tabs(["📊 Single/All Columns", "🪞 Dual-Axis Plot"])

    with tab1:
        column_options = ["All"] + list(df.columns)
        chosen = st.selectbox("Choose a single column or All", column_options, index=0)

        if chosen == "All":
            df_num = df_filtered.select_dtypes(include='number')
            if df_num.shape[1] == 0:
                st.warning("No numeric columns to plot for 'All'.")
            else:
                # Normalization is crucial for plotting all columns together
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
                # Use Matplotlib for dual-axis plots as Streamlit's built-in charts 
                # do not natively support this configuration easily.

                fig, ax1 = plt.subplots(figsize=(10, 5))

                # Left Y-axis
                ax1.plot(df_filtered.index, df_filtered[col1], 'b-', label=col1)
                ax1.set_xlabel('Index')
                ax1.set_ylabel(col1, color='b')
                ax1.tick_params(axis='y', labelcolor='b')

                # Right Y-axis
                ax2 = ax1.twinx()
                ax2.plot(df_filtered.index, df_filtered[col2], 'g-', label=col2)
                ax2.set_ylabel(col2, color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                
                # Add title and grid
                plt.title(f"Dual-axis Plot: {col1} vs {col2}")
                fig.tight_layout()
                plt.grid(True)
                
                # Display the Matplotlib figure in Streamlit
                st.pyplot(fig)
            else:
                st.info("Please select two different columns for dual-axis plotting.")

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
        "About/Test": show_header # A simple placeholder page
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