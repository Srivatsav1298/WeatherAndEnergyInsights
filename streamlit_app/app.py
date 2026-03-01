"""
IND320 Weather and Energy Insights Dashboard - Part 1

A Streamlit application for visualizing and analyzing weather and energy data
with interactive plots, statistical summaries, and data exploration capabilities.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Callable, Tuple
from pathlib import Path


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

APP_CONFIG = {
    "title": "IND320 — Dashboard basics (Part 1)",
    "subtitle": "Streamlit demo app for Part 1. Use the sidebar to navigate between pages.",
    "data_path": "data/open-meteo-subset.csv",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

DATE_FORMAT = "%Y-%m-%dT%H:%M"
FIRST_MONTH_FILTER = 1  # January
SPARKLINE_SCALE = None  # Auto-scale for better variation visibility

STATS_COLUMNS = {
    "Variable": st.column_config.TextColumn("Variable"),
    "First month sparkline": st.column_config.LineChartColumn(
        "First month sparkline",
        y_min=SPARKLINE_SCALE,
        y_max=SPARKLINE_SCALE,
        help="Each variable uses its own y-axis scale for better variation visibility."
    ),
    "Count (first month)": st.column_config.NumberColumn("Count (first month)"),
    "Mean (first month)": st.column_config.NumberColumn("Mean (first month)"),
    "Min (first month)": st.column_config.NumberColumn("Min (first month)"),
    "Max (first month)": st.column_config.NumberColumn("Max (first month)"),
    "Std Dev": st.column_config.NumberColumn("Std Dev"),
    "Range": st.column_config.NumberColumn("Range", help="Difference between max and min")
}

PLOT_COLORS = {"primary": "b", "secondary": "g"}
PLOT_FIGURE_SIZE = (10, 5)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def custom_date_parser(date_string: str) -> datetime:
    """
    Parse date strings in ISO format (YYYY-MM-DDTHH:MM).
    
    Args:
        date_string: Date string to parse
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date string format is invalid
    """
    return datetime.strptime(date_string, DATE_FORMAT)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load and parse CSV data with datetime index.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Parsed DataFrame with datetime index, or empty DataFrame on error
    """
    try:
        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=custom_date_parser
        )
        
        if df.index.isnull().any():
            st.warning("⚠️ Some date values could not be parsed correctly. Check the index.")
        
        return df
    
    except FileNotFoundError:
        st.error(f"❌ Error: CSV file not found at '{path}'")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"❌ Error parsing CSV: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error loading CSV: {e}")
        return pd.DataFrame()


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns from DataFrame."""
    return df.select_dtypes(include='number').columns.tolist()


def validate_datetime_index(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has a proper datetime index."""
    if df.index.empty:
        st.error("❌ Index appears empty. Check CSV and index parsing.")
        return False
    
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        st.error("❌ Index is not datetime. Please check your data.")
        return False
    
    return True


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric data to [0, 1] range."""
    return (df - df.min()) / (df.max() - df.min())


def get_month_names(df: pd.DataFrame) -> list:
    """Extract unique month names from datetime index."""
    return df.index.strftime("%B").unique().tolist()


def build_summary_statistics(df: pd.DataFrame, month: int = FIRST_MONTH_FILTER) -> pd.DataFrame:
    """
    Build summary statistics for numeric columns for a specific month.
    
    Args:
        df: Input DataFrame with datetime index
        month: Month number to filter (1-12)
        
    Returns:
        DataFrame with summary statistics
    """
    month_data = df[df.index.month == month]
    summary_data = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = month_data[col].dropna()
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
    
    return pd.DataFrame(summary_data)


def create_dual_axis_plot(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a dual-axis plot for comparing two variables.
    
    Args:
        df: Input DataFrame
        col1: Column for left y-axis
        col2: Column for right y-axis
        
    Returns:
        Figure and axes objects
    """
    fig, ax1 = plt.subplots(figsize=PLOT_FIGURE_SIZE)
    
    ax1.plot(df.index, df[col1], f'{PLOT_COLORS["primary"]}-', label=col1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(col1, color=PLOT_COLORS["primary"])
    ax1.tick_params(axis='y', labelcolor=PLOT_COLORS["primary"])
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[col2], f'{PLOT_COLORS["secondary"]}-', label=col2)
    ax2.set_ylabel(col2, color=PLOT_COLORS["secondary"])
    ax2.tick_params(axis='y', labelcolor=PLOT_COLORS["secondary"])
    
    plt.title(f"Dual-axis Plot: {col1} vs {col2}")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    return fig, ax1


# ============================================================================
# PAGE COMPONENTS
# ============================================================================

def render_header() -> None:
    """Render application header."""
    st.title(APP_CONFIG["title"])
    st.write(APP_CONFIG["subtitle"])


def page_home() -> None:
    """Render home page."""
    render_header()
    st.markdown("## Welcome 👋")
    st.write(
        "This is the home page. The sidebar contains four pages: Home, Data Table, Plots, About/Test."
    )
    
    st.markdown("### Quick Checklist for Part 1:")
    st.markdown("""
    - ✅ CSV file loaded from `data/open-meteo-subset.csv` (local)
    - 📊 Table preview on the **Data Table** page
    - 📈 Row-wise small charts displayed on the **Data Table** page
    - 🎯 Interactive plotting with month range selection on **Plots** page
    """)


def page_table(df: pd.DataFrame) -> None:
    """Render data table page with summary statistics and sparklines.
    
    Args:
        df: Input DataFrame
    """
    st.header("📈 Variables Summary for January (2020-01)")
    
    summary_df = build_summary_statistics(df)
    
    if summary_df.empty:
        st.warning("No numeric columns found in data.")
        return
    
    st.dataframe(
        summary_df,
        column_config=STATS_COLUMNS,
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("ℹ️ Each sparkline uses its own scale — variations are amplified and clearer.")


def page_plots(df: pd.DataFrame) -> None:
    """Render interactive plotting page.
    
    Args:
        df: Input DataFrame
    """
    st.header("📊 Interactive Plots")
    st.write("Choose a column (or All), and a month to visualize.")
    
    if not validate_datetime_index(df):
        return
    
    # Month selection
    months = get_month_names(df)
    month_choice = st.select_slider("Select Month", options=months, value=months[0])
    df_filtered = df[df.index.strftime("%B") == month_choice]
    st.markdown(f"### Showing data for **{month_choice}** ({len(df_filtered)} rows)")
    
    # Plot tabs
    tab1, tab2 = st.tabs(["📊 Single/All Columns", "🪞 Dual-Axis Plot"])
    
    with tab1:
        _render_single_column_plot(df, df_filtered)
    
    with tab2:
        _render_dual_axis_plot(df_filtered)


def _render_single_column_plot(df: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    """Render single column or all columns plot tab."""
    column_options = ["All"] + list(df.columns)
    chosen = st.selectbox("Choose a single column or All", column_options, index=0)
    
    if chosen == "All":
        df_num = df_filtered.select_dtypes(include='number')
        if df_num.empty:
            st.warning("⚠️ No numeric columns to plot for 'All'.")
        else:
            df_norm = normalize_data(df_num)
            st.line_chart(df_norm)
            st.caption("All numeric columns normalized to [0,1] for comparison.")
    else:
        try:
            series = pd.to_numeric(df_filtered[chosen], errors='coerce')
            st.line_chart(series)
            st.caption(f"Plot for column: **{chosen}**")
        except Exception as e:
            st.error(f"❌ Could not plot column '{chosen}': {e}")


def _render_dual_axis_plot(df_filtered: pd.DataFrame) -> None:
    """Render dual-axis plot tab."""
    numeric_cols = get_numeric_columns(df_filtered)
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least two numeric columns for dual-axis plot.")
        return
    
    col1 = st.selectbox("Left Y-axis variable", numeric_cols, index=0)
    col2 = st.selectbox("Right Y-axis variable", numeric_cols, index=1)
    
    if col1 == col2:
        st.info("ℹ️ Please select two different columns for dual-axis plotting.")
        return
    
    fig, _ = create_dual_axis_plot(df_filtered, col1, col2)
    st.pyplot(fig)


def page_about() -> None:
    """Render about page."""
    st.header("ℹ️ About This Project")
    st.write("""
    This Streamlit application demonstrates data visualization and interactive analysis
    for weather and energy insights. It includes interactive plots, statistical summaries,
    and data exploration capabilities.
    """)
    
    st.markdown("### Project Resources")
    st.markdown("""
    - 🔗 **GitHub Repository:** [WeatherAndEnergyInsights](https://github.com/Srivatsav1298/WeatherAndEnergyInsights)
    - 🌐 **Live App:** [Streamlit Application](https://weatherandenergyinsightspart1.streamlit.app)
    - 📚 **Course:** IND320 - Dashboard Basics
    """)
    
    st.markdown("### Features")
    st.markdown("""
    - **Data Import:** Load and parse CSV data with automatic datetime parsing
    - **Summary Statistics:** View key metrics, distributions, and trends
    - **Interactive Plots:** Visualize data with flexible filtering options
    - **Dual-Axis Analysis:** Compare two variables with different scales
    - **Data Normalization:** Normalize variables for cross-variable comparison
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
    )
    
    # Load data
    df = load_data(APP_CONFIG["data_path"])
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    pages: Dict[str, Callable] = {
        "Home": page_home,
        "Data Table": lambda: page_table(df),
        "Plots": lambda: page_plots(df),
        "About/Test": page_about
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("📊 IND320 Dashboard Demo - Weather and Energy Insights")
    
    # Render selected page
    if selection in ["Data Table", "Plots"]:
        if df.empty:
            st.error("❌ Cannot display page: Data loading failed.")
        else:
            pages[selection]()
    else:
        pages[selection]()


if __name__ == '__main__':
    main()