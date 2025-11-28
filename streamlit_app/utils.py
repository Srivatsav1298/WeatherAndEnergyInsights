# utils.py
import streamlit as st
import pandas as pd
import time
from pymongo import MongoClient
from datetime import datetime

@st.cache_data(show_spinner=False)
def load_data(path: str):
    """Load local weather CSV with caching and safe parsing."""
    try:
        start = time.time()
        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True
        )
        # ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # leave as-is and warn
                st.warning("Could not parse index to datetime in weather CSV.")
        st.success(f"Weather CSV loaded: {len(df):,} rows ({time.time()-start:.2f}s)")
        return df
    except FileNotFoundError as fe:
        st.error(f"Weather CSV not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading weather CSV: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_mongo_data(password: str):
    """Load Elhub production data from your MongoDB (cached)."""
    try:
        start = time.time()
        username = "abbuvatsav"
        uri = f"mongodb+srv://{username}:{password}@cluster0.klxry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        collection = client["Cluster0"]["elhub_production_data"]
        docs = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(docs)
        if not df.empty and "startTime" in df.columns:
            df["startTime"] = pd.to_datetime(df["startTime"])
        st.success(f"Mongo data loaded: {len(df):,} rows ({time.time()-start:.2f}s)")
        return df
    except Exception as e:
        st.error(f"Could not connect/read MongoDB: {e}")
        return pd.DataFrame()

def safe_set_page_config():
    # centralised page config
    try:
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
