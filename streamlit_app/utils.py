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
    """Load BOTH production and consumption data from MongoDB, merge them, and standardize columns."""
    try:
        start = time.time()
        username = "abbuvatsav"
        uri = f"mongodb+srv://{username}:{password}@cluster0.klxry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        db = client["Cluster0"]

        # ---- READ PRODUCTION ----
        prod_docs = list(db["elhub_production_data"].find({}, {"_id": 0}))
        df_prod = pd.DataFrame(prod_docs)

        # ---- READ CONSUMPTION ----
        cons_docs = list(db["elhub_consumption_2021_2024"].find({}, {"_id": 0}))
        df_cons = pd.DataFrame(cons_docs)

        # If consumption is missing, warn but do not crash
        if df_cons.empty:
            st.warning("⚠ No consumption data found in MongoDB.")

        # ---- STANDARDIZE COLUMN NAMES ----
        def canonicalize(df):
            if df.empty:
                return df
            df.columns = [c.strip() for c in df.columns]

            col_map = {c.lower(): c for c in df.columns}

            canonical_map = {
                "starttime": "startTime",
                "pricearea": "priceArea",
                "quantitykwh": "quantityKwh",
                "productiongroup": "productionGroup",
                "consumptiongroup": "consumptionGroup",
                "recordtype": "recordType"
            }

            rename_dict = {}
            for lower_key, canonical in canonical_map.items():
                if lower_key in col_map and canonical not in df.columns:
                    rename_dict[col_map[lower_key]] = canonical

            if rename_dict:
                df = df.rename(columns=rename_dict)

            if "startTime" in df.columns:
                df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")

            return df

        df_prod = canonicalize(df_prod)
        df_cons = canonicalize(df_cons)

        # ---- MERGE BOTH (union) ----
        df_all = pd.concat([df_prod, df_cons], ignore_index=True)

        st.success(f"Mongo data loaded: {len(df_all):,} rows ({time.time()-start:.2f}s)")
        return df_all

    except Exception as e:
        st.error(f"Could not connect/read MongoDB: {e}")
        return pd.DataFrame()


def safe_set_page_config():
    # centralised page config
    try:
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
