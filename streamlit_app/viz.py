# viz.py
import streamlit as st
def safe_set_page_config():
    try:
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
