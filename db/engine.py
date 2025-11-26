# db/engine.py
from pathlib import Path
import sqlite3
import psycopg2
import streamlit as st

DB_PATH = Path("scanner.sqlite")

def get_neon_conn():
    try:
        url = st.secrets["neon"]["database_url"]
    except Exception:
        return None
    return psycopg2.connect(url, sslmode="require")

def get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_data(show_spinner=False, ttl=60)
def get_db_status() -> str:
    try:
        _ = st.secrets["neon"]["database_url"]
        return "neon"
    except Exception:
        pass

    if DB_PATH.exists():
        return "sqlite"
    return "none"