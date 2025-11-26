# db/engine.py
from pathlib import Path
import sqlite3
import psycopg
import streamlit as st

DB_PATH = Path("scanner.sqlite")

def get_neon_conn():
    """Return a new Neon PostgreSQL connection using Streamlit secrets.

    Expects st.secrets["neon"]["database_url"] to contain a full Postgres URI.
    """
    try:
        url = st.secrets["neon"]["database_url"]
    except Exception:
        # Secrets not configured; Neon not available.
        return None

    try:
        conn = psycopg.connect(url)
        return conn
    except Exception as e:
        st.caption(f"⚠️ Neon connection failed (engine): {e}")
        return None

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