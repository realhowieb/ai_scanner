# db/engine.py
from pathlib import Path
import sqlite3
import psycopg
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "scanner.sqlite"

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
        conn = psycopg.connect(url, row_factory=psycopg.rows.dict_row)
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
    """Return 'neon', 'sqlite', or 'none' based on actual connectivity."""
    # Try Neon
    try:
        conn = get_neon_conn()
        if conn is not None:
            conn.close()
            return "neon"
    except Exception:
        pass

    # Try SQLite
    try:
        conn = get_sqlite_conn()
        conn.close()
        return "sqlite"
    except Exception:
        return "none"