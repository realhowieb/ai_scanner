# db/engine.py
from pathlib import Path
import os
import sqlite3

try:
    import streamlit as st
except Exception:  # pragma: no cover - depends on optional UI dependency
    class _StreamlitShim:
        secrets: dict = {}

        @staticmethod
        def caption(*_args, **_kwargs):
            return None

        @staticmethod
        def cache_data(*_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    st = _StreamlitShim()  # type: ignore[assignment]

DB_PATH = Path(__file__).resolve().parent.parent / "scanner.sqlite"

def get_neon_conn():
    """Return a new Neon PostgreSQL connection.

    We try, in order:
    - NEON_DATABASE_URL or DATABASE_URL env var
    - st.secrets["neon"]["database_url"]
    - st.secrets["DATABASE_URL"]
    - st.secrets["neon_database_url"]
    - st.secrets["database_url"]

    If no URL is configured, return None. If a URL is configured but
    the connection fails, also return None (callers should handle this).
    """
    url = None

    # 1) Environment variable (useful in many deployment setups)
    env_url = (
        os.environ.get("NEON_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("database_url")
    )
    if env_url:
        url = env_url

    # 2) Streamlit secrets nested key
    if url is None:
        try:
            url = st.secrets["neon"]["database_url"]  # type: ignore[index]
        except Exception:
            pass

    # 3) Streamlit secrets flat keys
    if url is None:
        for key in ("NEON_DATABASE_URL", "DATABASE_URL", "neon_database_url", "database_url"):
            try:
                candidate = st.secrets[key]  # type: ignore[index]
                if candidate:
                    url = candidate
                    break
            except Exception:
                continue

    if not url:
        # No Neon URL configured at all
        return None

    try:
        import psycopg

        conn = psycopg.connect(url, row_factory=psycopg.rows.dict_row)
        return conn
    except ImportError as e:
        try:
            st.caption(f"⚠️ Neon driver unavailable (psycopg): {e}")
        except Exception:
            pass
        return None
    except Exception as e:
        # Surface a gentle hint in the UI, but don't crash callers.
        try:
            st.caption(f"⚠️ Neon connection failed (engine): {e}")
        except Exception:
            # In non-Streamlit contexts, st.caption may not be available.
            pass
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
