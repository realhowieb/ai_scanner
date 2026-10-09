

# db/core.py
# Centralized database connection helper

import os
from typing import Optional


def _get_database_url() -> str:
    """
    Return the configured Postgres URL from env or Streamlit secrets.

    Keep this in sync with db.engine/config.py so every DB path recognizes the
    same Neon deployment settings.
    """
    dsn = (
        os.getenv("NEON_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("database_url")
    )
    if not dsn:
        try:
            import streamlit as st  # type: ignore

            dsn = st.secrets["neon"]["database_url"]  # type: ignore[index]
        except Exception:
            dsn = None
    if not dsn:
        try:
            import streamlit as st  # type: ignore

            for key in ("NEON_DATABASE_URL", "DATABASE_URL", "neon_database_url", "database_url"):
                candidate = st.secrets[key]  # type: ignore[index]
                if candidate:
                    dsn = candidate
                    break
        except Exception:
            pass
    if not dsn:
        raise RuntimeError(
            "Database URL is not set (checked NEON_DATABASE_URL, DATABASE_URL, and database_url)"
        )
    return dsn.strip()


def get_conn():
    """
    Return a live Postgres connection.

    Tries psycopg v3 first, then psycopg2 as a fallback.
    This is intentionally lightweight so all modules (earnings, users,
    billing, snapshots) can rely on a single entry point.
    """
    dsn = _get_database_url()

    # Prefer psycopg (v3)
    try:
        import psycopg  # type: ignore

        from db.traffic import connect_options, count

        conn = psycopg.connect(dsn, connect_timeout=10, **connect_options())
        count(connections_opened=1)
        return conn
    except ImportError:
        pass
    except Exception:
        raise RuntimeError("Unable to connect to Postgres") from None

    # Fallback to psycopg2
    try:
        import psycopg2  # type: ignore

        from db.traffic import connect_options, count

        conn = psycopg2.connect(dsn, connect_timeout=10, **connect_options("psycopg2"))
        count(connections_opened=1)
        return conn
    except Exception:
        raise RuntimeError("Unable to connect to Postgres") from None
