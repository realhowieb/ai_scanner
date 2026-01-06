

# db/core.py
# Centralized database connection helper

import os
from typing import Optional


def _get_database_url() -> str:
    """
    Return DATABASE_URL from env or Streamlit secrets.
    Supports both uppercase and lowercase keys.
    """
    dsn = os.getenv("DATABASE_URL") or os.getenv("database_url")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (checked DATABASE_URL and database_url)")
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
        return psycopg.connect(dsn)
    except Exception:
        pass

    # Fallback to psycopg2
    try:
        import psycopg2  # type: ignore
        return psycopg2.connect(dsn)
    except Exception as e:
        raise RuntimeError(f"Unable to connect to Postgres: {e}")