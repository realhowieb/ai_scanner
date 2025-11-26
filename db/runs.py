# db/runs.py
from typing import Optional, List, Dict, Any
import streamlit as st
import json
import sqlite3

from .engine import get_neon_conn, get_sqlite_conn
from .schema import ensure_neon_runs_schema, ensure_sqlite_runs_schema


def save_run(
    name: str,
    results_json: str,
    *,
    label: Optional[str] = None,
    username: Optional[str] = None,
    row_count: Optional[int] = None,
    duration_sec: Optional[float] = None,
    is_snapshot: bool = False,
) -> None:
    """
    Save a run (scan results) to Neon if available, otherwise SQLite.
    """

    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            ensure_neon_runs_schema(conn)
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO runs (
                    name, results_json, label, username, row_count,
                    duration_sec, is_snapshot
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name,
                    results_json,
                    label,
                    username,
                    row_count,
                    duration_sec,
                    is_snapshot,
                ),
            )

            conn.commit()
            cur.close()
            conn.close()
            return
    except Exception as e:
        st.caption(f"⚠️ Neon DB write failed, falling back to SQLite: {e}")

    # --- SQLite fallback ---
    try:
        conn = get_sqlite_conn()
        ensure_sqlite_runs_schema(conn)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO runs (
                name, results_json, label, username, row_count,
                duration_sec, is_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                results_json,
                label,
                username,
                row_count,
                duration_sec,
                1 if is_snapshot else 0,
            ),
        )

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"❌ Failed to save run in SQLite fallback: {e}")