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
    allow_sqlite_fallback: bool = True,
) -> None:
    """
    Save a run to Neon/Postgres if available, optionally falling back to SQLite.
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
        if not allow_sqlite_fallback:
            raise RuntimeError(f"Neon DB write failed and SQLite fallback is disabled: {e}") from e
        st.caption(f"⚠️ Neon DB write failed, falling back to SQLite: {e}")

    if not allow_sqlite_fallback:
        raise RuntimeError("Neon DB is not configured or unavailable; SQLite fallback is disabled")

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


# --- Additional helper functions ---
def save_daily_snapshot(
    name: str,
    results_json: str,
    *,
    username: Optional[str] = None,
    row_count: Optional[int] = None,
    duration_sec: Optional[float] = None,
) -> None:
    """
    Convenience wrapper to save a daily snapshot.
    """
    save_run(
        name=name,
        results_json=results_json,
        label="daily_snapshot",
        username=username,
        row_count=row_count,
        duration_sec=duration_sec,
        is_snapshot=True,
    )


def list_runs(
    limit: int = 50,
    include_snapshots: bool = True,
    username: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List recent runs from Neon if available, otherwise SQLite.

    Returns a list of dicts:
    {
        "id": int,
        "name": str,
        "label": str | None,
        "username": str | None,
        "row_count": int | None,
        "duration_sec": float | None,
        "is_snapshot": bool,
        "created_at": datetime | None,
    }
    """
    runs: List[Dict[str, Any]] = []

    # ---- Try Neon first ----
    try:
        conn = get_neon_conn()
        if conn is not None:
            ensure_neon_runs_schema(conn)
            cur = conn.cursor()

            where_clauses = []
            params: List[Any] = []

            if not include_snapshots:
                where_clauses.append("is_snapshot = FALSE")

            if username:
                where_clauses.append("username = %s")
                params.append(username)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            sql = f"""
                SELECT id, name, label, username, row_count, duration_sec, is_snapshot, created_at
                FROM runs
                {where_sql}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)

            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            for r in rows:
                if isinstance(r, dict):
                    runs.append(
                        {
                            "id": r.get("id"),
                            "name": r.get("name"),
                            "label": r.get("label"),
                            "username": r.get("username"),
                            "row_count": r.get("row_count"),
                            "duration_sec": r.get("duration_sec"),
                            "is_snapshot": bool(r.get("is_snapshot")),
                            "created_at": r.get("created_at"),
                        }
                    )
                else:
                    runs.append(
                        {
                            "id": r[0],
                            "name": r[1],
                            "label": r[2],
                            "username": r[3],
                            "row_count": r[4],
                            "duration_sec": r[5],
                            "is_snapshot": bool(r[6]),
                            "created_at": r[7],
                        }
                    )

            return runs
    except Exception as e:
        st.caption(f"⚠️ Failed to list runs from Neon, trying SQLite: {e}")

    # ---- SQLite fallback ----
    try:
        conn = get_sqlite_conn()
        ensure_sqlite_runs_schema(conn)
        cur = conn.cursor()

        where_clauses = []
        params2: List[Any] = []

        if not include_snapshots:
            where_clauses.append("is_snapshot = 0")

        if username:
            where_clauses.append("username = ?")
            params2.append(username)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT id, name, label, username, row_count, duration_sec, is_snapshot, created_at
            FROM runs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params2.append(limit)

        cur.execute(sql, tuple(params2))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        for r in rows:
            runs.append(
                {
                    "id": r[0],
                    "name": r[1],
                    "label": r[2],
                    "username": r[3],
                    "row_count": r[4],
                    "duration_sec": r[5],
                    "is_snapshot": bool(r[6]),
                    "created_at": r[7],
                }
            )

        return runs
    except Exception as e:
        st.error(f"❌ Failed to list runs from SQLite: {e}")
        return []


def load_run_results(run_id: int) -> Optional[str]:
    """
    Load the results_json for a given run ID, preferring Neon then SQLite.

    Returns:
        results_json string or None if not found.
    """

    # ---- Try Neon first ----
    try:
        conn = get_neon_conn()
        if conn is not None:
            ensure_neon_runs_schema(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT results_json FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            if isinstance(row, dict):
                return row.get("results_json")
            else:
                return row[0]
    except Exception as e:
        st.caption(f"⚠️ Failed to load run {run_id} from Neon, trying SQLite: {e}")

    # ---- SQLite fallback ----
    try:
        conn = get_sqlite_conn()
        ensure_sqlite_runs_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT results_json FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return None

        return row[0]
    except Exception as e:
        st.error(f"❌ Failed to load run {run_id} from SQLite: {e}")
        return None
