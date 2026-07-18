# db/runs.py
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from .engine import get_neon_conn, get_sqlite_conn
from .schema import ensure_neon_runs_schema, ensure_sqlite_runs_schema

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows scheduler/headless imports without Streamlit
    class _StreamlitShim:
        @staticmethod
        def caption(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    st = _StreamlitShim()  # type: ignore[assignment]


def sqlite_fallback_enabled(default: bool = True) -> bool:
    """Return whether DB calls may fall back to local SQLite.

    Local Streamlit use defaults to allowing SQLite. Production jobs can set
    AI_SCANNER_SQLITE_FALLBACK=false to fail fast when Neon/Postgres is missing.
    """
    raw = os.getenv("AI_SCANNER_SQLITE_FALLBACK")
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def save_run(
    name: str,
    results_json: str,
    *,
    label: Optional[str] = None,
    username: Optional[str] = None,
    row_count: Optional[int] = None,
    duration_sec: Optional[float] = None,
    is_snapshot: bool = False,
    allow_sqlite_fallback: bool | None = None,
) -> None:
    """
    Save a run to Neon/Postgres if available, optionally falling back to SQLite.
    """

    fallback_allowed = sqlite_fallback_enabled(default=True) if allow_sqlite_fallback is None else bool(allow_sqlite_fallback)

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
        if not fallback_allowed:
            raise RuntimeError(f"Neon DB write failed and SQLite fallback is disabled: {e}") from e
        st.caption(f"⚠️ Neon DB write failed, falling back to SQLite: {e}")

    if not fallback_allowed:
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
    """Save (or refresh) the daily snapshot for this (name, username, UTC day).

    Idempotent per calendar day: if a snapshot for the same universe/user
    already exists today, update it in place (latest wins) instead of inserting
    a duplicate. Multiple in-app scans of the same universe before noon UTC used
    to each create a snapshot row, inflating every per-day analytic (track
    record, calendar heatmap). Best-effort — any failure falls back to a plain
    insert so a snapshot is never lost.
    """
    try:
        from datetime import datetime, timezone

        day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        conn = get_neon_conn()
        if conn is not None:
            ensure_neon_runs_schema(conn)
            cur = conn.cursor()
            if username is None:
                user_clause, user_params = "username IS NULL", ()
            else:
                user_clause, user_params = "username = %s", (username,)
            cur.execute(
                f"""
                UPDATE runs
                   SET results_json = %s, row_count = %s, duration_sec = %s,
                       name = %s, created_at = NOW()
                 WHERE id = (
                     SELECT id FROM runs
                      WHERE is_snapshot = TRUE AND label = 'daily_snapshot'
                        AND name = %s AND {user_clause}
                        AND created_at >= %s
                      ORDER BY created_at DESC
                      LIMIT 1
                 )
                RETURNING id
                """,
                (results_json, row_count, duration_sec, name, name,
                 *user_params, day_start),
            )
            updated = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
            if updated:
                return
    except Exception:
        pass  # fall through to a plain insert below

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


def list_snapshot_runs(days: int = 45, limit: int = 60) -> List[Dict[str, Any]]:
    """Recent snapshot runs only, filtered in SQL.

    Callers used to pull thousands of run-metadata rows via list_runs() to find
    a handful of snapshots; this pushes the is_snapshot + recency filter into
    the query. Neon only (snapshot history features are cloud-backed); returns
    [] when Neon is unavailable.
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            return []
        ensure_neon_runs_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, label, username, row_count, duration_sec,
                   is_snapshot, created_at
            FROM runs
            WHERE is_snapshot = TRUE
              AND created_at >= NOW() - make_interval(days => %s)
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (int(days), int(limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
        out: List[Dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                out.append(dict(r))
            else:
                out.append(
                    {
                        "id": r[0], "name": r[1], "label": r[2], "username": r[3],
                        "row_count": r[4], "duration_sec": r[5],
                        "is_snapshot": bool(r[6]), "created_at": r[7],
                    }
                )
        return out
    except Exception:
        return []


def load_many_run_results(run_ids: List[int]) -> Dict[int, str]:
    """Batch-fetch results_json for several runs in one query.

    Replaces sequential per-run round trips in history loaders. Returns
    {run_id: results_json}; missing/unavailable ids are simply absent.
    """
    ids = [int(i) for i in run_ids if i is not None]
    if not ids:
        return {}
    try:
        conn = get_neon_conn()
        if conn is None:
            return {}
        cur = conn.cursor()
        cur.execute(
            "SELECT id, results_json FROM runs WHERE id = ANY(%s)", (ids,)
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
        out: Dict[int, str] = {}
        for r in rows:
            rid = r["id"] if isinstance(r, dict) else r[0]
            rj = r["results_json"] if isinstance(r, dict) else r[1]
            if rj is not None:
                out[int(rid)] = rj
        return out
    except Exception:
        return {}


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
