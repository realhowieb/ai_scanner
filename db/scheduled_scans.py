"""Storage for scheduled custom scans (Premium, services/scheduled_scans.py).

hsf_scheduled_scans has one row per saved schedule: the owner, a name, the custom scan
request (the same JSON body as POST /v1/scans), when it runs (open, midday or close) and
what happened last time. `last_key` ("2026-11-02:open") is the run the row last claimed,
so a restart or a second API instance never runs the same slot twice.

Postgres (Neon) in production; tests pass a SQLite connection through
set_connection_factory.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once

RUN_AT = ("open", "midday", "close")
MAX_PER_USER = 3

_conn_factory: Optional[Callable[[], Any]] = None
_COLS = ("id", "username", "name", "request", "run_at", "created_at", "last_key", "last_status",
         "last_detail", "last_run_at", "last_scan_id")


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory


class ScheduleLimitReached(ValueError):
    """The account already has MAX_PER_USER schedules."""


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_scheduled_scans (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            name TEXT NOT NULL,
            request TEXT NOT NULL,
            run_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_key TEXT,
            last_status TEXT,
            last_detail TEXT,
            last_run_at TEXT,
            last_scan_id TEXT
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS hsf_scheduled_scans_user ON hsf_scheduled_scans (username)")
    conn.commit()
    cur.close()


def _connect() -> Tuple[Any, bool]:
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")
    is_sqlite = isinstance(conn, sqlite3.Connection)
    _ensure_schema(conn, is_sqlite)
    return conn, is_sqlite


def _sql(query: str, is_sqlite: bool) -> str:
    return query.replace("%s", "?") if is_sqlite else query


def _close(conn, is_sqlite: bool) -> None:
    if is_sqlite and _conn_factory is not None:
        return  # the test's shared connection
    try:
        conn.close()
    except Exception:
        pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _row(r: Any) -> Dict[str, Any]:
    vals = list(r.values()) if isinstance(r, dict) else list(r)
    out = dict(zip(_COLS, vals))
    try:
        out["request"] = json.loads(out["request"] or "{}")
    except (TypeError, ValueError):
        out["request"] = {}
    return out


def _select(where: str, params: tuple) -> List[Dict[str, Any]]:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT {', '.join(_COLS)} FROM hsf_scheduled_scans {where}", sq), params)
        return [_row(r) for r in cur.fetchall() or []]
    finally:
        _close(conn, sq)


def list_for_user(username: str) -> List[Dict[str, Any]]:
    return _select("WHERE username = %s ORDER BY created_at", (username,))


def list_due(run_at: str, key: str) -> List[Dict[str, Any]]:
    """Schedules for this slot that haven't claimed this run yet, oldest first."""
    return _select("WHERE run_at = %s AND (last_key IS NULL OR last_key <> %s) ORDER BY created_at", (run_at, key))


def create(username: str, name: str, request: Dict[str, Any], run_at: str) -> Dict[str, Any]:
    if run_at not in RUN_AT:
        raise ValueError("run_at must be open, midday or close.")
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT COUNT(*) FROM hsf_scheduled_scans WHERE username = %s", sq), (username,))
        r = cur.fetchone()
        n = int((list(r.values()) if isinstance(r, dict) else list(r))[0] or 0)
        if n >= MAX_PER_USER:
            raise ScheduleLimitReached(f"You can have up to {MAX_PER_USER} scheduled scans.")
        row = {"id": uuid.uuid4().hex, "username": username, "name": name.strip()[:60] or "My scan",
               "request": json.dumps(request), "run_at": run_at, "created_at": _now()}
        cur.execute(_sql("INSERT INTO hsf_scheduled_scans (id, username, name, request, run_at, created_at) "
                         "VALUES (%s, %s, %s, %s, %s, %s)", sq),
                    (row["id"], row["username"], row["name"], row["request"], row["run_at"], row["created_at"]))
        conn.commit()
    finally:
        _close(conn, sq)
    return get(username, row["id"]) or {}


def get(username: str, schedule_id: str) -> Optional[Dict[str, Any]]:
    rows = _select("WHERE id = %s AND username = %s", (schedule_id, username))
    return rows[0] if rows else None


def delete(username: str, schedule_id: str) -> bool:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_scheduled_scans WHERE id = %s AND username = %s", sq),
                    (schedule_id, username))
        conn.commit()
        return (cur.rowcount or 0) > 0
    finally:
        _close(conn, sq)


def delete_all_for_user(username: str) -> int:
    """Account deletion."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_scheduled_scans WHERE username = %s", sq), (username,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        _close(conn, sq)


def claim(schedule_id: str, key: str) -> Optional[str]:
    """Claim this run for the schedule. Returns the previous key ("" when none) for
    release(), or None when another pass already claimed it."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT last_key FROM hsf_scheduled_scans WHERE id = %s", sq), (schedule_id,))
        r = cur.fetchone()
        if r is None:
            return None
        prev = (list(r.values()) if isinstance(r, dict) else list(r))[0] or ""
        if prev == key:
            return None
        if prev:
            cur.execute(_sql("UPDATE hsf_scheduled_scans SET last_key = %s WHERE id = %s AND last_key = %s", sq),
                        (key, schedule_id, prev))
        else:
            cur.execute(_sql("UPDATE hsf_scheduled_scans SET last_key = %s WHERE id = %s AND last_key IS NULL", sq),
                        (key, schedule_id))
        conn.commit()
        return prev if (cur.rowcount or 0) == 1 else None
    finally:
        _close(conn, sq)


def release(schedule_id: str, prev_key: str) -> None:
    """Undo claim() so the next pass tries again (scanner busy, one scan already running)."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("UPDATE hsf_scheduled_scans SET last_key = %s WHERE id = %s", sq),
                    (prev_key or None, schedule_id))
        conn.commit()
    finally:
        _close(conn, sq)


def record(schedule_id: str, status: str, detail: str = "", scan_id: Optional[str] = None) -> None:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("UPDATE hsf_scheduled_scans SET last_status = %s, last_detail = %s, last_run_at = %s, "
                         "last_scan_id = COALESCE(%s, last_scan_id) WHERE id = %s", sq),
                    (status, (detail or "")[:300], _now(), scan_id, schedule_id))
        conn.commit()
    finally:
        _close(conn, sq)
