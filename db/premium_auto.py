"""Storage for Premium automations (services/premium_auto.py, 2026-11-03).

hsf_premium_auto keeps what the automations made, so the app can show it without a
button press and a restart never makes it twice:

    kind              username   ref                    body
    brief_narrative   "*"        brief snapshot time    the AI Brief summary
    ticker_note       "*"        "<run id>:<TICKER>"    an AI stock note (shared by everyone)
    watchlist_note    account    watchlist id           the latest AI watchlist rundown
    journal_coach     account    "latest"               the latest AI journal coaching note
    trade_alerts      account    TICKER                 JSON: the price alerts added for an open trade

Postgres (Neon) in production; tests pass a SQLite connection through
set_connection_factory.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once

SHARED = "*"
_conn_factory: Optional[Callable[[], Any]] = None
_COLS = ("username", "kind", "ref", "body", "created_at")


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_premium_auto (
            username TEXT NOT NULL,
            kind TEXT NOT NULL,
            ref TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (username, kind, ref)
        )""")
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


def _row(r: Any) -> Dict[str, Any]:
    vals = list(r.values()) if isinstance(r, dict) else list(r)
    return dict(zip(_COLS, vals))


def save(username: str, kind: str, ref: str, body: str) -> None:
    """Insert or replace one row."""
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_premium_auto WHERE username = %s AND kind = %s AND ref = %s", sq),
                    (username, kind, str(ref)))
        cur.execute(_sql("INSERT INTO hsf_premium_auto (username, kind, ref, body, created_at) "
                         "VALUES (%s, %s, %s, %s, %s)", sq), (username, kind, str(ref), body, now))
        conn.commit()
    finally:
        _close(conn, sq)


def get(username: str, kind: str, ref: str) -> Optional[Dict[str, Any]]:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT {', '.join(_COLS)} FROM hsf_premium_auto "
                         "WHERE username = %s AND kind = %s AND ref = %s", sq), (username, kind, str(ref)))
        r = cur.fetchone()
        return _row(r) if r else None
    finally:
        _close(conn, sq)


def delete(username: str, kind: str, ref: str) -> None:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_premium_auto WHERE username = %s AND kind = %s AND ref = %s", sq),
                    (username, kind, str(ref)))
        conn.commit()
    finally:
        _close(conn, sq)


def list_kind(username: str, kind: str) -> List[Dict[str, Any]]:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT {', '.join(_COLS)} FROM hsf_premium_auto WHERE username = %s AND kind = %s", sq),
                    (username, kind))
        return [_row(r) for r in cur.fetchall() or []]
    finally:
        _close(conn, sq)


def prune(kind: str, older_than: dt.datetime) -> int:
    """Drop old rows of one kind (stock notes pile up one set per scan)."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_premium_auto WHERE kind = %s AND created_at < %s", sq),
                    (kind, older_than.astimezone(dt.timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
        return cur.rowcount or 0
    finally:
        _close(conn, sq)


def delete_all_for_user(username: str) -> int:
    """Account deletion."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_premium_auto WHERE username = %s", sq), (username,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        _close(conn, sq)
