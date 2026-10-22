"""Storage for the backup scan check (api/scan_watchdog.py).

hsf_scan_watchdog has one row per scan slot the check flagged (missed or failed). The
slot is the primary key, so a slot is flagged once even if the API restarts or runs on
two instances. Postgres (Neon) in production; tests pass a SQLite connection through
set_connection_factory.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any, Callable, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once

_conn_factory: Optional[Callable[[], Any]] = None


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hsf_scan_watchdog (
            slot TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            detail TEXT,
            flagged_at TEXT NOT NULL
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


def claim_slot(slot: str, status: str, detail: str = "") -> bool:
    """Record a flagged slot. False when this slot was already flagged."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        try:
            cur.execute(_sql("INSERT INTO hsf_scan_watchdog (slot, status, detail, flagged_at) "
                             "VALUES (%s, %s, %s, %s)", sq),
                        (slot, status, detail[:500],
                         dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
    finally:
        _close(conn, sq)


def recent_flags(limit: int = 20) -> List[Tuple[str, str, str, str]]:
    """[(slot, status, detail, flagged_at)], newest slot first."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT slot, status, detail, flagged_at FROM hsf_scan_watchdog "
                         "ORDER BY slot DESC LIMIT %s", sq), (int(limit),))
        rows = cur.fetchall() or []
        return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]
    finally:
        _close(conn, sq)


def admin_emails() -> List[str]:
    """Active admin accounts that are email addresses (who gets the ops email when
    HSF_OPS_ALERT_EMAIL isn't set)."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE (is_admin = TRUE OR lower(tier) = 'admin') "
                    "AND is_active = TRUE" if not sq else
                    "SELECT username FROM users WHERE (is_admin = 1 OR lower(tier) = 'admin') AND is_active = 1")
        rows = cur.fetchall() or []
        names = [(tuple(r.values()) if isinstance(r, dict) else tuple(r))[0] for r in rows]
        return sorted({str(n).strip().lower() for n in names if n and "@" in str(n)})
    finally:
        _close(conn, sq)
