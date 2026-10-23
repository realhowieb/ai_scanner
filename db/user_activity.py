"""When each account last used the app (for the come-back email, services/onboarding_emails.py).

hsf_user_activity has one row per account: last_active_at, updated at most once an
hour per account per API process (touch() keeps the throttle in memory), so a busy
session costs one small write an hour. Postgres (Neon); tests pass a SQLite connection
through set_connection_factory.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once

TOUCH_EVERY_S = 3600
_conn_factory: Optional[Callable[[], Any]] = None
_last_touch: Dict[str, float] = {}
_lock = threading.Lock()


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory
    _last_touch.clear()


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS hsf_user_activity ("
                "user_id TEXT PRIMARY KEY, last_active_at TEXT NOT NULL)")
    conn.commit()
    cur.close()


def _connect() -> Tuple[Any, bool]:
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available.")
    is_sqlite = isinstance(conn, sqlite3.Connection)
    _ensure_schema(conn, is_sqlite)
    return conn, is_sqlite


def _close(conn, is_sqlite: bool) -> None:
    if is_sqlite and _conn_factory is not None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def touch(user_id: str, now: Optional[dt.datetime] = None) -> bool:
    """Record activity, at most once per TOUCH_EVERY_S per account. Never raises."""
    u = str(user_id or "").strip().lower()
    if not u:
        return False
    mono = time.monotonic()
    with _lock:
        if mono - _last_touch.get(u, -1e12) < TOUCH_EVERY_S:
            return False
        _last_touch[u] = mono
        if len(_last_touch) > 50_000:
            _last_touch.clear()
    try:
        conn, sq = _connect()
    except Exception:
        return False
    try:
        cur = conn.cursor()
        ph = "?" if sq else "%s"
        cur.execute(f"INSERT INTO hsf_user_activity (user_id, last_active_at) VALUES ({ph}, {ph}) "
                    "ON CONFLICT (user_id) DO UPDATE SET last_active_at = EXCLUDED.last_active_at",
                    (u, _iso(now or dt.datetime.now(dt.timezone.utc))))
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        _close(conn, sq)


def last_active_map() -> Dict[str, str]:
    """{user_id: last_active_at ISO} for every account seen."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id, last_active_at FROM hsf_user_activity")
        rows = cur.fetchall() or []
    finally:
        _close(conn, sq)
    out: Dict[str, str] = {}
    for r in rows:
        vals: List[Any] = list(r.values()) if isinstance(r, dict) else list(r)
        out[str(vals[0])] = str(vals[1])
    return out
