"""Small per-account web app state and in-app feedback.

hsf_web_state has one row per account with three times (ISO text, UTC):
  onboarding_dismissed_at  the Getting started card on Today was closed
  guide_seen_at            the card's "How the HSF Score works" step was opened
  alerts_seen_at           the Alerts page was last opened (for the unread badge)

hsf_feedback keeps every "Send feedback" message from the web app.

Postgres (Neon); tests pass a SQLite connection through set_connection_factory.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once

FIELDS = ("onboarding_dismissed_at", "guide_seen_at", "alerts_seen_at")
_conn_factory: Optional[Callable[[], Any]] = None


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    global _conn_factory
    _conn_factory = factory


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS hsf_web_state ("
                "user_id TEXT PRIMARY KEY, onboarding_dismissed_at TEXT, guide_seen_at TEXT, alerts_seen_at TEXT)")
    serial = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "BIGSERIAL PRIMARY KEY"
    cur.execute(f"CREATE TABLE IF NOT EXISTS hsf_feedback (id {serial}, user_id TEXT NOT NULL, "
                "kind TEXT NOT NULL, message TEXT NOT NULL, path TEXT, plan TEXT, build TEXT, created_at TEXT NOT NULL)")
    # Which messages an admin has answered (founder priority support).
    cur.execute("CREATE TABLE IF NOT EXISTS hsf_feedback_status (feedback_id BIGINT PRIMARY KEY, "
                "answered_at TEXT NOT NULL, answered_by TEXT NOT NULL)")
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


def _sql(sql: str, is_sqlite: bool) -> str:
    return sql.replace("%s", "?") if is_sqlite else sql


def now_iso(now: Optional[dt.datetime] = None) -> str:
    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def get_state(user_id: str) -> Dict[str, Optional[str]]:
    """The account's times; every field None when there is no row yet."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT onboarding_dismissed_at, guide_seen_at, alerts_seen_at FROM hsf_web_state "
                         "WHERE user_id = %s", sq), (user_id,))
        row = cur.fetchone()
    finally:
        _close(conn, sq)
    if not row:
        return {f: None for f in FIELDS}
    vals: List[Any] = [row[f] for f in FIELDS] if isinstance(row, dict) else list(row)
    return {f: (str(v) if v else None) for f, v in zip(FIELDS, vals)}


def mark(user_id: str, field: str, when: Optional[str] = None) -> str:
    """Set one time (now by default) and return it."""
    if field not in FIELDS:
        raise ValueError(field)
    stamp = when or now_iso()
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"INSERT INTO hsf_web_state (user_id, {field}) VALUES (%s, %s) "
                         f"ON CONFLICT (user_id) DO UPDATE SET {field} = EXCLUDED.{field}", sq), (user_id, stamp))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, sq)
    return stamp


def add_feedback(user_id: str, kind: str, message: str, path: Optional[str], plan: Optional[str],
                 build: Optional[str], now: Optional[dt.datetime] = None) -> None:
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("INSERT INTO hsf_feedback (user_id, kind, message, path, plan, build, created_at) "
                         "VALUES (%s, %s, %s, %s, %s, %s, %s)", sq),
                    (user_id, kind, message, path, plan, build, now_iso(now)))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, sq)


FOUNDER_MARK = "Founder"  # api.web_state adds the founder tier label to `plan`


def recent_feedback(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest first, for the admin page, with whether each one came from a founder and
    whether an admin has answered it. Unanswered founder messages come first."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT f.id, f.user_id, f.kind, f.message, f.path, f.plan, f.build, f.created_at, "
                         "s.answered_at, s.answered_by FROM hsf_feedback f "
                         "LEFT JOIN hsf_feedback_status s ON s.feedback_id = f.id "
                         "ORDER BY f.id DESC LIMIT %s", sq), (int(limit),))
        rows = cur.fetchall() or []
    finally:
        _close(conn, sq)
    keys = ("id", "user_id", "kind", "message", "path", "plan", "build", "created_at", "answered_at", "answered_by")
    out = [{k: (r[k] if isinstance(r, dict) else r[i]) for i, k in enumerate(keys)} for r in rows]
    for f in out:
        f["founder"] = FOUNDER_MARK in str(f.get("plan") or "")
    out.sort(key=lambda f: 0 if (f["founder"] and not f["answered_at"]) else 1)  # stable: newest first within
    return out


def mark_answered(feedback_id: int, admin: str, answered: bool = True, now: Optional[dt.datetime] = None) -> bool:
    """Mark one message answered (or not). False when there is no such message."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT id FROM hsf_feedback WHERE id = %s", sq), (int(feedback_id),))
        if not cur.fetchone():
            return False
        cur.execute(_sql("DELETE FROM hsf_feedback_status WHERE feedback_id = %s", sq), (int(feedback_id),))
        if answered:
            cur.execute(_sql("INSERT INTO hsf_feedback_status (feedback_id, answered_at, answered_by) "
                             "VALUES (%s, %s, %s)", sq), (int(feedback_id), now_iso(now), admin))
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, sq)


def unanswered_founder_feedback() -> Dict[str, Any]:
    """How many founder messages still wait for an answer, and when the oldest arrived."""
    conn, sq = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT COUNT(*) AS n, MIN(f.created_at) AS oldest FROM hsf_feedback f "
                         "LEFT JOIN hsf_feedback_status s ON s.feedback_id = f.id "
                         "WHERE s.feedback_id IS NULL AND f.plan LIKE %s", sq), (f"%{FOUNDER_MARK}%",))
        r = cur.fetchone()
    finally:
        _close(conn, sq)
    n = (r["n"] if isinstance(r, dict) else r[0]) if r else 0
    oldest = (r["oldest"] if isinstance(r, dict) else r[1]) if r else None
    return {"count": int(n or 0), "oldest": oldest}
