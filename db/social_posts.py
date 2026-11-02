"""Weekly social posts sent to Buffer as drafts (services/social_posts.py).

Two tables:
    hsf_social_posts   one row per (kind, week): the post as it was sent, so the image
                       Buffer fetches later shows exactly the names in the text.
    hsf_social_drafts  one row per (kind, week, channel): the Buffer draft id, or the
                       last error and how many attempts failed. A channel with a draft is
                       never sent again for that week.

Timestamps are ISO text so the same SQL runs on SQLite in tests. Best effort: reads
return empty and writes return False when the database can't be reached.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from .engine import get_neon_conn, schema_once

POSTS_SQL = """
CREATE TABLE IF NOT EXISTS hsf_social_posts (
    kind TEXT NOT NULL,
    week TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, week)
)
"""
DRAFTS_SQL = """
CREATE TABLE IF NOT EXISTS hsf_social_drafts (
    kind TEXT NOT NULL,
    week TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    service TEXT,
    channel_name TEXT,
    status TEXT NOT NULL,
    buffer_id TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (kind, week, channel_id)
)
"""


def _iso(t: Optional[dt.datetime] = None) -> str:
    return (t or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).isoformat()


def _rows(cur) -> List[Dict[str, Any]]:
    out = []
    for r in cur.fetchall() or []:
        out.append(dict(r) if isinstance(r, dict) else {d[0]: v for d, v in zip(cur.description, r)})
    return out


@schema_once
def _ensure(conn) -> None:
    cur = conn.cursor()
    cur.execute(POSTS_SQL)
    cur.execute(DRAFTS_SQL)
    conn.commit()
    cur.close()


def _conn():
    conn = get_neon_conn()
    if conn is None:
        return None
    _ensure(conn)
    return conn


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def stored_post(kind: str, week: str) -> Optional[Dict[str, Any]]:
    try:
        conn = _conn()
        if conn is None:
            return None
    except Exception:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM hsf_social_posts WHERE kind = %s AND week = %s", (kind, week))
        rows = _rows(cur)
        return json.loads(rows[0]["payload"]) if rows else None
    except Exception:
        return None
    finally:
        _close(conn)


def store_post(post: Dict[str, Any]) -> bool:
    """Keep the first version of a week's post (a second call leaves it as it was)."""
    try:
        conn = _conn()
        if conn is None:
            return False
    except Exception:
        return False
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO hsf_social_posts (kind, week, payload, created_at) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (kind, week) DO NOTHING",
                    (post["kind"], post["week"], json.dumps(post, default=str), _iso()))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        _close(conn)


def drafts_for(kind: str, week: str) -> Dict[str, Dict[str, Any]]:
    """channel_id -> its row, for one week's post."""
    try:
        conn = _conn()
        if conn is None:
            return {}
    except Exception:
        return {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM hsf_social_drafts WHERE kind = %s AND week = %s", (kind, week))
        return {str(r["channel_id"]): r for r in _rows(cur)}
    except Exception:
        return {}
    finally:
        _close(conn)


def record(kind: str, week: str, channel: Dict[str, Any], *, buffer_id: Optional[str] = None,
           error: Optional[str] = None) -> bool:
    """Save a draft (buffer_id) or a failure (error, one more attempt)."""
    status = "drafted" if buffer_id else "failed"
    try:
        conn = _conn()
        if conn is None:
            return False
    except Exception:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO hsf_social_drafts (kind, week, channel_id, service, channel_name, status, buffer_id, error, "
            "attempts, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (kind, week, channel_id) DO UPDATE SET status = excluded.status, "
            "buffer_id = excluded.buffer_id, error = excluded.error, updated_at = excluded.updated_at, "
            "attempts = hsf_social_drafts.attempts + excluded.attempts",
            (kind, week, str(channel.get("id")), channel.get("service"), (channel.get("name") or "")[:120], status,
             buffer_id, (error or "")[:300] or None, 0 if buffer_id else 1, _iso()))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        _close(conn)


def recent(limit: int = 30) -> List[Dict[str, Any]]:
    try:
        conn = _conn()
        if conn is None:
            return []
    except Exception:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT kind, week, service, channel_name, status, error, attempts, updated_at "
                    "FROM hsf_social_drafts ORDER BY updated_at DESC LIMIT %s", (int(limit),))
        return _rows(cur)
    except Exception:
        return []
    finally:
        _close(conn)
