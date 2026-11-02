"""The "weekly setups" email list: people who gave an email on a public page without
creating an account.

Double opt-in: a sign-up is `pending` until the emailed confirm link is opened, and only
`confirmed` addresses are exported or synced to the newsletter tool (services/marketing_list.py).
Every row has one random token; the confirm and unsubscribe links carry it, so a link
can only change that one address's list status. Where the visitor came from (source and
utm tags, as ui.acquisition cleans them) is kept for the admin Marketing report.

Timestamps are ISO text so the same SQL runs on SQLite in tests. Best effort: reads
return empty and writes return None when the database can't be reached.
"""
from __future__ import annotations

import datetime as dt
import re
import secrets
from typing import Any, Dict, List, Mapping, Optional

from .engine import get_neon_conn, schema_once

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hsf_marketing_subscribers (
    email TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    token TEXT UNIQUE NOT NULL,
    source TEXT,
    utm_source TEXT,
    utm_medium TEXT,
    utm_campaign TEXT,
    referrer_domain TEXT,
    surface TEXT,
    created_at TEXT NOT NULL,
    confirm_sent_at TEXT,
    confirmed_at TEXT,
    unsubscribed_at TEXT
)
"""
STATUSES = ("pending", "confirmed", "unsubscribed")
RESEND_AFTER = dt.timedelta(minutes=10)  # one confirm email per address per 10 minutes
EMAIL_RE = re.compile(r"^[^@\s<>\"',;]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}$")


def normalize_email(raw: Any) -> Optional[str]:
    email = str(raw or "").strip().lower()
    return email if len(email) <= 254 and EMAIL_RE.match(email) and ".." not in email else None


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat()


def _parse(v: Any) -> Optional[dt.datetime]:
    if not v:
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(str(v))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _row(cur, row: Any) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    return {d[0]: v for d, v in zip(cur.description, row)}


@schema_once
def _ensure_table(conn) -> None:
    cur = conn.cursor()
    cur.execute(TABLE_SQL)
    conn.commit()
    cur.close()


def _conn():
    conn = get_neon_conn()
    if conn is None:
        return None
    _ensure_table(conn)
    return conn


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _get(cur, column: str, value: str) -> Optional[Dict[str, Any]]:
    cur.execute(f"SELECT * FROM hsf_marketing_subscribers WHERE {column} = %s", (value,))
    return _row(cur, cur.fetchone())


def subscribe(email: str, attribution: Mapping[str, Any] | None = None, surface: str = "",
              now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """Add or refresh a sign-up. Returns {token, status, send_confirm} or None (no database).

    A new or unsubscribed address becomes pending and gets a confirm email; a pending one
    gets another only after RESEND_AFTER; a confirmed one gets nothing (the caller answers
    the same way in every case, so the form never reveals who is on the list)."""
    email = normalize_email(email)
    if not email:
        return None
    now = now or _now()
    attr = dict(attribution or {})
    try:
        conn = _conn()
        if conn is None:
            return None
    except Exception:
        return None
    try:
        cur = conn.cursor()
        row = _get(cur, "email", email)
        if row is None:
            token = secrets.token_urlsafe(24)
            cur.execute(
                "INSERT INTO hsf_marketing_subscribers (email, status, token, source, utm_source, utm_medium, "
                "utm_campaign, referrer_domain, surface, created_at, confirm_sent_at) "
                "VALUES (%s, 'pending', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (email, token, attr.get("source") or "direct", attr.get("utm_source"), attr.get("utm_medium"),
                 attr.get("utm_campaign"), attr.get("referrer_domain"), (surface or "web")[:40], _iso(now), _iso(now)))
            conn.commit()
            return {"token": token, "status": "pending", "send_confirm": True}
        status, token = str(row.get("status") or "pending"), str(row["token"])
        if status == "confirmed":
            return {"token": token, "status": status, "send_confirm": False}
        sent = _parse(row.get("confirm_sent_at"))
        if status == "pending" and sent and now - sent < RESEND_AFTER:
            return {"token": token, "status": status, "send_confirm": False}
        cur.execute("UPDATE hsf_marketing_subscribers SET status = 'pending', confirm_sent_at = %s, "
                    "unsubscribed_at = NULL WHERE email = %s", (_iso(now), email))
        conn.commit()
        return {"token": token, "status": "pending", "send_confirm": True}
    except Exception:
        return None
    finally:
        _close(conn)


def _set_status(token: str, status: str, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    token = str(token or "").strip()
    if not token or status not in STATUSES:
        return None
    now = now or _now()
    try:
        conn = _conn()
        if conn is None:
            return None
    except Exception:
        return None
    try:
        cur = conn.cursor()
        row = _get(cur, "token", token)
        if row is None:
            return None
        before = str(row.get("status") or "pending")
        if before != status:
            stamp = "confirmed_at" if status == "confirmed" else "unsubscribed_at"
            cur.execute(f"UPDATE hsf_marketing_subscribers SET status = %s, {stamp} = %s WHERE token = %s",
                        (status, _iso(now), token))
            conn.commit()
        return {"email": row["email"], "status": status, "changed": before != status}
    except Exception:
        return None
    finally:
        _close(conn)


def confirm(token: str, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """{email, status, changed} for a valid token (an unsubscribed address stays off), else None."""
    token = str(token or "").strip()
    try:
        conn = _conn()
        if conn is None:
            return None
        try:
            row = _get(conn.cursor(), "token", token) if token else None
        finally:
            _close(conn)
    except Exception:
        return None
    if row is None:
        return None
    if row.get("status") == "unsubscribed":
        return {"email": row["email"], "status": "unsubscribed", "changed": False}
    return _set_status(token, "confirmed", now)


def unsubscribe(token: str, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    return _set_status(token, "unsubscribed", now)


def lookup(token: str) -> Optional[Dict[str, Any]]:
    """{email, status} for a token, without changing anything (mail scanners open links)."""
    token = str(token or "").strip()
    if not token:
        return None
    try:
        conn = _conn()
        if conn is None:
            return None
        try:
            row = _get(conn.cursor(), "token", token)
        finally:
            _close(conn)
    except Exception:
        return None
    return {"email": row["email"], "status": row.get("status") or "pending"} if row else None


def all_rows(limit: int = 50_000) -> List[Dict[str, Any]]:
    """Every row (newest first), for the admin report and export. Empty when unavailable."""
    try:
        conn = _conn()
        if conn is None:
            return []
    except Exception:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT email, status, source, utm_source, utm_medium, utm_campaign, referrer_domain, surface, "
                    "created_at, confirmed_at, unsubscribed_at FROM hsf_marketing_subscribers "
                    "ORDER BY created_at DESC LIMIT %s", (int(limit),))
        return [r for r in (_row(cur, x) for x in cur.fetchall() or []) if r]
    except Exception:
        return []
    finally:
        _close(conn)
