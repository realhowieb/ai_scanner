"""P1-41 — per-account email preferences and unsubscribe links.

Three email types, each on by default:
  digest   morning market digest       (scheduler/morning_digest.py)
  evening  evening market wrap         (scheduler/evening_wrap.py)
  alerts   alert emails                (scheduler/alert_runner.py, intelligence
           alerts, and the billing service's live alerts)

Every email carries a link to pages/unsubscribe.py with the account's random
unsubscribe token. The token is looked up in this table, so no shared secret is
needed across Streamlit Cloud, GitHub Actions and Render. A token only lets
someone switch that account's emails off; it reveals nothing else.

Reads fail open (no row, or no database → the email type counts as on), so a
missing table can't silently stop every email; the jobs log opted-out accounts
as the skip reason "unsubscribed".
"""
from __future__ import annotations

import secrets
from typing import Any, Dict, Optional

from .engine import get_neon_conn

KINDS = ("digest", "evening", "alerts")
LABELS = {"digest": "Morning market digest", "evening": "Evening market wrap", "alerts": "Alert emails"}
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hsf_email_prefs (
    user_id TEXT PRIMARY KEY,
    digest BOOLEAN NOT NULL DEFAULT TRUE,
    evening BOOLEAN NOT NULL DEFAULT TRUE,
    alerts BOOLEAN NOT NULL DEFAULT TRUE,
    unsub_token TEXT UNIQUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _norm(user: Any) -> str:
    return str(user or "").strip().lower()


def _conn():
    conn = get_neon_conn()
    if conn is None:
        return None
    cur = conn.cursor()
    cur.execute(TABLE_SQL)
    conn.commit()
    cur.close()
    return conn


def get_prefs(user: Any) -> Dict[str, bool]:
    """{digest, evening, alerts} for the account; all True when unknown."""
    prefs = {k: True for k in KINDS}
    u = _norm(user)
    if not u:
        return prefs
    try:
        conn = _conn()
        if conn is None:
            return prefs
        cur = conn.cursor()
        cur.execute("SELECT digest, evening, alerts FROM hsf_email_prefs WHERE user_id = %s", (u,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        return prefs
    if row:
        vals = list(row.values()) if isinstance(row, dict) else list(row)
        prefs.update({k: bool(v) for k, v in zip(KINDS, vals)})
    return prefs


def wants_email(user: Any, kind: str) -> bool:
    return bool(get_prefs(user).get(kind, True))


def set_prefs(user: Any, **changes: bool) -> bool:
    """Update one or more of digest / evening / alerts. True when saved."""
    u = _norm(user)
    cols = {k: bool(v) for k, v in changes.items() if k in KINDS}
    if not u or not cols:
        return False
    names = list(cols)
    assign = ", ".join(f"{c} = EXCLUDED.{c}" for c in names)
    try:
        conn = _conn()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO hsf_email_prefs (user_id, {', '.join(names)}) VALUES (%s{', %s' * len(names)}) "
            f"ON CONFLICT (user_id) DO UPDATE SET {assign}, updated_at = NOW()",
            (u, *[cols[c] for c in names]),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def unsubscribe_token(user: Any) -> Optional[str]:
    """The account's unsubscribe token, created on first use. None if unavailable."""
    u = _norm(user)
    if not u:
        return None
    try:
        conn = _conn()
        if conn is None:
            return None
        cur = conn.cursor()
        cur.execute("SELECT unsub_token FROM hsf_email_prefs WHERE user_id = %s", (u,))
        row = cur.fetchone()
        token = (list(row.values())[0] if isinstance(row, dict) else row[0]) if row else None
        if not token:
            token = secrets.token_urlsafe(24)
            cur.execute(
                "INSERT INTO hsf_email_prefs (user_id, unsub_token) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET unsub_token = COALESCE(hsf_email_prefs.unsub_token, EXCLUDED.unsub_token)",
                (u, token),
            )
            cur.execute("SELECT unsub_token FROM hsf_email_prefs WHERE user_id = %s", (u,))
            row = cur.fetchone()
            token = list(row.values())[0] if isinstance(row, dict) else row[0]
        conn.commit()
        cur.close()
        conn.close()
        return token
    except Exception:
        return None


def user_for_token(token: Any) -> Optional[str]:
    t = str(token or "").strip()
    if not t or len(t) > 64:
        return None
    try:
        conn = _conn()
        if conn is None:
            return None
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM hsf_email_prefs WHERE unsub_token = %s", (t,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        return None
    if not row:
        return None
    return list(row.values())[0] if isinstance(row, dict) else row[0]


def unsubscribe_url(user: Any, kind: str) -> Optional[str]:
    """Link for the footer / List-Unsubscribe header, or None if unavailable."""
    token = unsubscribe_token(user)
    if not token:
        return None
    try:
        from config import APP_BASE_URL
    except Exception:
        APP_BASE_URL = "https://hsf-beta.streamlit.app"
    k = kind if kind in KINDS else "all"
    return f"{APP_BASE_URL.rstrip('/')}/unsubscribe?t={token}&k={k}"
