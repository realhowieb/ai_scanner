"""One-time "email me a sign-in link" tokens (sign in without a password).

Same shape as db.password_reset: only a SHA-256 of the token is stored, a token works
once, expires after SIGN_IN_LINK_TTL_MINUTES, and an account can ask for 3 per hour.
A new link replaces any older unused one.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from db.engine import get_neon_conn, schema_once
except Exception:
    get_neon_conn = None  # type: ignore[assignment]
    schema_once = lambda fn: fn  # noqa: E731

TTL_MINUTES = 15
MAX_PER_HOUR = 3


@schema_once
def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sign_in_link_tokens (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            used BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_silt_user_created ON sign_in_link_tokens (username, created_at)")
    conn.commit()
    cur.close()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _first(row):
    return row[0] if isinstance(row, (tuple, list)) else list(row.values())[0]


def create_token(username: str, ttl_minutes: int = TTL_MINUTES) -> Optional[str]:
    """A new URL-safe token for the account, or None when it already asked 3 times this hour."""
    if get_neon_conn is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sign_in_link_tokens WHERE username = %s "
                    "AND created_at > NOW() - INTERVAL '1 hour'", (username,))
        row = cur.fetchone()
        if row and int(_first(row) or 0) >= MAX_PER_HOUR:
            cur.close()
            return None
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        cur.execute("UPDATE sign_in_link_tokens SET used = TRUE WHERE username = %s AND used = FALSE", (username,))
        cur.execute("INSERT INTO sign_in_link_tokens (username, token_hash, expires_at) VALUES (%s, %s, %s)",
                    (username, _hash(token), expires))
        conn.commit()
        cur.close()
        return token
    finally:
        conn.close()


def consume_token(token: str) -> Optional[str]:
    """The account's username when the token is valid, unused and unexpired; marks it used
    in the same statement, so two clicks can't both sign in. None otherwise."""
    if get_neon_conn is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("UPDATE sign_in_link_tokens SET used = TRUE WHERE token_hash = %s AND used = FALSE "
                    "AND expires_at > NOW() RETURNING username", (_hash(token),))
        row = cur.fetchone()
        conn.commit()
        cur.close()
        return str(_first(row)) if row else None
    finally:
        conn.close()
