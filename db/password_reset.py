"""Password reset token storage and validation."""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from db.engine import get_neon_conn
except Exception:
    get_neon_conn = None  # type: ignore[assignment]


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens (token_hash)"
    )
    conn.commit()
    cur.close()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


_RESET_MAX_PER_HOUR = 3


def _rate_limited(cur, username: str) -> bool:
    """Return True if user has requested too many resets in the last hour."""
    cur.execute(
        """
        SELECT COUNT(*) FROM password_reset_tokens
        WHERE username = %s AND created_at > NOW() - INTERVAL '1 hour'
        """,
        (username,),
    )
    row = cur.fetchone()
    return (row[0] if row else 0) >= _RESET_MAX_PER_HOUR


def create_reset_token(username: str, ttl_minutes: int = 30) -> Optional[str]:
    """Generate a URL-safe token, store its hash, return the raw token.

    Returns None if the user has already requested 3+ resets in the last hour.
    """
    if get_neon_conn is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None

    _ensure_schema(conn)

    cur = conn.cursor()
    if _rate_limited(cur, username):
        cur.close()
        conn.close()
        return None
    cur.close()

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)

    cur = conn.cursor()
    # Invalidate any existing unused tokens for this user first.
    cur.execute(
        "UPDATE password_reset_tokens SET used = TRUE WHERE username = %s AND used = FALSE",
        (username,),
    )
    cur.execute(
        """
        INSERT INTO password_reset_tokens (username, token_hash, expires_at)
        VALUES (%s, %s, %s)
        """,
        (username, _hash(token), expires),
    )
    conn.commit()
    cur.close()
    conn.close()
    return token


def consume_reset_token(token: str) -> Optional[str]:
    """Validate token and return username if valid; mark it used. Returns None if invalid/expired."""
    if get_neon_conn is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None

    _ensure_schema(conn)
    h = _hash(token)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username
        FROM password_reset_tokens
        WHERE token_hash = %s
          AND used = FALSE
          AND expires_at > NOW()
        LIMIT 1
        """,
        (h,),
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return None

    row_id, username = row
    cur.execute("UPDATE password_reset_tokens SET used = TRUE WHERE id = %s", (row_id,))
    conn.commit()
    cur.close()
    conn.close()
    return username
