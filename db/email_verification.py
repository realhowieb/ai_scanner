"""Email verification token storage and validation."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from db.engine import get_neon_conn
except Exception:
    get_neon_conn = None  # type: ignore[assignment]

_TOKEN_TTL_HOURS = 24
_TOKEN_LEN = 32


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_verifications (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            verified_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ev_token ON email_verifications (token_hash)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ev_username ON email_verifications (username)"
    )
    # Mark unverified users in users table
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE")
    conn.commit()
    cur.close()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_verification_token(username: str) -> Optional[str]:
    """Create an email verification token. Returns raw token or None on failure."""
    if get_neon_conn is None:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None

    _ensure_schema(conn)
    token = secrets.token_urlsafe(_TOKEN_LEN)
    expires = datetime.now(timezone.utc) + timedelta(hours=_TOKEN_TTL_HOURS)

    cur = conn.cursor()
    # Invalidate prior unverified tokens for this user.
    cur.execute(
        "DELETE FROM email_verifications WHERE username = %s AND verified_at IS NULL",
        (username,),
    )
    cur.execute(
        """
        INSERT INTO email_verifications (username, token_hash, expires_at)
        VALUES (%s, %s, %s)
        """,
        (username, _hash(token), expires),
    )
    conn.commit()
    cur.close()
    conn.close()
    return token


def consume_verification_token(token: str) -> Optional[str]:
    """Validate and consume a verification token. Returns username or None."""
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
        FROM email_verifications
        WHERE token_hash = %s
          AND verified_at IS NULL
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
    cur.execute(
        "UPDATE email_verifications SET verified_at = NOW() WHERE id = %s",
        (row_id,),
    )
    cur.execute(
        "UPDATE users SET email_verified = TRUE WHERE username = %s",
        (username,),
    )
    conn.commit()
    cur.close()
    conn.close()
    return username


def is_email_verified(username: str) -> bool:
    """Return True if the user's email is verified. Defaults to True when DB unavailable."""
    if get_neon_conn is None:
        return True
    conn = get_neon_conn()
    if conn is None:
        return True
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT email_verified FROM users WHERE username = %s LIMIT 1",
            ((username or "").strip().lower(),),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return True  # unknown user — don't block
        return bool(row[0])
    except Exception:
        return True  # fail open so a DB blip doesn't lock everyone out
