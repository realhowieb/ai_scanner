"""Email verification token storage and validation."""
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

_TOKEN_TTL_HOURS = 24
_TOKEN_LEN = 32


@schema_once
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

    if isinstance(row, (tuple, list)):
        row_id, username = row[0], row[1]
    else:
        row_id, username = row["id"], row["username"]
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


class VerificationUnavailable(RuntimeError):
    """The verification status couldn't be read (no database, or the lookup failed).

    Distinct from "not verified": callers that gate paid or outbound actions must
    block and say so, not treat it as verified."""


def email_verification_status(username: str) -> bool:
    """True when the account's email is verified, False when it isn't (or the account
    has no row). Raises VerificationUnavailable when the status can't be read."""
    if get_neon_conn is None:
        raise VerificationUnavailable("database driver unavailable")
    try:
        conn = get_neon_conn()
    except Exception as e:
        raise VerificationUnavailable(type(e).__name__) from e
    if conn is None:
        raise VerificationUnavailable("database unavailable")
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT email_verified FROM users WHERE username = %s LIMIT 1",
            ((username or "").strip().lower(),),
        )
        row = cur.fetchone()
        cur.close()
    except Exception as e:
        raise VerificationUnavailable(type(e).__name__) from e
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if row is None:
        return False
    verified = row[0] if isinstance(row, (tuple, list)) else row.get("email_verified")
    return bool(verified)


def is_email_verified(username: str) -> bool:
    """True only when the email is confirmed verified. Fails closed: an unknown
    account or an unreadable status counts as not verified, so a database blip
    can't open an upgrade or send email to an unconfirmed address. Use
    email_verification_status to tell "not verified" from "can't tell"."""
    try:
        return email_verification_status(username)
    except VerificationUnavailable:
        return False
