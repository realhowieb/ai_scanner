"""Database access for the API: accounts, sign-in checks and refresh tokens.

Reads the app's existing `users` and `login_attempts` tables; writes only its
own `api_refresh_tokens` table.
"""
from __future__ import annotations

import datetime as dt
import functools
from typing import Any, Dict, Optional

from db.engine import get_neon_conn, schema_once


@functools.lru_cache(maxsize=1)
def _dummy_hash() -> bytes:
    """A bcrypt hash of random bytes, checked when the account doesn't exist so a
    missing account takes as long as a wrong password."""
    import secrets

    import bcrypt

    return bcrypt.hashpw(secrets.token_bytes(16), bcrypt.gensalt(12))


def _conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("database unavailable")
    return conn


def _row(cur) -> Optional[Dict[str, Any]]:
    row = cur.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    return {d.name: v for d, v in zip(cur.description, row)}


@schema_once
def ensure_refresh_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_refresh_tokens (
            token_hash TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ,
            client TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS api_refresh_tokens_user ON api_refresh_tokens (username)")
    conn.commit()
    cur.close()


def get_account(username: str) -> Optional[Dict[str, Any]]:
    """username, full_name, password hash, tier, is_admin, is_active, or None."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT username, full_name, password, tier, is_admin, is_active FROM users "
            "WHERE lower(username) = %s LIMIT 1",
            ((username or "").strip().lower(),),
        )
        row = _row(cur)
        cur.close()
        return row
    finally:
        conn.close()


def check_password(account: Optional[Dict[str, Any]], password: str) -> bool:
    """bcrypt check; tries the password as typed and stripped (same as the web app).
    Only bcrypt hashes are accepted (plain-text passwords were migrated, P1-55)."""
    import bcrypt

    stored = (account or {}).get("password")
    if isinstance(stored, (bytes, bytearray)):
        stored = stored.decode("utf-8", "ignore")
    stored_b = str(stored or "").encode("utf-8")
    if not stored_b.startswith((b"$2a$", b"$2b$", b"$2y$")):
        bcrypt.checkpw(b"x", _dummy_hash())
        return False
    for candidate in dict.fromkeys([password, password.strip()]):
        if candidate and bcrypt.checkpw(candidate.encode("utf-8"), stored_b):
            return True
    return False


def burn_password_check() -> None:
    import bcrypt

    bcrypt.checkpw(b"x", _dummy_hash())


def save_refresh_token(token_hash: str, username: str, ttl_s: int, client: str | None) -> None:
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO api_refresh_tokens (token_hash, username, expires_at, client) "
            "VALUES (%s, %s, NOW() + make_interval(secs => %s), %s)",
            (token_hash, username, int(ttl_s), (client or "")[:80] or None),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def use_refresh_token(token_hash: str) -> tuple[str, Optional[str]]:
    """Consume a refresh token (one use). Returns (status, username):
    "ok" (now revoked; caller issues a new one), "reused" (already revoked:
    every session of that account is revoked, since the token was copied),
    or "invalid" (unknown or expired)."""
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT username, expires_at, revoked_at FROM api_refresh_tokens WHERE token_hash = %s FOR UPDATE",
            (token_hash,),
        )
        row = _row(cur)
        if row is None:
            conn.rollback()
            return "invalid", None
        username = row["username"]
        if row["revoked_at"] is not None:
            cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW() "
                        "WHERE username = %s AND revoked_at IS NULL", (username,))
            conn.commit()
            return "reused", username
        expires = row["expires_at"]
        if expires is not None and expires <= dt.datetime.now(dt.timezone.utc):
            conn.rollback()
            return "invalid", None
        cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW() WHERE token_hash = %s", (token_hash,))
        conn.commit()
        cur.close()
        return "ok", username
    finally:
        conn.close()


def revoke_refresh_token(token_hash: str) -> None:
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW() "
                    "WHERE token_hash = %s AND revoked_at IS NULL", (token_hash,))
        conn.commit()
        cur.close()
    finally:
        conn.close()
