"""Database access for the API: accounts, sign-in checks and refresh tokens.

Reads the app's existing `users` and `login_attempts` tables; writes only its
own `api_refresh_tokens` table.
"""
from __future__ import annotations

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


# P1-59 review: a rotated token replayed within this many seconds is a lost
# response or two concurrent refreshes (flaky phone network), not theft.
REUSE_GRACE_S = 30
# Tokens revoked for these reasons answer "invalid" when replayed (no theft sweep).
SIGNED_OUT_REASONS = ("password_change", "password_reset")


class DatabaseUnavailable(RuntimeError):
    """No connection or a connection-level error; the API answers 503."""


def _conn():
    try:
        conn = get_neon_conn()
    except Exception as e:  # driver/connect errors never carry through with details
        raise DatabaseUnavailable("database unavailable") from e
    if conn is None:
        raise DatabaseUnavailable("database unavailable")
    return conn


def ping() -> None:
    """One round trip to the database; DatabaseUnavailable when it can't answer."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
    except Exception as e:
        raise DatabaseUnavailable("database unavailable") from e
    finally:
        conn.close()


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
    cur.execute("ALTER TABLE api_refresh_tokens ADD COLUMN IF NOT EXISTS revoked_reason TEXT")
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
    "ok" (now revoked as rotated; caller issues a new pair),
    "grace" (rotated less than REUSE_GRACE_S ago: a retry after a lost response
    or a concurrent refresh; caller issues another pair, nothing is revoked),
    "reused" (rotated earlier, or replayed after logout or a reuse sweep: the
    token was copied, so every session of that account is revoked), or
    "invalid" (unknown, expired, or signed out by a password change/reset)."""
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT username, expires_at <= NOW() AS expired, revoked_at IS NOT NULL AS revoked, "
            "revoked_reason, revoked_at > NOW() - make_interval(secs => %s) AS recent "
            "FROM api_refresh_tokens WHERE token_hash = %s FOR UPDATE",
            (REUSE_GRACE_S, token_hash),
        )
        row = _row(cur)
        if row is None:
            conn.rollback()
            return "invalid", None
        username = row["username"]
        if row["revoked"]:
            if row["revoked_reason"] == "rotated" and row["recent"] and not row["expired"]:
                conn.rollback()
                return "grace", username
            if row["revoked_reason"] in SIGNED_OUT_REASONS:
                # A device signed out by a password change/reset retrying its old
                # token is expected, not theft: no sweep (it would also sign out
                # the session that just changed the password).
                conn.rollback()
                return "invalid", None
            cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW(), revoked_reason = 'reuse' "
                        "WHERE username = %s AND revoked_at IS NULL", (username,))
            conn.commit()
            return "reused", username
        if row["expired"]:
            conn.rollback()
            return "invalid", None
        cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW(), revoked_reason = 'rotated' "
                    "WHERE token_hash = %s", (token_hash,))
        conn.commit()
        cur.close()
        return "ok", username
    finally:
        conn.close()


def revoke_refresh_token(token_hash: str) -> Optional[str]:
    """Sign out one session. Returns the token's account (None when unknown), also
    for a token already revoked, so a repeated sign-out can still remove its device."""
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT username FROM api_refresh_tokens WHERE token_hash = %s", (token_hash,))
        row = _row(cur)
        cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW(), revoked_reason = 'logout' "
                    "WHERE token_hash = %s AND revoked_at IS NULL", (token_hash,))
        conn.commit()
        cur.close()
        return row["username"] if row else None
    finally:
        conn.close()


def revoke_all_refresh_tokens(username: str, reason: str) -> int:
    """Sign the account out of every app session (password reset or change).
    Access tokens already issued still expire on their own (15 min)."""
    conn = _conn()
    try:
        ensure_refresh_schema(conn)
        cur = conn.cursor()
        cur.execute("UPDATE api_refresh_tokens SET revoked_at = NOW(), revoked_reason = %s "
                    "WHERE username = %s AND revoked_at IS NULL", (str(reason)[:20], username))
        n = cur.rowcount or 0
        conn.commit()
        cur.close()
        return int(n)
    finally:
        conn.close()
