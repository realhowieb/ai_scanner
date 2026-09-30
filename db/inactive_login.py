"""P2-40 — look up a DEACTIVATED account's stored password.

load_users() skips inactive rows, so a deactivated user used to see
"User not found". Sign-in uses this to say "This account is deactivated"
only after the correct password, so it can't be used to probe which
addresses exist.
"""
from __future__ import annotations

from typing import Optional


def inactive_password(login_key: str) -> Optional[str]:
    """The stored password (hash or legacy text) of an inactive account, else None."""
    key = (login_key or "").strip().lower()
    if not key:
        return None
    try:
        from db.engine import get_neon_conn

        conn = get_neon_conn()
        if conn is None:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT password FROM users WHERE lower(username) = %s AND is_active = FALSE LIMIT 1",
                (key,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            conn.close()
    except Exception:
        return None
    if not row:
        return None
    pw = row[0] if isinstance(row, (tuple, list)) else row.get("password")
    if isinstance(pw, (bytes, bytearray)):
        pw = pw.decode("utf-8", errors="ignore")
    return str(pw) if pw else None


def password_matches(stored: Optional[str], candidates) -> bool:
    """True if any candidate matches the stored bcrypt hash or legacy plain text."""
    if not stored:
        return False
    if stored.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            import bcrypt

            return any(c and bcrypt.checkpw(c.encode("utf-8"), stored.encode("utf-8")) for c in candidates)
        except Exception:
            return False
    return any(c and c == stored for c in candidates)
