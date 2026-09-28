"""P1-37 / P1-38 — admin account operations for the Admin Users page.

Usernames double as email addresses (every email job sends to the username),
so new accounts must have a real, lowercase address. Passwords are always
stored as bcrypt hashes, the same as self-service sign-up.

The admin role (users.is_admin, or the legacy tier 'admin') is separate from
the plan (tier): granting admin never changes the plan.

Every function returns a (ok, message) pair or data and never raises, so the
page can show a plain result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .engine import get_neon_conn
from .schema import ensure_neon_users_schema

PLAN_TIERS = ("basic", "pro", "premium")
_EMAIL_RE = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")


def normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def is_valid_email(value: Any) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(value)))


def _conn():
    conn = get_neon_conn()
    if conn is None:
        return None
    ensure_neon_users_schema(conn)
    from .email_verification import _ensure_schema  # adds users.email_verified if missing

    _ensure_schema(conn)
    return conn


def _audit(actor: Any, action: str, target: Any, detail: Dict[str, Any] | None = None) -> None:
    """P1-42: record a successful admin action; never raises."""
    try:
        from .admin_audit import record_admin_event

        record_admin_event(actor, action, target, detail)
    except Exception:
        pass


def _clear_user_cache() -> None:
    try:
        from .users import load_users

        load_users.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def create_user(email: Any, full_name: Any, password: str, tier: str = "basic",
                active: bool = True, *, actor: Any = None) -> Tuple[bool, str]:
    """Create an account whose username is a valid lowercase email address."""
    username = normalize_email(email)
    name = str(full_name or "").strip()
    if not username or not name or not password:
        return False, "Email, full name and password are all required."
    if not is_valid_email(username):
        return False, "The username must be a valid email address (it's where the account's emails go)."
    if tier not in PLAN_TIERS:
        return False, "Choose a plan: basic, pro or premium."
    try:
        conn = _conn()
        if conn is None:
            return False, "Database unavailable; the account was not created."
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE lower(username) = %s LIMIT 1", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return False, f"An account for {username} already exists."
        try:
            import bcrypt

            pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        except Exception:
            cur.close()
            conn.close()
            return False, "Couldn't hash the password; the account was not created."
        cur.execute(
            "INSERT INTO users (username, full_name, password, tier, is_active) VALUES (%s, %s, %s, %s, %s)",
            (username, name, pw_hash, tier, bool(active)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        return False, f"Failed to create the account ({type(exc).__name__})."
    _clear_user_cache()
    _audit(actor, "create_user", username, {"tier": tier, "active": bool(active)})
    return True, f"Created {username}. Their email isn't verified yet: resend verification or mark it verified."


def fetch_users_for_admin() -> List[Dict[str, Any]]:
    """Accounts with plan, active, admin and verification status, newest first."""
    try:
        conn = _conn()
        if conn is None:
            return []
        cur = conn.cursor()
        cur.execute(
            """
            SELECT username, full_name, tier, is_active,
                   COALESCE(is_admin, FALSE) OR lower(COALESCE(tier, '')) = 'admin',
                   COALESCE(email_verified, FALSE), created_at
            FROM users ORDER BY created_at DESC NULLS LAST
            """
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        return []
    keys = ("username", "full_name", "tier", "is_active", "is_admin", "email_verified", "created_at")
    out = []
    for r in rows:
        vals = list(r.values()) if isinstance(r, dict) else list(r)
        out.append(dict(zip(keys, vals)))
    return out


def _update(sql: str, params: tuple) -> bool:
    try:
        conn = _conn()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute(sql, params)
        ok = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        return False
    _clear_user_cache()
    return ok


def mark_email_verified(username: Any, *, actor: Any = None) -> Tuple[bool, str]:
    u = normalize_email(username)
    if not is_valid_email(u):
        return False, "Only accounts with an email username can be verified."
    if _update("UPDATE users SET email_verified = TRUE WHERE lower(username) = %s", (u,)):
        _audit(actor, "mark_email_verified", u)
        return True, f"{u} is now marked verified."
    return False, "Couldn't update the account."


def set_admin(username: Any, make_admin: bool, *, acting_user: Any) -> Tuple[bool, str]:
    """Grant or revoke the admin role. The plan is left alone, except that a
    legacy tier of 'admin' becomes 'basic' on revoke (otherwise the account
    would still be admin). An admin can't revoke their own role."""
    u = normalize_email(username)
    if not u:
        return False, "No account selected."
    if not make_admin and u == normalize_email(acting_user):
        return False, "You can't remove your own admin role."
    if make_admin:
        ok = _update("UPDATE users SET is_admin = TRUE WHERE lower(username) = %s", (u,))
        if ok:
            _audit(acting_user, "grant_admin", u)
        return (True, f"{u} is now an admin.") if ok else (False, "Couldn't update the account.")
    ok = _update(
        "UPDATE users SET is_admin = FALSE, "
        "tier = CASE WHEN lower(COALESCE(tier, '')) = 'admin' THEN 'basic' ELSE tier END "
        "WHERE lower(username) = %s",
        (u,),
    )
    if ok:
        _audit(acting_user, "revoke_admin", u)
    return (True, f"{u} is no longer an admin.") if ok else (False, "Couldn't update the account.")
