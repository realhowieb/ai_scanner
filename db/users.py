from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
import streamlit as st

from .engine import get_neon_conn
from .schema import ensure_neon_users_schema, ensure_neon_login_attempts_schema


# Optional: auth library + hasher for seeding hashed passwords
try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None

try:
    from streamlit_authenticator.utilities.hasher import Hasher as SAHasher
except Exception:
    SAHasher = None

try:
    import bcrypt
except Exception:
    bcrypt = None


def _demo_users_enabled() -> bool:
    return os.getenv("ENABLE_DEMO_USERS", "").strip().lower() in {"1", "true", "yes", "on"}


def _demo_password(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return None


def seed_neon_users_from_local() -> None:
    """
    If Neon.users is empty, seed it using the opt-in demo USERS_DB.
    This runs once, then never again unless Neon is wiped.
    """
    if not _demo_users_enabled():
        return
    try:
        conn = get_neon_conn()
        if conn is None:
            return

        ensure_neon_users_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
        row = cur.fetchone()
        count = row[0] if row is not None and len(row) > 0 else 0

        # Nothing in table? → seed from USERS_DB
        if count == 0:
            for uname, cfg in USERS_DB.items():
                # Hash passwords before inserting into Neon when auth library is available
                raw_pwd = cfg["password"]
                hashed_pwd = raw_pwd
                try:
                    if SAHasher is not None:
                        # New-style hasher (utilities.hasher.Hasher)
                        hashed_pwd = SAHasher([raw_pwd]).generate()[0]
                    elif stauth is not None and hasattr(stauth, "Hasher"):
                        # Legacy stauth.Hasher if available
                        hashed_pwd = stauth.Hasher([raw_pwd]).generate()[0]
                    elif bcrypt is not None:
                        hashed_pwd = bcrypt.hashpw(raw_pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                except Exception:
                    # Never seed a plain-text fallback password.
                    continue

                cur.execute(
                    """
                    INSERT INTO users (username, full_name, password, tier, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (uname, cfg["name"], hashed_pwd, cfg["tier"], True),
                )
            conn.commit()

        cur.close()
        conn.close()
    except Exception as e:
        # Best-effort seeding; log to UI but do not crash
        # st.caption(f"⚠️ Neon user seeding failed: {e}")
        pass


# Helper: update a user's password in Neon (best-effort, silent fail)
def update_neon_user_password(username: str, hashed_password: str) -> None:
    """
    Update a user's password in Neon to the new bcrypt hash.
    Best-effort: failures are ignored so login still succeeds.
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            return

        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET password = %s WHERE username = %s",
            (hashed_password, username),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # Silent fail — login will still succeed, migration just won't persist
        pass


# Helper: update a user's display name (full_name) in Neon (best-effort, silent fail)
def update_neon_user_full_name(username: str, full_name: str) -> None:
    """Update a user's full_name in Neon. Best-effort: failures are ignored."""
    try:
        uname = (username or "").strip().lower()
        name = (full_name or "").strip()
        if not uname or not name:
            return

        conn = get_neon_conn()
        if conn is None:
            return

        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET full_name = %s WHERE username = %s",
            (name, uname),
        )
        conn.commit()
        cur.close()
        conn.close()

        # Clear cached users so login sees updated full_name immediately
        try:
            load_users.clear()  # type: ignore[attr-defined]
        except Exception:
            try:
                st.cache_data.clear()
            except Exception:
                pass
    except Exception:
        pass


def create_user_account(
    email: str,
    password_hash: str,
    tier: str = "basic",
    full_name: str | None = None,
) -> Dict[str, Any]:
    """Create a new user in Neon (preferred) and return a user record.

    This is used by the Sign Up flow. Users always start as `basic` unless an admin
    later upgrades them.

    Args:
        email: Email/username (stored as `username` in DB)
        password_hash: bcrypt hash string
        tier: initial tier (defaults to 'basic')
        full_name: optional display name

    Returns:
        dict with keys at least: username, display_name, tier, is_admin

    Raises:
        ValueError if email is invalid/empty or already exists.
        RuntimeError if Neon is unavailable.
    """
    username = (email or "").strip().lower()
    if not username:
        raise ValueError("email is required")

    # Normalize tier (defensive)
    tier_key = (tier or "basic").strip().lower()
    if tier_key not in {"basic", "pro", "premium", "admin"}:
        tier_key = "basic"

    display_name = (full_name or username).strip() or username

    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon DB is not available")

    ensure_neon_users_schema(conn)
    cur = conn.cursor()

    # Ensure uniqueness
    cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
    if cur.fetchone() is not None:
        cur.close()
        conn.close()
        raise ValueError("user already exists")

    # Insert (created_at is handled by schema default if present)
    cur.execute(
        """
        INSERT INTO users (username, full_name, password, tier, is_active)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (username, display_name, password_hash, tier_key, True),
    )
    conn.commit()

    # Ensure auth sees the new user immediately (load_users is cached)
    try:
        load_users.clear()  # type: ignore[attr-defined]
    except Exception:
        try:
            st.cache_data.clear()
        except Exception:
            pass

    # Fetch the record back (best-effort)
    cur.execute(
        """
        SELECT username, full_name, tier, is_active
        FROM users
        WHERE username = %s
        """,
        (username,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    # Cursor rows can be tuple-like or dict-like depending on cursor factory
    if isinstance(row, tuple):
        u = row[0]
        n = row[1]
        t = (row[2] or "basic").strip().lower()
    else:
        u = row.get("username")
        n = row.get("full_name")
        t = (row.get("tier") or "basic").strip().lower()

    return {
        "username": u or username,
        "display_name": (n or u or username),
        "tier": t,
        "is_admin": t == "admin",
    }


def create_neon_user(*args, **kwargs) -> Dict[str, Any]:
    """Backward-compatible alias for older call sites."""
    return create_user_account(*args, **kwargs)


def find_username_by_display_name(display_name: str) -> str | None:
    """Return the account username (email key) for a given display name (full_name).

    This enables login via the user-chosen Username while keeping email as the primary key.
    Matching is case-insensitive.
    """
    try:
        name = (display_name or "").strip()
        if not name:
            return None

        conn = get_neon_conn()
        if conn is None:
            return None

        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT username
            FROM users
            WHERE is_active = TRUE
              AND LOWER(TRIM(full_name)) = LOWER(TRIM(%s))
            ORDER BY created_at DESC NULLS LAST
            LIMIT 1
            """,
            (name,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return None
        return row[0] if isinstance(row, tuple) else row.get("username")
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=30)
def load_users() -> Dict[str, Dict[str, Any]]:
    """
    Load users from Neon if available; otherwise fall back to the local USERS_DB.

    Returns:
        Mapping of username -> { "name": full_name, "password": hashed_or_plain, "tier": tier }
    """
    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            ensure_neon_users_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT username, full_name, password, tier, is_active
                FROM users
                """
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()

            data: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                # psycopg2 rows can be tuple-like or dict-like; handle both
                username = r[0] if isinstance(r, tuple) else r["username"]
                full_name = r[1] if isinstance(r, tuple) else r["full_name"]
                password = r[2] if isinstance(r, tuple) else r["password"]
                tier = r[3] if isinstance(r, tuple) else r["tier"]
                is_active = r[4] if isinstance(r, tuple) else r["is_active"]

                if not is_active:
                    continue

                data[username] = {
                    "name": full_name,
                    "password": password,
                    "tier": tier or "basic",
                }

            if data:
                return data
    except Exception as e:
        st.caption(f"⚠️ Failed to load users from Neon, using local USERS_DB: {e}")

    # Fallback: local USERS_DB
    return {
        uname: {
            "name": cfg["name"],
            "password": cfg["password"],
            "tier": cfg.get("tier", "basic"),
        }
        for uname, cfg in USERS_DB.items()
    }


def fetch_all_users() -> pd.DataFrame:
    """
    Fetch all users for admin display.

    Returns a DataFrame with at least:
    [username, full_name, tier, is_active, created_at]
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            # Fallback: build DataFrame from local USERS_DB
            data = [
                {
                    "username": uname,
                    "full_name": cfg["name"],
                    "tier": cfg.get("tier", "basic"),
                    "is_active": True,
                    "created_at": None,
                }
                for uname, cfg in USERS_DB.items()
            ]
            return pd.DataFrame(data)

        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT username, full_name, tier, is_active, created_at
            FROM users
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return pd.DataFrame(
                columns=["username", "full_name", "tier", "is_active", "created_at"]
            )

        # psycopg2 rows → build DataFrame
        data = []
        for r in rows:
            if isinstance(r, tuple):
                username, full_name, tier, is_active, created_at = r
            else:
                username = r["username"]
                full_name = r["full_name"]
                tier = r["tier"]
                is_active = r["is_active"]
                created_at = r["created_at"]
            data.append(
                {
                    "username": username,
                    "full_name": full_name,
                    "tier": tier or "basic",
                    "is_active": bool(is_active),
                    "created_at": created_at,
                }
            )

        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to fetch users from Neon: {e}")
        if not _demo_users_enabled():
            return pd.DataFrame(
                columns=["username", "full_name", "tier", "is_active", "created_at"]
            )
        # Final fallback: opt-in static demo users.
        data = [
            {
                "username": uname,
                "full_name": cfg["name"],
                "tier": cfg.get("tier", "basic"),
                "is_active": True,
                "created_at": None,
            }
            for uname, cfg in USERS_DB.items()
        ]
        return pd.DataFrame(data)
def _build_demo_users() -> Dict[str, Dict[str, str]]:
    if not _demo_users_enabled():
        return {}

    configured = {
        "basic1": ("Basic Demo User", _demo_password("DEMO_BASIC_PASSWORD"), "basic"),
        "pro1": ("Pro Demo User", _demo_password("DEMO_PRO_PASSWORD"), "pro"),
        "premium1": ("Premium Demo User", _demo_password("DEMO_PREMIUM_PASSWORD"), "premium"),
        "admin1": ("Admin Demo User", _demo_password("DEMO_ADMIN_PASSWORD"), "admin"),
    }
    return {
        username: {"name": name, "password": password, "tier": tier}
        for username, (name, password, tier) in configured.items()
        if password
    }


# Optional local fallback user config. Disabled unless ENABLE_DEMO_USERS=1 and
# passwords are supplied through DEMO_*_PASSWORD env vars or Streamlit secrets.
USERS_DB = _build_demo_users()


def is_admin_from_db(username: str) -> bool:
    """Server-side admin check: queries DB is_admin flag and tier='admin'.

    This is the authoritative check — do not rely solely on session state.
    Returns False if DB is unreachable (safe default).
    """
    username_norm = (username or "").strip().lower()
    if not username_norm:
        return False
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT is_admin, tier FROM users WHERE lower(username) = %s AND is_active = TRUE LIMIT 1",
            (username_norm,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return False
        if isinstance(row, dict):
            return bool(row.get("is_admin")) or str(row.get("tier", "")).lower() == "admin"
        return bool(row[0]) or str(row[1] or "").lower() == "admin"
    except Exception:
        return False


def grant_admin(username: str) -> bool:
    """Promote a user to admin in the DB. Returns True on success."""
    username_norm = (username or "").strip().lower()
    if not username_norm:
        return False
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET is_admin = TRUE, tier = 'admin' WHERE lower(username) = %s",
            (username_norm,),
        )
        conn.commit()
        updated = cur.rowcount > 0
        cur.close()
        conn.close()
        try:
            load_users.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        return updated
    except Exception:
        return False


def revoke_admin(username: str) -> bool:
    """Remove admin flag from a user. Returns True on success."""
    username_norm = (username or "").strip().lower()
    if not username_norm:
        return False
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET is_admin = FALSE, tier = 'basic' WHERE lower(username) = %s",
            (username_norm,),
        )
        conn.commit()
        updated = cur.rowcount > 0
        cur.close()
        conn.close()
        try:
            load_users.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        return updated
    except Exception:
        return False


# --- Login rate limiting ---
from config import LOGIN_RATE_LIMIT_WINDOW_SEC as _LOGIN_WINDOW_SECONDS, LOGIN_RATE_LIMIT_MAX_ATTEMPTS as _LOGIN_MAX_ATTEMPTS


def record_login_attempt(username: str, *, success: bool, ip_address: str | None = None) -> None:
    """Persist a login attempt for rate-limit tracking. Best-effort; never raises."""
    username_norm = (username or "").strip().lower()
    if not username_norm:
        return
    try:
        conn = get_neon_conn()
        if conn is None:
            return
        ensure_neon_login_attempts_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO login_attempts (username, success, ip_address) VALUES (%s, %s, %s)",
            (username_norm, success, ip_address),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def is_login_rate_limited(username: str) -> bool:
    """Return True if the user has exceeded failed login attempts in the window.

    Counts only failed attempts in the last _LOGIN_WINDOW_SECONDS seconds.
    Returns False (allow) when DB is unavailable so auth still works offline.
    """
    username_norm = (username or "").strip().lower()
    if not username_norm:
        return False
    try:
        conn = get_neon_conn()
        if conn is None:
            return False
        ensure_neon_login_attempts_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE username = %s
              AND success = FALSE
              AND attempted_at > NOW() - INTERVAL '%s seconds'
            """,
            (username_norm, _LOGIN_WINDOW_SECONDS),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        count = row[0] if row else 0
        return int(count) >= _LOGIN_MAX_ATTEMPTS
    except Exception:
        return False
