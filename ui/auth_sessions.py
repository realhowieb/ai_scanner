"""Cookie-backed auth session helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import streamlit as st


try:
    from streamlit_cookies_manager import EncryptedCookieManager
except ImportError:  # pragma: no cover - optional dependency
    EncryptedCookieManager = None

try:
    from db.engine import get_neon_conn
except ImportError:  # pragma: no cover - optional DB dependency
    get_neon_conn = None


COOKIE_PREFIX = os.environ.get("COOKIE_PREFIX", "ai_scanner")
COOKIE_NAME = os.environ.get("COOKIE_NAME", f"{COOKIE_PREFIX}_sid")
COOKIE_MANAGER_STATE_KEY = "_ai_scanner_cookie_manager"


def _get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except (AttributeError, RuntimeError, KeyError, OSError):
        pass
    return None


def _profile() -> str:
    return (os.environ.get("PROFILE") or os.environ.get("ENV") or "dev").strip().lower()


COOKIE_PASSWORD = _get_secret("COOKIE_PASSWORD")
if not COOKIE_PASSWORD and _profile() in {"dev", "local", "test"}:
    COOKIE_PASSWORD = "dev-only-cookie-password"


def cookies_ready_or_stop() -> Optional["EncryptedCookieManager"]:
    """Return cookie manager if available and initialized."""
    if EncryptedCookieManager is None:
        st.error(
            "Cookie sessions are not available because `streamlit-cookies-manager` is not installed. "
            "Add it to requirements.txt and redeploy."
        )
        return None
    if not COOKIE_PASSWORD:
        st.error("Cookie sessions require COOKIE_PASSWORD to be configured.")
        return None

    cached = st.session_state.get(COOKIE_MANAGER_STATE_KEY)
    if cached is not None:
        return cached

    cookies = EncryptedCookieManager(prefix=COOKIE_PREFIX, password=COOKIE_PASSWORD)
    if not cookies.ready():
        st.stop()
    st.session_state[COOKIE_MANAGER_STATE_KEY] = cookies
    return cookies


def ensure_auth_sessions_schema(conn) -> None:
    """Create auth_sessions table if missing."""
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    except (RuntimeError, OSError, TypeError, ValueError):
        pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
          session_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          username text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          expires_at timestamptz NOT NULL,
          last_seen_at timestamptz NOT NULL DEFAULT now()
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_username ON auth_sessions(username);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);")
    conn.commit()
    cur.close()


def create_session(username: str, ttl_days: int = 14) -> Optional[str]:
    """Create a new session row and return session_id as str."""
    try:
        if get_neon_conn is None:
            return None
        user_key = (username or "").strip().lower()
        if not user_key:
            return None
        conn = get_neon_conn()
        if conn is None:
            return None
        ensure_auth_sessions_schema(conn)

        expires = datetime.now(timezone.utc) + timedelta(days=int(ttl_days))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO auth_sessions (username, expires_at)
            VALUES (%s, %s)
            RETURNING session_id;
            """,
            (user_key, expires),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return str(row[0]) if row else None
    except (RuntimeError, OSError, TypeError, ValueError):
        return None


def get_username_for_session(session_id: str) -> Optional[str]:
    """Return username for a valid, unexpired session_id."""
    try:
        if get_neon_conn is None:
            return None
        sid = (session_id or "").strip()
        if not sid:
            return None
        conn = get_neon_conn()
        if conn is None:
            return None
        ensure_auth_sessions_schema(conn)

        cur = conn.cursor()
        cur.execute(
            """
            SELECT username
            FROM auth_sessions
            WHERE session_id = %s
              AND expires_at > now()
            LIMIT 1;
            """,
            (sid,),
        )
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE auth_sessions SET last_seen_at = now() WHERE session_id = %s;", (sid,))
            conn.commit()
        cur.close()
        conn.close()
        return row[0] if row else None
    except (RuntimeError, OSError, TypeError, ValueError):
        return None


def delete_session(session_id: str) -> None:
    try:
        if get_neon_conn is None:
            return
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = get_neon_conn()
        if conn is None:
            return
        ensure_auth_sessions_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM auth_sessions WHERE session_id = %s;", (sid,))
        conn.commit()
        cur.close()
        conn.close()
    except (RuntimeError, OSError, TypeError, ValueError):
        return
