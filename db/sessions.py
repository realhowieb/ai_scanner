from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from db.engine import get_neon_conn
except Exception:
    get_neon_conn = None


def ensure_auth_sessions_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
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


def create_session(username: str, ttl_days: int | None = None) -> Optional[str]:
    if get_neon_conn is None:
        return None
    u = (username or "").strip().lower()
    if not u:
        return None

    if ttl_days is None:
        try:
            from config import SESSION_TTL_DAYS
            ttl_days = SESSION_TTL_DAYS
        except (ImportError, AttributeError):
            ttl_days = 14

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
        (u, expires),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not row:
        return None
    return str(row[0])


def get_username_for_session(session_id: str) -> Optional[str]:
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
    # Atomic: only touch last_seen if still valid; prevents extending expired sessions.
    cur.execute(
        """
        UPDATE auth_sessions
        SET last_seen_at = now()
        WHERE session_id = %s
          AND expires_at > now()
        RETURNING username;
        """,
        (sid,),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return row[0] if row else None


def delete_session(session_id: str) -> None:
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