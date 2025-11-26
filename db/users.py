from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import streamlit as st

from .engine import get_neon_conn
from .schema import ensure_neon_users_schema
from config import USERS_DB  # local fallback user config

# Optional: auth library + hasher for seeding hashed passwords
try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None

try:
    from streamlit_authenticator.utilities.hasher import Hasher as SAHasher
except Exception:
    SAHasher = None


def seed_neon_users_from_local() -> None:
    """
    If Neon.users is empty, seed it using the hard-coded USERS_DB.
    This runs once, then never again unless Neon is wiped.
    """
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
                except Exception:
                    # Fallback to raw password if hashing fails; not ideal, but avoids seeding failure
                    hashed_pwd = raw_pwd

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
        st.caption(f"⚠️ Neon user seeding failed: {e}")


@st.cache_data(show_spinner=False, ttl=60)
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
        # Final fallback: static USERS_DB snapshot
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