"""Tier resolution + Stripe/DB sync.

DB is the source of truth when available; falls back to legacy users_map/session.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _norm_tier_key(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def _safe_get_user_from_db(username: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort DB user record lookup via db.users helper functions.

    Returns: (user_dict_or_none, error_string_or_none)
    """
    try:
        import importlib

        m = importlib.import_module("db.users")
        for name in (
            "get_user_by_username",
            "get_user_by_email",
            "get_user",
            "get_user_by_identifier",
        ):
            fn = getattr(m, name, None)
            if callable(fn):
                u = fn(username)
                if u is None:
                    return None, None
                if isinstance(u, dict):
                    return u, None
                if hasattr(u, "_asdict"):
                    return u._asdict(), None  # type: ignore[attr-defined]
                return dict(u), None  # type: ignore[arg-type]
        return (
            None,
            "No supported user lookup function found in db.users. Expected one of: get_user_by_username/get_user_by_email/get_user/get_user_by_identifier",
        )
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
def resolve_user_tier(
    username: str,
    *,
    users_map: Optional[Dict[str, Any]] = None,
    session_tier_key: Optional[str] = None,
    # New app.py contract (extras are optional; unknown kwargs are ignored)
    Tier: Any = None,
    get_user_tier: Any = None,
    get_db_conn: Any = None,
    admin_users: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Resolve user tier.

    Backward compatible with older calls:
      resolve_user_tier(username, users_map=..., session_tier_key=...)

    Also supports the newer app.py contract:
      resolve_user_tier(username=..., users_map=..., Tier=..., get_user_tier=..., get_db_conn=..., admin_users=...)

    Priority:
      1) Admin override (if username in admin_users)
      2) DB tier (if DB reachable / lookup available)
      3) users_map[username]['tier'] (legacy)
      4) session_tier_key (last-resort)
      5) 'basic'

    Returns a dict that app.py can normalize. Important keys:
      - tier_obj: Enum/member or legacy tier object when possible
      - tier_key: str
      - forced_tier_key: Optional[str]
      - db_user_debug: Optional[dict]
      - db_tier_err: Optional[str]

    Additional keys are included for compatibility/debugging.
    """

    username_norm = (username or "").strip().lower()
    session_key_norm = _norm_tier_key(session_tier_key)

    # ---- Admin override
    try:
        admins = {str(u).strip().lower() for u in (admin_users or [])}
        if username_norm and username_norm in admins:
            forced = "admin"
            tier_obj = None
            if Tier is not None:
                try:
                    if hasattr(Tier, "__members__") and "ADMIN" in getattr(Tier, "__members__"):
                        tier_obj = Tier["ADMIN"]
                    else:
                        tier_obj = Tier(forced)
                except Exception:
                    tier_obj = None
            return {
                "tier_obj": tier_obj,
                "tier_key": forced,
                "forced_tier_key": forced,
                "db_user_debug": {"username": username_norm, "tier": forced, "source": "admin_users"},
                "db_tier_err": None,
                "tier": tier_obj,
                "db_tier": forced,
                "user": {"username": username_norm, "tier": forced},
                "is_admin": True,
                "users_map": users_map,
                "invalidate_entitlements": bool(session_key_norm and session_key_norm != forced),
                "debug": {
                    "username": username_norm,
                    "db_tier": forced,
                    "legacy_tier": "-",
                    "session_tier_key": session_key_norm or "-",
                    "resolved_tier_key": forced,
                    "db_lookup_error": "-",
                    "source": "admin_users",
                },
            }
    except Exception as _e:
        print(f"[tier_sync] admin fast-path error: {type(_e).__name__}: {_e}")

    # ---- DB lookup (prefer direct DB conn when provided, otherwise fall back to db.users helpers)
    user: Optional[Dict[str, Any]] = None
    db_err: Optional[str] = None
    db_tier: Optional[str] = None

    # 1) Direct query via get_db_conn()
    if callable(get_db_conn) and username_norm:
        try:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    # Prefer username match; if your schema uses email in username, this still works.
                    cur.execute(
                        "SELECT tier FROM users WHERE lower(username)=lower(%s) LIMIT 1",
                        (username_norm,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        db_tier = _norm_tier_key(row[0])
                        user = {"username": username_norm, "tier": db_tier}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            db_err = f"{type(e).__name__}: {e}"

    # 2) Fallback: db.users lookup helpers
    if db_tier is None and username_norm:
        user2, db_err2 = _safe_get_user_from_db(username_norm)
        if user2 is not None:
            user = user2
            db_tier = _norm_tier_key(user2.get("tier"))
        if db_err is None and db_err2:
            db_err = db_err2

    # ---- Legacy map
    legacy_tier: Optional[str] = None
    if isinstance(users_map, dict) and username_norm:
        try:
            legacy_tier = _norm_tier_key((users_map.get(username_norm) or {}).get("tier"))  # type: ignore[union-attr]
        except Exception:
            legacy_tier = None

    forced_tier_key = db_tier or legacy_tier or session_key_norm or "basic"

    # Keep users_map in sync for legacy code paths.
    if isinstance(users_map, dict) and username_norm:
        if username_norm not in users_map:
            users_map[username_norm] = {}
        if isinstance(users_map.get(username_norm), dict):
            users_map[username_norm]["tier"] = forced_tier_key

    prev = session_key_norm or ""
    invalidate_entitlements = bool(prev and forced_tier_key and prev != forced_tier_key)

    # Build a tier_obj if Tier enum is provided; otherwise fall back to legacy get_user_tier()
    tier_obj = None
    if Tier is not None:
        try:
            if hasattr(Tier, "__members__") and forced_tier_key.upper() in getattr(Tier, "__members__"):
                tier_obj = Tier[forced_tier_key.upper()]
            else:
                tier_obj = Tier(forced_tier_key)
        except Exception:
            tier_obj = None

    if tier_obj is None and callable(get_user_tier):
        try:
            tier_obj = get_user_tier(username_norm, users_map or {})
        except Exception:
            tier_obj = None

    debug = {
        "username": username_norm,
        "db_tier": db_tier or "-",
        "legacy_tier": legacy_tier or "-",
        "session_tier_key": session_key_norm or "-",
        "resolved_tier_key": forced_tier_key,
        "db_lookup_error": db_err or "-",
        "source": "db" if db_tier else ("users_map" if legacy_tier else ("session" if session_key_norm else "default")),
    }

    return {
        # New normalized keys app.py expects
        "tier_obj": tier_obj,
        "tier_key": forced_tier_key,
        "forced_tier_key": db_tier or legacy_tier or session_key_norm,
        "db_user_debug": user,
        "db_tier_err": db_err,

        # Back-compat keys used elsewhere
        "tier": tier_obj,
        "db_tier": db_tier,
        "user": user,
        "is_admin": forced_tier_key == "admin",
        "users_map": users_map,
        "invalidate_entitlements": invalidate_entitlements,
        "debug": debug,
    }
