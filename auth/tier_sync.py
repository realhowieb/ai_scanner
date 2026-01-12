"""Tier resolution + Stripe/DB sync (single source of truth).

Goal: keep `app.py` thin.

This module resolves the user's tier using the DB as the source of truth when available,
while remaining compatible with legacy `users_map`/YAML flows.

Return shape is a dict so `app.py` can decide how to store session_state and how to map
string tier keys to any local Enum (Tier) it may define.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


def _norm_tier_key(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def _try_import_user_lookup() -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """Return (callable, name) for a supported DB user lookup function in db.users."""
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
                return fn, name
        return None, None
    except Exception:
        return None, None


def _safe_get_user_from_db(username: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort DB user record lookup.

    Returns: (user_dict_or_none, error_string_or_none)
    """
    fn, fn_name = _try_import_user_lookup()
    if not fn:
        return None, "No supported user lookup function found in db.users. Expected one of: get_user_by_username/get_user_by_email/get_user/get_user_by_identifier"

    try:
        u = fn(username)
        if u is None:
            return None, None
        if isinstance(u, dict):
            return u, None
        # Some implementations may return row objects / tuples; best-effort dict conversion
        if hasattr(u, "_asdict"):
            return u._asdict(), None  # type: ignore[attr-defined]
        return dict(u), None  # type: ignore[arg-type]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def resolve_user_tier(
    username: str,
    *,
    users_map: Optional[Dict[str, Any]] = None,
    session_tier_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve user tier.

    Priority:
      1) DB tier (if DB reachable / lookup available)
      2) users_map[username]['tier'] (legacy)
      3) session_tier_key (as last-resort)
      4) 'basic'

    Returns dict:
      - tier_key: str
      - db_tier: Optional[str]
      - user: Optional[dict]
      - is_admin: bool
      - users_map: (possibly patched) users_map
      - invalidate_entitlements: bool
      - debug: dict (safe-to-render)
    """

    # --- Baselines
    session_key_norm = _norm_tier_key(session_tier_key)

    # --- DB lookup
    user, db_err = _safe_get_user_from_db(username)
    db_tier = _norm_tier_key(user.get("tier") if isinstance(user, dict) else None) if user else None

    # --- Legacy map
    legacy_tier = None
    if isinstance(users_map, dict):
        try:
            legacy_tier = _norm_tier_key((users_map.get(username) or {}).get("tier"))  # type: ignore[union-attr]
        except Exception:
            legacy_tier = None

    forced_tier_key = db_tier or legacy_tier or session_key_norm or "basic"

    # Keep users_map in sync for legacy code paths.
    if isinstance(users_map, dict):
        if username not in users_map:
            users_map[username] = {}
        if isinstance(users_map.get(username), dict):
            users_map[username]["tier"] = forced_tier_key

    prev = session_key_norm or ""
    invalidate_entitlements = bool(prev and forced_tier_key and prev != forced_tier_key)

    # Admin handling: treat 'admin' as full access
    is_admin = forced_tier_key == "admin"

    debug = {
        "username": username,
        "db_tier": db_tier or "-",
        "legacy_tier": legacy_tier or "-",
        "session_tier_key": session_key_norm or "-",
        "resolved_tier_key": forced_tier_key,
        "db_lookup_error": db_err or "-",
    }

    return {
        "tier_key": forced_tier_key,
        "db_tier": db_tier,
        "user": user,
        "is_admin": is_admin,
        "users_map": users_map,
        "invalidate_entitlements": invalidate_entitlements,
        "debug": debug,
    }
