"""Session, tier, and entitlement helpers for the Streamlit app."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

FEATURE_MIN_TIER: dict[str, str] = {
    "can_scan_sp500": "basic",
    "can_scan_nasdaq": "pro",
    "can_export_csv": "pro",
    "can_earnings": "pro",
    "can_ai_notes": "premium",
    "can_scan_history": "pro",
    "can_early_breakout": "premium",
    "can_full_universe": "premium",
    "can_diagnostics": "admin",
    "can_admin_panel": "admin",
}

TIER_ORDER = {
    "basic": 0,
    "pro": 1,
    "premium": 2,
    "admin": 3,
}


def norm_str(value: object | None) -> str:
    """Normalize user-provided or DB-provided strings to a safe canonical form."""
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def norm_lower(value: object | None) -> str:
    return norm_str(value).lower()


def normalize_admin_users(admin_users: object) -> set[str]:
    """Normalize configured admin users to lowercase usernames."""
    try:
        if isinstance(admin_users, (list, set, tuple)):
            return {str(user).strip().lower() for user in admin_users}
        if isinstance(admin_users, dict):
            return {str(user).strip().lower() for user in admin_users.keys()}
    except Exception:
        pass
    return set()


def is_admin_user(
    username: str | None,
    tier_obj: object | None,
    *,
    admin_users: object,
) -> bool:
    """Admin check that is resilient to whitespace, case, and tier-object shape."""
    username_norm = norm_lower(username)
    if username_norm and username_norm in normalize_admin_users(admin_users):
        return True

    try:
        if norm_lower(getattr(tier_obj, "key", None)) == "admin":
            return True
    except Exception:
        pass

    try:
        if norm_lower(getattr(tier_obj, "name", None)) == "admin":
            return True
    except Exception:
        pass

    return norm_lower(tier_obj) == "admin"


def tier_key(tier_obj: object | None) -> str:
    """Return a stable tier key string for logging, comparisons, and UI state."""
    try:
        key = getattr(tier_obj, "key", None)
        if key is not None:
            return norm_lower(key)
    except Exception:
        pass

    try:
        name = getattr(tier_obj, "name", None)
        if name is not None:
            return norm_lower(name)
    except Exception:
        pass

    return norm_lower(tier_obj) or "basic"


def _fallback_has_min_tier(tier_obj: object | None, required: str) -> bool:
    current_rank = TIER_ORDER.get(tier_key(tier_obj), 0)
    required_rank = TIER_ORDER.get(norm_lower(required) or "basic", 0)
    return current_rank >= required_rank


def compute_entitlements(
    *,
    tier_obj: object | None,
    is_admin: bool,
    has_min_tier_fn: Callable[[Any, str], bool] | None = None,
) -> dict[str, bool]:
    """Compute deterministic feature flags from tier state."""
    if bool(is_admin):
        return {feature: True for feature in FEATURE_MIN_TIER}

    if has_min_tier_fn is None:
        has_min_tier_fn = _fallback_has_min_tier

    flags: dict[str, bool] = {}
    for feature, min_tier in FEATURE_MIN_TIER.items():
        if min_tier == "admin":
            flags[feature] = False
            continue
        try:
            flags[feature] = bool(has_min_tier_fn(tier_obj, min_tier))
        except Exception:
            flags[feature] = False
    return flags
