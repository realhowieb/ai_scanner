"""Canonical plan limits and feature rules; no UI or database dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

FEATURE_MIN_TIER: dict[str, str] = {
    "can_scan_sp500": "basic",
    "can_scan_nasdaq": "pro",
    "can_premarket": "pro",
    "can_afterhours": "pro",
    "can_unusual_volume": "pro",
    "can_export_csv": "pro",
    "can_earnings": "pro",
    "can_ai_notes": "premium",
    "can_scan_history": "pro",
    "can_track_record": "pro",
    "can_early_breakout": "premium",
    "can_full_universe": "premium",
    "can_paper_trade": "premium",
    "can_email_alerts": "pro",  # in-app alerts are open to all; email is Pro+
    "can_day_trader": "pro",  # live intraday monitor (pages/day_trader.py)
    "can_diagnostics": "admin",
    "can_admin_panel": "admin",
}

ALERT_LIMIT_BY_TIER: dict[str, int] = {
    "basic": 1,
    "pro": 5,
    "premium": 25,
    "admin": 25,
}


def alert_limit_for_tier(tier_key: object | None) -> int:
    """Resolve the max-alerts cap for a tier key (defaults to Basic = 1)."""
    key = str(tier_key or "basic").strip().lower()
    return ALERT_LIMIT_BY_TIER.get(key, 1)


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


has_min_tier = _fallback_has_min_tier
