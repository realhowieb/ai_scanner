"""
Tiering module for user access levels.

This module centralizes:
    - USERS_DB fallback (local, non‑database accounts)
    - ADMIN_USERS list
    - Tier dataclass
    - get_user_tier(username) resolver
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    import streamlit as st
except Exception:
    class _StShim:
        @staticmethod
        def warning(*_a, **_kw): pass
        @staticmethod
        def get(*_a, **_kw): return None
        session_state: dict = {}
    st = _StShim()  # type: ignore[assignment]

# Load tier configuration from config.py
from config import TIERS_CONFIG


# ------------------------------
# Local fallback user store
# ------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


USERS_DB: Dict[str, Dict[str, str]] = {
    "demo_basic": {
        "username": "demo_basic",
        "full_name": "Demo Basic",
        "tier": "basic",
        "is_active": True,
        "hashed_password": "",
        "created_at": _utc_now_iso(),
    },
    "demo_pro": {
        "username": "demo_pro",
        "full_name": "Demo Pro",
        "tier": "pro",
        "is_active": True,
        "hashed_password": "",
        "created_at": _utc_now_iso(),
    },
    "demo_premium": {
        "username": "demo_premium",
        "full_name": "Demo Premium",
        "tier": "premium",
        "is_active": True,
        "hashed_password": "",
        "created_at": _utc_now_iso(),
    },
}


# ------------------------------
# Admin users — loaded from secrets/env, NOT hardcoded
# ------------------------------

def _load_admin_users() -> set[str]:
    """Load admin usernames from st.secrets or ADMIN_USERS env var.

    Format: comma-separated list, e.g. "alice,bob"
    Falls back to empty set if not configured (DB is_admin flag is the real source of truth).
    """
    raw: str | None = None
    try:
        import streamlit as _st
        raw = _st.secrets.get("ADMIN_USERS") or _st.secrets.get("admin_users")
    except Exception:
        pass
    if not raw:
        import os as _os
        raw = _os.environ.get("ADMIN_USERS", "")
    if not raw:
        return set()
    return {u.strip().lower() for u in str(raw).split(",") if u.strip()}


ADMIN_USERS: set[str] = _load_admin_users()


# ------------------------------
# Tier object
# ------------------------------
@dataclass
class Tier:
    key: str
    name: str
    features: list
    max_results: int
    is_premium: bool = False
    can_premarket: bool = False
    can_afterhours: bool = False
    can_unusual_volume: bool = False
    can_export_csv: bool = False
    can_ai_notes: bool = False


# ------------------------------
# Tier comparison helpers
# ------------------------------

# Rank order for tiers from lowest to highest privilege
TIER_ORDER = {
    "basic": 0,
    "pro": 1,
    "premium": 2,
    "admin": 3,
}


def _normalize_tier_key(tier_or_key) -> str:
    """
    Normalize a Tier instance or string into a lowercase tier key.
    Falls back to 'basic' when unknown.
    """
    if isinstance(tier_or_key, Tier):
        key = tier_or_key.key
    else:
        key = str(tier_or_key or "basic")
    return key.strip().lower()


def has_min_tier(tier_or_key, required: str) -> bool:
    """
    Return True if the current tier is >= the required tier based on TIER_ORDER.

    Example:
        has_min_tier(current_tier, "pro")
    """
    current_key = _normalize_tier_key(tier_or_key)
    required_key = str(required or "basic").strip().lower()
    current_rank = TIER_ORDER.get(current_key, 0)
    required_rank = TIER_ORDER.get(required_key, 0)
    return current_rank >= required_rank


def require_min_tier(tier_or_key, required: str, feature_name: str) -> bool:
    """
    UI-friendly helper: checks if the user has the required tier and,
    if not, shows a warning in the Streamlit UI.

    Returns True if allowed, False if blocked.

    Example:
        if not require_min_tier(user_tier, "premium", "Premarket Scanner"):
            return
    """
    if has_min_tier(tier_or_key, required):
        return True

    # Lazy import to avoid hard dependency at module load time
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return False

    current_key = _normalize_tier_key(tier_or_key)
    required_key = str(required or "basic").strip().lower()

    current_label = current_key.capitalize()
    required_label = required_key.capitalize()

    st.warning(
        f"🚫 **{feature_name}** is not available on your current plan "
        f"(`{current_label}`). Upgrade to **{required_label}** to use this feature."
    )
    return False


# ------------------------------
# Tier resolver
# ------------------------------
def get_user_tier(username: str, users: Optional[Dict[str, Dict[str, str]]] = None) -> Tier:
    """
    Determine which tier a user has.
    Priority:
        1. Neon users table (passed as `users`)
        2. Local USERS_DB fallback
        3. Default to Basic tier
    """
    tier_key = "basic"

    # 1. Neon DB results (if provided)
    if users and username in users:
        raw_tier = users[username].get("tier", "basic")
        if raw_tier is not None:
            tier_key = str(raw_tier).strip().lower()

    # 2. Local fallback
    elif username in USERS_DB:
        raw_tier = USERS_DB[username].get("tier", "basic")
        if raw_tier is not None:
            tier_key = str(raw_tier).strip().lower()

    # Ensure we only use known tiers; otherwise, treat as basic
    cfg = TIERS_CONFIG.get(tier_key)
    if cfg is None:
        tier_key = "basic"
        cfg = TIERS_CONFIG["basic"]

    features = cfg.get("features", [])
    # Normalize feature names for matching (case-insensitive)
    features_lower = [str(f).lower() for f in features]

    return Tier(
        key=tier_key,
        name=cfg.get("name", tier_key.capitalize()),
        features=features,
        max_results=cfg.get("max_results", 25),
        is_premium=(tier_key == "premium"),
        can_premarket=("premarket" in features_lower),
        can_afterhours=("afterhours" in features_lower),
        can_unusual_volume=("unusualvolume" in features_lower),
        can_export_csv=("exportcsv" in features_lower),
        can_ai_notes=("ai notes" in features_lower),
    )

def render_pricing_section():
    # Resolve the current tier name
    current_tier_raw = st.session_state.get("tier", None)
    if hasattr(current_tier_raw, "key"):
        current_tier = current_tier_raw.key.lower()
    elif isinstance(current_tier_raw, str):
        current_tier = current_tier_raw.lower()
    else:
        current_tier = "basic"

    # ... (other pricing cards, e.g., Basic)

    # Pro pricing card
    # ... (Pro card layout and feature bullets)
    # Button logic for Pro
    if current_tier in ("pro", "premium", "admin"):
        # User already has Pro or better – show a disabled indicator instead of a subscribe button.
        st.button("Included in your plan", key="pro_included", disabled=True)
    else:
        st.button("Subscribe Pro (Monthly)", key="subscribe_pro_monthly")

    # Premium pricing card
    # ... (Premium card layout and feature bullets)
    # Button logic for Premium
    if current_tier in ("premium", "admin"):
        # User is already on Premium (or higher) – show a disabled indicator instead of a subscribe button.
        st.button("Current plan", key="premium_current", disabled=True)
    else:
        st.button("Subscribe Premium (Monthly)", key="subscribe_premium_monthly")
