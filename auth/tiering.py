"""
Tiering module for user access levels.

This module centralizes:
    - USERS_DB fallback (local, non‑database accounts)
    - ADMIN_USERS list
    - Tier dataclass
    - get_user_tier(username) resolver
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

# Load tier configuration from config.py
from config import TIERS_CONFIG


# ------------------------------
# Local fallback user store
# ------------------------------
USERS_DB: Dict[str, Dict[str, str]] = {
    "demo_basic": {
        "username": "demo_basic",
        "full_name": "Demo Basic",
        "tier": "basic",
        "is_active": True,
        "hashed_password": "",
        "created_at": datetime.utcnow().isoformat(),
    },
    "demo_pro": {
        "username": "demo_pro",
        "full_name": "Demo Pro",
        "tier": "pro",
        "is_active": True,
        "hashed_password": "",
        "created_at": datetime.utcnow().isoformat(),
    },
    "demo_premium": {
        "username": "demo_premium",
        "full_name": "Demo Premium",
        "tier": "premium",
        "is_active": True,
        "hashed_password": "",
        "created_at": datetime.utcnow().isoformat(),
    },
}


# ------------------------------
# Admin users
# ------------------------------
ADMIN_USERS = {
    "admin",
    "howard",
}


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
        f"(`{current_label}`). Upgrade to **{required_label}** or higher to use this feature."
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
