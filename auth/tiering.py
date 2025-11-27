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
    "demo_admin",
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
        tier_key = users[username].get("tier", "basic").lower()

    # 2. Local fallback
    elif username in USERS_DB:
        tier_key = USERS_DB[username].get("tier", "basic").lower()

    # Pull tier config
    cfg = TIERS_CONFIG.get(tier_key, TIERS_CONFIG["basic"])

    return Tier(
        key=tier_key,
        name=cfg.get("name", tier_key.capitalize()),
        features=cfg.get("features", []),
        max_results=cfg.get("max_results", 25),
        is_premium=(tier_key == "premium"),
    )
