"""Fallback Tier class and get_user_tier logic.
Used only when main tiering module fails to import.
"""

from dataclasses import dataclass
from typing import Any, Dict

# Default fallback DB (empty)
USERS_DB: Dict[str, Dict[str, Any]] = {}
ADMIN_USERS = set()


@dataclass
class Tier:
    name: str = "Basic"
    max_results: int = 25
    can_scan_sp500: bool = True
    can_scan_nasdaq: bool = False
    can_premarket: bool = False
    can_afterhours: bool = False
    can_unusual_volume: bool = False
    can_export_csv: bool = False
    can_ai_notes: bool = False
    features: list = None


def get_user_tier(username: str, users: Dict[str, Dict[str, Any]]):
    """Fallback tier resolution.
    If user exists in USERS_DB, return its tier.
    Otherwise defaults to Basic tier.
    """
    try:
        user = users.get(username)
        if user and "tier" in user:
            name = user["tier"]
            # Could expand with more tiers later if desired
            if name.lower() == "premium":
                return Tier(name="Premium", max_results=200, can_scan_nasdaq=True,
                            can_premarket=True, can_afterhours=True,
                            can_unusual_volume=True, can_export_csv=True,
                            can_ai_notes=True)
            if name.lower() == "pro":
                return Tier(name="Pro", max_results=100, can_scan_nasdaq=True,
                            can_premarket=True, can_afterhours=False,
                            can_unusual_volume=True, can_export_csv=True,
                            can_ai_notes=True)
            if name.lower() == "basic":
                return Tier(name="Basic", max_results=25, can_scan_nasdaq=False)
    except Exception:
        pass

    # Default fallback
    return Tier()
