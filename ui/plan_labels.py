"""Run 85B — the one customer-facing name for each plan.

Internally (database, Stripe mapping, entitlements) the entry tier is `basic`;
customers buy and see it as **Free** (Run 73 packaging: Free / Pro / Premium).
This module only turns an internal tier into display text. Never use it for
comparisons or gating, and never compare against these labels.
"""
from __future__ import annotations

from typing import Any

PLAN_LABELS = {"basic": "Free", "free": "Free", "pro": "Pro", "premium": "Premium", "admin": "Admin"}
DEFAULT_LABEL = "Free"


def _key(tier: Any) -> str:
    """Internal tier key from a key string, a display name or a tier object."""
    if tier is None:
        return ""
    for attr in ("key", "name"):
        val = getattr(tier, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return str(tier).strip().lower()


def plan_label(tier: Any, *, is_admin: bool = False) -> str:
    """Customer-facing plan name: basic/free → Free, pro → Pro, premium → Premium,
    admin (role or tier) → Admin. Unknown or missing → Free (the entry plan)."""
    if is_admin:
        return "Admin"
    return PLAN_LABELS.get(_key(tier), DEFAULT_LABEL)
