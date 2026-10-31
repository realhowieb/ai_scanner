"""Founder value meter and the 5-year comparison, without the API stack, so the morning
digest (GitHub Actions, no FastAPI) and the API share one calculation."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from domain import founding as rules


def monthly_cents(plan: str) -> int:
    """The monthly price of an HSF plan in cents, from the same source as the pricing page
    (ui.pricing.PRICES, e.g. "$25/mo")."""
    from ui import pricing

    raw = str(pricing.PRICES.get(plan, "")).split("/")[0].replace("$", "").replace(",", "").strip()
    try:
        return int(round(float(raw) * 100))
    except ValueError:
        return 0


def five_year(tier: str) -> Dict[str, int]:
    return rules.five_year_math(tier, monthly_cents(rules.PLAN_FOR_TIER[tier]))


def for_membership(m: Dict[str, Any], user: Optional[str] = None,
                   now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """Months used of 60, what they would have cost monthly, and for Founder Elite (when
    `user` is given) how many of the 50 alerts are in use."""
    from db.founding import parse_ts

    start = parse_ts(m.get("activated_at"))
    if not start or m.get("tier") not in rules.TIERS:
        return None
    out: Dict[str, Any] = rules.value_meter(m["tier"], start, monthly_cents(rules.PLAN_FOR_TIER[m["tier"]]), now)
    out["alerts_used"] = None
    if user and m["tier"] == rules.FOUNDER_ELITE:
        try:
            from db import alert_rules

            out["alerts_used"] = int(alert_rules.count_active(user))
        except Exception:
            pass
    return out


def for_user(user: str, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """The value meter for an active member, else None."""
    from db import founding as db

    m = db.membership(user)
    if not m or m["status"] != "active":
        return None
    return for_membership(m, None, now)
