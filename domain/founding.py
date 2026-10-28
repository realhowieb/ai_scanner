"""Founding 100 memberships: tiers, prices, seat counts, flags, the 5-year term and
what each tier unlocks. Pure rules, no database or Stripe.

Founder Pro ($499, 75 seats) and Founder Elite ($999, 25 seats) are one-time payments
for five calendar years of access. Founder Pro unlocks today's Pro plan and Founder
Elite today's Premium plan. Founder numbers run #001 to #100 across both tiers.

Prices live here, never in the browser: checkout (Milestone 2) charges these amounts
and fulfilment refuses a payment whose amount doesn't match.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, Optional

FOUNDER_PRO = "founder_pro"
FOUNDER_ELITE = "founder_elite"
TIERS = (FOUNDER_PRO, FOUNDER_ELITE)

CAPACITY: Dict[str, int] = {FOUNDER_PRO: 75, FOUNDER_ELITE: 25}
TOTAL_CAPACITY = 100
assert sum(CAPACITY.values()) == TOTAL_CAPACITY

CURRENCY = "usd"
PRICE_CENTS: Dict[str, int] = {FOUNDER_PRO: 49_900, FOUNDER_ELITE: 99_900}
UPGRADE_PRICE_CENTS = 50_000  # Founder Pro -> Founder Elite
TERM_YEARS = 5
# Stripe Checkout sessions last at least 30 minutes, so a seat is held a little longer
# than the session that pays for it (CHECKOUT_MINUTES).
RESERVATION_MINUTES = 35
CHECKOUT_MINUTES = 31

# The existing plan each founder tier unlocks while the membership is active.
PLAN_FOR_TIER: Dict[str, str] = {FOUNDER_PRO: "pro", FOUNDER_ELITE: "premium"}
TIER_LABEL: Dict[str, str] = {FOUNDER_PRO: "Founder Pro", FOUNDER_ELITE: "Founder Elite"}
BADGE: Dict[str, str] = {FOUNDER_PRO: "gold", FOUNDER_ELITE: "platinum"}

STATUSES = ("pending", "active", "expired", "refunded", "revoked")
RESERVATION_KINDS = ("purchase", "upgrade")
RESERVATION_STATUSES = ("held", "converted", "released", "expired", "needs_review")

# Payments stay off until each switch is set to 1 on purpose (unlike most HSF switches,
# where unset means on).
FLAGS = ("FOUNDING_100_ENABLED", "FOUNDING_CHECKOUT_ENABLED", "FOUNDING_ELITE_ENABLED", "FOUNDING_UPGRADES_ENABLED",
         "FOUNDING_LEGAL_PAGES_ENABLED", "FOUNDING_REFUNDS_ENABLED", "FOUNDING_PRORATED_REFUNDS_ENABLED")


def flag(name: str, env: Optional[Dict[str, str]] = None) -> bool:
    if name not in FLAGS:
        raise KeyError(name)
    return str((env if env is not None else os.environ).get(name, "0")).strip() == "1"


def tier_available(tier: str, env: Optional[Dict[str, str]] = None) -> bool:
    """Whether new seats of `tier` can be sold right now (program, checkout and Elite switches)."""
    if tier not in TIERS:
        return False
    if not (flag("FOUNDING_100_ENABLED", env) and flag("FOUNDING_CHECKOUT_ENABLED", env)):
        return False
    return tier != FOUNDER_ELITE or flag("FOUNDING_ELITE_ENABLED", env)


def expected_amount(kind: str, tier: str) -> int:
    """The only amount, in cents, that pays for a purchase of `tier` or an upgrade to it."""
    if kind == "upgrade":
        if tier != FOUNDER_ELITE:
            raise ValueError("Only Founder Pro can be upgraded, to Founder Elite")
        return UPGRADE_PRICE_CENTS
    if kind != "purchase" or tier not in PRICE_CENTS:
        raise ValueError(f"unknown founding purchase {kind!r} {tier!r}")
    return PRICE_CENTS[tier]


def add_years(t: dt.datetime, years: int) -> dt.datetime:
    """Same month, day and time `years` later (calendar years, not 365-day blocks).
    29 February falls back to 28 February when the target year has no leap day."""
    try:
        return t.replace(year=t.year + years)
    except ValueError:  # 29 Feb -> 28 Feb
        return t.replace(year=t.year + years, day=28)


def term_end(activated_at: dt.datetime) -> dt.datetime:
    """When a membership activated at `activated_at` (UTC) stops giving access."""
    if activated_at.tzinfo is None:
        activated_at = activated_at.replace(tzinfo=dt.timezone.utc)
    return add_years(activated_at.astimezone(dt.timezone.utc), TERM_YEARS)


def format_number(number: Optional[int]) -> Optional[str]:
    return f"{int(number):03d}" if number is not None else None


def badge_title(tier: str, number: Optional[int]) -> Optional[str]:
    """'HSF AI Founder #017' / 'HSF AI Elite Founder #004'; kept after the membership expires."""
    n = format_number(number)
    if n is None or tier not in TIERS:
        return None
    return f"HSF AI {'Elite ' if tier == FOUNDER_ELITE else ''}Founder #{n}"


# The brief's entitlement names, mapped onto the feature flags HSF enforces today
# (domain/plans.FEATURE_MIN_TIER). "planned" ones have no working feature yet: they
# grant nothing and the page must say "planned", not "included".
ENTITLEMENTS: Dict[str, Dict[str, object]] = {
    "scanner.basic": {"min_plan": "basic", "flags": ["can_scan_sp500"]},
    "scanner.advanced": {"min_plan": "pro", "flags": ["can_scan_nasdaq", "can_unusual_volume", "can_scan_history"]},
    "market_intelligence.basic": {"min_plan": "basic", "flags": []},
    "market_intelligence.advanced": {"min_plan": "pro", "flags": ["can_premarket", "can_afterhours", "can_earnings", "can_day_trader"]},
    "ml.insights": {"min_plan": "pro", "flags": ["can_track_record"]},
    "ml.premium_insights": {"min_plan": "premium", "flags": ["can_early_breakout", "can_ai_notes"]},
    "watchlists.expanded": {"min_plan": "pro", "flags": ["can_email_alerts", "can_scan_alerts"]},
    "outcomes.advanced": {"min_plan": "pro", "flags": ["can_scan_history", "can_track_record"]},
    "scanner.custom_configurations": {"min_plan": "premium", "flags": ["can_full_universe"]},
    "api.personal_access": {"min_plan": "premium", "flags": [], "planned": True},
    "founder.early_access": {"founder": True, "flags": [], "planned": True},
    "founder.priority_feedback": {"founder": True, "flags": []},
}


def entitlements_for(plan: str, founder_tier: Optional[str] = None) -> Dict[str, bool]:
    """Which named entitlements an account has, from its effective plan and, for the
    founder-only ones, an active founder membership. Planned entitlements are always False."""
    from domain.plans import has_min_tier

    out: Dict[str, bool] = {}
    for name, rule in ENTITLEMENTS.items():
        if rule.get("planned"):
            out[name] = False
        elif rule.get("founder"):
            out[name] = founder_tier in TIERS
        else:
            out[name] = has_min_tier(plan, str(rule["min_plan"]))
    return out


def price_label(cents: int) -> str:
    return f"${cents / 100:,.0f}" if cents % 100 == 0 else f"${cents / 100:,.2f}"


def disclosure(tier: str, kind: str = "purchase") -> Dict[str, str]:
    """What checkout must show before payment (the Founding 100 legal brief, Phase 4)."""
    amount = expected_amount(kind, tier)
    name = f"HSF AI {TIER_LABEL[tier]}"
    if kind == "upgrade":
        return {"name": f"{name} upgrade", "payment": f"One-time payment: {price_label(amount)}",
                "duration": "Membership duration: Keeps your current end date",
                "recurring": "Recurring payments: None", "refund": "Refund guarantee: 14 days"}
    return {"name": name, "payment": f"One-time payment: {price_label(amount)}",
            "duration": "Membership duration: Five years", "recurring": "Recurring payments: None",
            "refund": "Refund guarantee: 14 days"}
