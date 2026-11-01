"""Plan upgrade explanations shared by API and presentation adapters."""

from typing import Optional

from domain.plan_labels import PLAN_LABELS
from domain.plans import FEATURE_MIN_TIER

TIERS = ("basic", "pro", "premium")
TIER_NAMES = {k: PLAN_LABELS[k] for k in TIERS}

UPGRADE_MESSAGES = {
    "can_export_csv": (
        "Pro adds CSV export of Scanner results, so you can take the filtered list into "
        "your own spreadsheet."
    ),
    "can_email_alerts": ("Pro adds email delivery so HSF can notify you when an alert you created fires."),
    "can_scan_nasdaq": ("Pro adds your own Nasdaq and combined-market scans for broader discovery."),
    "can_premarket": ("Pro adds premarket, after-hours and unusual-volume filters for deeper investigation."),
    "can_day_trader": (
        "Pro adds the live Day Trader monitor: intraday movers, VWAP, relative volume and the stair-stepper check."
    ),
    "can_earnings": ("Pro adds earnings timing and filters so you can investigate event risk around a setup."),
    "can_scan_history": (
        "Pro adds scan history and historical research so you can compare current opportunities with prior runs."
    ),
    "can_scan_alerts": (
        "Pro adds scan alerts: save scanner filters once and HSF tells you when new stocks match after each scan."
    ),
    "can_insider_alerts": (
        "Pro adds insider-buy alerts: HSF tells you when a company insider reports buying shares (SEC Form 4)."
    ),
    "can_earnings_autoscore": (
        "Premium scores every earnings reporter automatically, so the Earnings tab shows an HSF Score on each one."
    ),
    "can_track_record": ("Pro adds historical research so you can inspect how prior HSF observations evolved."),
    "can_ai_notes": (
        "Premium adds AI scan summaries and results chat, setup notes on each result and watchlist rundowns, "
        "for a deeper research workflow."
    ),
    "can_early_breakout": (
        "Premium adds Early Breakout candidate research for investigating setups before they fully form."
    ),
    "can_full_universe": ("Premium adds the full-market custom scanning workflow for advanced research."),
    "can_paper_trade": (
        "Premium adds an Alpaca paper-trading workflow so you can practice and journal setups without real-money orders."
    ),
}


def required_tier(flag: str) -> Optional[str]:
    tier = FEATURE_MIN_TIER.get(flag)
    return tier if tier in TIERS else None


def upgrade_message(flag: str) -> str:
    """Capability-specific, truthful upgrade copy from the entitlement source."""
    tier = required_tier(flag)
    if tier is None:
        return "This capability is not available as a customer plan upgrade."
    return UPGRADE_MESSAGES.get(
        flag,
        f"{TIER_NAMES[tier]} unlocks this capability.",
    )
