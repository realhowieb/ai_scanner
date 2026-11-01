"""Run 72 — one source of truth for plans and pricing (P1-15).

The Billing page table, the Billing benefits text and the signed-out landing
"Plans" cards are all generated from ROWS below, whose ✅/❌ cells are DERIVED
from the real entitlement map (ui.app_session.FEATURE_MIN_TIER) and alert
limits (ALERT_LIMIT_BY_TIER). Copy can no longer drift from what each tier
actually gets.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Tuple

from domain.plans import ALERT_LIMIT_BY_TIER, FEATURE_MIN_TIER, TIER_ORDER
from domain.upgrades import UPGRADE_MESSAGES, required_tier, upgrade_message

TIERS = ("basic", "pro", "premium")
from ui.plan_labels import PLAN_LABELS  # noqa: E402  (Run 85B: one source of plan names)

TIER_NAMES = {k: PLAN_LABELS[k] for k in ("basic", "pro", "premium")}
PRICES = {"basic": "Free", "pro": "$25/mo", "premium": "$40/mo"}
YEARLY_PRICES = {"pro": "$250/yr", "premium": "$400/yr"}  # 10 x monthly: two months free
TAGLINES = {
    "basic": "Discover what matters in the market.",
    "pro": "Monitor and investigate the opportunities that matter to you.",
    "premium": "Go deeper with advanced research and workflow tools.",
}

ALERTS = "__alerts__"
# (label, entitlement flag). flag=None → included in every plan; ALERTS → alert limits.
ROWS: List[Tuple[str, Optional[str]]] = [
    ("Latest full-market ranking (HSF Score), Today and scheduled market opportunities", None),
    ("HSF Score and basic Stock Intelligence", None),
    ("Scanner filters and saved views", None),
    ("Watchlists with notes and price-vs-SPY tracking", None),
    ("Your own S&P 500 scans", "can_scan_sp500"),
    ("Alerts (price, watchlist, score and setup)", ALERTS),
    ("Email alert delivery", "can_email_alerts"),
    ("Scan alerts: get told when new stocks match your Scanner filters", "can_scan_alerts"),
    ("Insider-buy alerts from SEC filings", "can_insider_alerts"),
    ("CSV export of Scanner results", "can_export_csv"),
    ("Your own Nasdaq and combined-market scans", "can_scan_nasdaq"),
    ("Premarket, after-hours and unusual-volume filters", "can_premarket"),
    ("Live Day Trader monitor: intraday movers, VWAP, relative volume and stair-steppers", "can_day_trader"),
    ("Earnings calendar & filters", "can_earnings"),
    ("Every earnings reporter scored automatically on the Earnings tab", "can_earnings_autoscore"),
    ("Scan history & historical research", "can_scan_history"),
    ("AI scan summaries and results chat, setup notes on each result and watchlist rundowns", "can_ai_notes"),
    ("Early Breakout candidate research", "can_early_breakout"),
    ("Full-market custom scanning workflow", "can_full_universe"),
    ("Alpaca paper-trading workflow", "can_paper_trade"),
]


def alert_upgrade_message(current_tier: str) -> str:
    key = str(current_tier or "basic").strip().lower()
    if key == "basic":
        return "Pro raises your alert limit to 5 and adds email delivery."
    if key == "pro":
        return "Premium raises your alert limit from 5 to 25 for a larger monitoring workflow."
    return "Your plan includes the maximum customer alert limit."


def included(flag: Optional[str], tier: str) -> bool:
    """Whether `tier` gets the feature behind `flag` (per FEATURE_MIN_TIER)."""
    if flag is None or flag == ALERTS:
        return True
    min_tier = FEATURE_MIN_TIER.get(flag)
    if min_tier is None or min_tier == "admin":
        return False
    return TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(min_tier, 99)


def cell(flag: Optional[str], tier: str) -> str:
    if flag == ALERTS:
        return str(ALERT_LIMIT_BY_TIER.get(tier, 1))
    return "✅" if included(flag, tier) else "❌"


def pricing_markdown() -> str:
    """The comparison table (Markdown) for the Billing page."""
    lines = [f"| Feature | {TIER_NAMES['basic']} | Pro | Premium |", "|---|---:|---:|---:|",
             f"| **Price** | **{PRICES['basic']}** | **{PRICES['pro']}** | **{PRICES['premium']}** |"]
    for label, flag in ROWS:
        lines.append(f"| {label} | " + " | ".join(cell(flag, t) for t in TIERS) + " |")
    return "\n".join(lines)


def plan_highlights() -> Dict[str, List[str]]:
    """What each plan adds over the one below it (derived from ROWS)."""
    out: Dict[str, List[str]] = {}
    prev: Optional[str] = None
    for tier in TIERS:
        items: List[str] = []
        for label, flag in ROWS:
            if flag == ALERTS:
                n = ALERT_LIMIT_BY_TIER.get(tier, 1)
                items.append(f"{n} alert{'s' if n != 1 else ''}")
            elif included(flag, tier) and (prev is None or not included(flag, prev)):
                items.append(label)
        out[tier] = items
        prev = tier
    return out


def benefits_markdown() -> str:
    """Short 'what each plan adds' text for the Billing page."""
    h = plan_highlights()
    return "\n".join([
        f"- **{TIER_NAMES['basic']} — Discover:** {'; '.join(h['basic'])}.",
        f"- **Pro — Monitor & investigate:** {'; '.join(h['pro'])}.",
        f"- **Premium — Research & workflow:** {'; '.join(h['premium'])}.",
    ])


_CSS = """
<style>
.hsf-plans{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.hsf-plan{border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:12px 14px;background:rgba(128,128,128,.08)}
.hsf-plan.featured{border:2px solid var(--hsf-gold,#b8892b);background:rgba(184,137,43,.08)}
.hsf-plan .badge{font-size:.72rem;font-weight:700;text-transform:uppercase;margin:0 0 5px;color:var(--hsf-gold,#b8892b)}
.hsf-plan h3{font-size:1rem;margin:0;padding:0}
.hsf-plan .price{font-size:1.15rem;font-weight:700;margin:2px 0 4px}
.hsf-plan .tag{font-size:.85rem;opacity:.8;margin:0 0 6px}
.hsf-plan ul{margin:0;padding-left:1.1rem;font-size:.88rem;line-height:1.4}
</style>
"""


def plans_html() -> str:
    """Three plan cards for the signed-out landing page."""
    h = plan_highlights()
    cards = []
    for tier in TIERS:
        prefix = [] if tier == "basic" else [f"Everything in {TIER_NAMES['basic' if tier == 'pro' else 'pro']}"]
        items = "".join(f"<li>{html.escape(i)}</li>" for i in prefix + h[tier])
        featured = " featured" if tier == "pro" else ""
        # P2-21: no popularity claim until customer data backs it.
        badge = '<p class="badge">Recommended</p>' if tier == "pro" else ""
        cards.append(
            f'<div class="hsf-plan{featured}">{badge}<h3>{TIER_NAMES[tier]}</h3>'
            f'<p class="price">{html.escape("$0" if tier == "basic" else PRICES[tier])}</p>'
            f'<p class="tag">{html.escape(TAGLINES[tier])}</p><ul>{items}</ul></div>'
        )
    return f'{_CSS}<div class="hsf-plans">{"".join(cards)}</div>'
