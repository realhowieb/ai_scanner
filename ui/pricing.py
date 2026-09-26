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

from ui.app_session import ALERT_LIMIT_BY_TIER, FEATURE_MIN_TIER, TIER_ORDER

TIERS = ("basic", "pro", "premium")
TIER_NAMES = {"basic": "Free", "pro": "Pro", "premium": "Premium"}
PRICES = {"basic": "Free", "pro": "$19/mo", "premium": "$39/mo"}
TAGLINES = {
    "basic": "The full daily product, free.",
    "pro": "Email alerts, exports and deeper history.",
    "premium": "AI notes, model tools and your own full-market scans.",
}

ALERTS = "__alerts__"
# (label, entitlement flag). flag=None → included in every plan; ALERTS → alert limits.
ROWS: List[Tuple[str, Optional[str]]] = [
    ("Latest full-market ranking (HSF Score), Today, Stock Intelligence", None),
    ("Lenses, card view & saved screens", None),
    ("My Stocks (watchlists) & charts", None),
    ("Your own S&P 500 scans", "can_scan_sp500"),
    ("Alerts (Breakout / Watchlist / Price)", ALERTS),
    ("Email alert delivery", "can_email_alerts"),
    ("CSV export & interactive results table", "can_export_csv"),
    ("Your own Nasdaq scans & advanced scan filters", "can_scan_nasdaq"),
    ("Earnings calendar & filters", "can_earnings"),
    ("Scan history & historical research", "can_scan_history"),
    ("AI notes, summaries & chat", "can_ai_notes"),
    ("Early Breakout candidates (model)", "can_early_breakout"),
    ("Your own full-universe scans (3-step scanner)", "can_full_universe"),
    ("Paper trading (Alpaca)", "can_paper_trade"),
]


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
    lines = ["| Feature | Free | Pro | Premium |", "|---|---:|---:|---:|",
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
        f"- **Free** includes: {'; '.join(h['basic'])}.",
        f"- **🔒 Pro** adds: {'; '.join(h['pro'])}.",
        f"- **🔒 Premium** adds: {'; '.join(h['premium'])}.",
    ])


_CSS = """
<style>
.hsf-plans{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.hsf-plan{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:12px 14px;background:rgba(128,128,128,.08)}
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
        cards.append(
            f'<div class="hsf-plan"><h3>{TIER_NAMES[tier]}</h3>'
            f'<p class="price">{html.escape("$0" if tier == "basic" else PRICES[tier])}</p>'
            f'<p class="tag">{html.escape(TAGLINES[tier])}</p><ul>{items}</ul></div>'
        )
    return f'{_CSS}<div class="hsf-plans">{"".join(cards)}</div>'
