"""Run 41 — Market Brief 2.0 composition (pure, deterministic, no I/O).

Assembles the Market Brief intelligence hierarchy from data the page ALREADY
computes (snapshot opportunities, breadth, regime, sectors) and the Run 40
OpportunityView engine. No scanner/model/DT change, no Opportunity Score, no new
prediction. Alert Priority remains an attention signal, never a return claim.

All logic lives here so the page (`ui/market_brief.py`) stays a thin renderer and
everything is testable outside Streamlit.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from analytics import opportunity_view as ov

# ui.opportunities signal -> Run 40 scanner name (+ default direction).
_SIGNAL_TO_SCANNER = {
    "prebreakout": ("prebreakout", "long"),
    "breakout": ("breakout", "long"),
    "golden_cross": ("golden_cross", "long"),
    "gapper": ("gap_up", "long"),
    "gainer": ("momentum", "long"),
}


def observation_from_opportunity(opp: Dict[str, Any], *, stale: bool = False) -> Dict[str, Any]:
    """Adapt a `ui.opportunities.build_opportunities` dict into a canonical-ish
    observation the Run 40 engine consumes. Only fields actually present are
    carried; snapshot picks lack intraday indicators (ADX/RVOL/VWAP), so those are
    honestly absent and the view will flag partial where relevant."""
    ticker = str(opp.get("ticker") or "").upper()
    scanners: List[Dict[str, Any]] = []
    for sig in (opp.get("signals") or []):
        mapped = _SIGNAL_TO_SCANNER.get(str(sig))
        if mapped:
            name, direction = mapped
            scanners.append({"name": name, "triggered": True, "direction": direction,
                             "version": "snapshot"})
    indicators: Dict[str, Any] = {}
    if opp.get("chg_pct") is not None:
        indicators["chg_pct"] = opp.get("chg_pct")
    if opp.get("gap_pct") is not None:
        indicators["gap_pct"] = opp.get("gap_pct")
    models: Dict[str, Any] = {}
    if opp.get("prob") is not None:
        models["prebreakout"] = {"version": "prebreakout", "probability": opp.get("prob")}
    return {
        "observation_id": f"brief:{ticker}",
        "symbol": ticker,
        "timestamp": opp.get("timestamp"),
        "session": opp.get("session"),
        "market": {"price": opp.get("last") or opp.get("price")},
        "indicators": indicators,
        "models": models,
        "scanners": scanners,
        "market_context": {"source": "market_brief"},
        "data_quality": {"feature_completeness": None, "fallback_used": False,
                         "stale": bool(stale)},
        # Carry the fading flag so the view can add a reversal risk.
        "_fading": bool(opp.get("fading")),
    }


def build_top_opportunity_views(
    opportunities: Sequence[Dict[str, Any]], *,
    watchlist: Optional[Sequence[str]] = None,
    prior_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
    top_n: int = 8,
    stale: bool = False,
) -> List[Dict[str, Any]]:
    """Build ranked Run 40 OpportunityViews from Market Brief opportunities."""
    prior_by_ticker = prior_by_ticker or {}
    views = []
    for opp in opportunities or []:
        obs = observation_from_opportunity(opp, stale=stale)
        prior = prior_by_ticker.get(obs["symbol"])
        prior_obs = observation_from_opportunity(prior, stale=stale) if prior else None
        v = ov.build_opportunity_view(obs, prior_obs=prior_obs, watchlist=watchlist)
        if obs.get("_fading") and "Fading / reversal risk" not in v["risk_reasons"]:
            v["risk_reasons"] = list(v["risk_reasons"]) + ["Fading / reversal risk"]
        views.append(v)
    return ov.rank_feed(views)[: max(0, int(top_n))]


def summarize_brief_state(
    views: Sequence[Dict[str, Any]], *, breadth: Any = None, regime: Optional[str] = None,
) -> Dict[str, Any]:
    """Compact, comparable snapshot of the brief's state (for What Changed)."""
    high = sum(1 for v in views if v.get("alert_priority") == "HIGH")
    setups = Counter(v.get("primary_setup") for v in views if v.get("primary_setup"))
    breadth_pct = None
    if isinstance(breadth, (tuple, list)) and len(breadth) == 2:
        adv, dec = breadth
        tot = (adv or 0) + (dec or 0)
        breadth_pct = round(100 * (adv or 0) / tot) if tot else None
    return {
        "high_priority": high,
        "total_opportunities": len(views),
        "regime": regime,
        "breadth_pct": breadth_pct,
        "by_setup": dict(setups),
        "tickers": sorted(str(v.get("symbol")) for v in views),
    }


def diff_brief_state(current: Dict[str, Any], prior: Optional[Dict[str, Any]]) -> List[str]:
    """What Changed since the previous meaningful brief (Task 5). State
    transitions over small numeric moves; empty when no prior."""
    if not prior:
        return []
    changes: List[str] = []
    ch, ph = current.get("high_priority"), prior.get("high_priority")
    if ch is not None and ph is not None and ch != ph:
        arrow = "↑" if ch > ph else "↓"
        changes.append(f"{arrow} High Priority setups: {ph} → {ch}")
    cr, prg = current.get("regime"), prior.get("regime")
    if cr and prg and cr != prg:
        changes.append(f"Market regime: {prg} → {cr}")
    cb, pb = current.get("breadth_pct"), prior.get("breadth_pct")
    if cb is not None and pb is not None and abs(cb - pb) >= 5:
        arrow = "↑" if cb > pb else "↓"
        changes.append(f"{arrow} Breadth: {pb}% → {cb}% advancing")
    new_tickers = [t for t in current.get("tickers", []) if t not in set(prior.get("tickers", []))]
    for t in new_tickers[:3]:
        changes.append(f"NEW opportunity: {t}")
    return changes


def build_watchlist_events(views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Watchlist symbols with a meaningful setup/state (Task 9). Not the static
    list — only entries that actually surfaced as opportunities."""
    out = []
    for v in views:
        if not v.get("is_watchlist"):
            continue
        out.append({
            "symbol": v.get("symbol"), "state": v.get("lifecycle_state"),
            "primary_setup": v.get("primary_setup"),
            "priority": v.get("alert_priority"),
            "reasons": (v.get("positive_reasons") or [])[:3],
        })
    return out


def market_pulse(
    views: Sequence[Dict[str, Any]], *, sectors: Optional[Sequence[Any]] = None,
) -> List[str]:
    """A few useful, data-supported observations about where activity is (Task 4).
    Only states things the data supports; omits weak themes."""
    pulse: List[str] = []
    setups = Counter(v.get("primary_setup") for v in views if v.get("primary_setup"))
    if setups:
        top, n = setups.most_common(1)[0]
        if n >= 2:
            pulse.append(f"{top} is the most common setup ({n})")
    uv = sum(1 for v in views if "Unusual Volume" in (v.get("scanner_names") or []))
    if uv >= 3:
        pulse.append(f"Unusual volume across {uv} names")
    bullish = sum(1 for v in views if v.get("direction") == "bullish")
    bearish = sum(1 for v in views if v.get("direction") == "bearish")
    if bullish or bearish:
        pulse.append(f"{bullish} bullish / {bearish} bearish setups")
    if sectors:
        lead = sectors[0] if sectors else None
        if isinstance(lead, (tuple, list)) and len(lead) >= 1:
            pulse.append(f"Sector leader: {lead[0]}")
    return pulse
