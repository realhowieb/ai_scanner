"""Run 42 — Watchlist Intelligence 2.0 (pure, deterministic, no I/O).

Turns a watchlist from saved tickers into a monitoring view: for EVERY watchlist
symbol (including ones with no active setup) it builds a WatchlistSymbolView by
reusing the Run 40 OpportunityView engine — no parallel scoring engine. It adds
attention-first sorting, a summary, filters, and a change activity feed.

Boundaries (same as Run 40/41): reuses analytics.opportunity_view; no new score,
no Opportunity Score, no DT/model/scanner change. Alert Priority is an attention
signal, never a return prediction. Every reason/risk is backed by present data.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from analytics import opportunity_view as ov

SCHEMA_VERSION = "hsf-watchlist-view-1.0"

# Common market-row keys -> canonical indicator fields (accepts day-trader /
# scanner metric rows). Only present keys are carried; nothing invented.
_INDICATOR_KEYS = {
    "chg_pct": "chg_pct", "gap_pct": "gap_pct", "rvol": "rvol",
    "vs_vwap_pct": "vs_vwap_pct", "adx": "adx",
    "supertrend_direction": "supertrend_direction", "ewo": "ewo",
    "atr_pct": "atr_pct", "volatility20d%": "atr_pct", "Volatility20D%": "atr_pct",
}
_PRICE_KEYS = ("last", "price", "Last", "close_today")


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _row_symbol(row: Dict[str, Any]) -> str:
    return str(row.get("ticker") or row.get("symbol") or row.get("Ticker") or "").upper()


def derive_row_scanners(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic scanner triggers from a symbol's current indicators. Mirrors
    the strategy predicates (read-only) so watchlist and scanner semantics agree.
    No trigger → empty list (a valid 'no active setup' state)."""
    scanners: List[Dict[str, Any]] = []
    rvol = _num(ind.get("rvol"))
    gap = _num(ind.get("gap_pct"))
    chg = _num(ind.get("chg_pct"))
    vsv = _num(ind.get("vs_vwap_pct"))
    # Thresholds so a quiet symbol (tiny drift, normal volume) is a valid
    # 'no active setup' state rather than a false trigger.
    if rvol is not None and rvol >= 2:
        scanners.append({"name": "unusual_vol", "triggered": True, "direction": "long",
                         "score": rvol, "version": "row"})
    if chg is not None and chg >= 1.0 and (vsv is None or vsv >= 0):
        scanners.append({"name": "momentum", "triggered": True, "direction": "long",
                         "score": chg, "version": "row"})
    if gap is not None and gap >= 0.5:
        scanners.append({"name": "gap_up", "triggered": True, "direction": "long",
                         "score": gap, "version": "row"})
    if gap is not None and gap <= -0.5:
        scanners.append({"name": "gap_down", "triggered": True, "direction": "short",
                         "score": gap, "version": "row"})
    return scanners


def observation_from_market_row(
    row: Dict[str, Any], *, session: Optional[str] = None, stale: bool = False,
) -> Dict[str, Any]:
    """Adapt a market/metrics row into a canonical observation for Run 40."""
    sym = _row_symbol(row)
    ind: Dict[str, Any] = {}
    for k, field in _INDICATOR_KEYS.items():
        if k in row and row.get(k) is not None and field not in ind:
            ind[field] = row.get(k)
    price = next((row.get(k) for k in _PRICE_KEYS if row.get(k) is not None), None)
    models: Dict[str, Any] = {}
    prob = _num(row.get("prebreakout_prob") or row.get("prob"))
    if prob is not None:
        models["prebreakout"] = {"version": "prebreakout", "probability": prob}
    aic = _num(row.get("ai_confidence"))
    if aic is not None:
        models["ai_confidence"] = {"version": "ai-confidence", "confidence": aic}
    present = sum(1 for f in ("chg_pct", "gap_pct", "rvol", "vs_vwap_pct", "adx",
                              "supertrend_direction", "ewo", "atr_pct") if f in ind)
    return {
        "observation_id": f"watch:{sym}",
        "symbol": sym,
        "timestamp": row.get("timestamp") or row.get("trade_ts"),
        "session": session,
        "market": {"price": price},
        "indicators": ind,
        "models": models,
        "scanners": derive_row_scanners(ind),
        "market_context": {"source": "watchlist"},
        "data_quality": {"feature_completeness": round(present / 8, 4),
                         "fallback_used": False, "stale": bool(stale)},
    }


def build_watchlist_symbol_view(
    row: Dict[str, Any], *,
    prior_row: Optional[Dict[str, Any]] = None,
    watchlist_id: Any = None, watchlist_name: Optional[str] = None,
    session: Optional[str] = None, stale: bool = False,
) -> Dict[str, Any]:
    """One WatchlistSymbolView (Task 2). Wraps a Run 40 OpportunityView and adds
    watchlist identity + a no_active_setup flag. Every symbol stays visible."""
    obs = observation_from_market_row(row, session=session, stale=stale)
    prior_obs = (observation_from_market_row(prior_row, session=session)
                 if prior_row else None)
    # Watchlist symbols are always "surfaced" — pass watchlist so is_watchlist=True.
    v = ov.build_opportunity_view(obs, prior_obs=prior_obs, watchlist=[obs["symbol"]])
    v["schema_version"] = SCHEMA_VERSION
    v["watchlist_id"] = watchlist_id
    v["watchlist_name"] = watchlist_name
    v["rvol"] = _num((obs.get("indicators") or {}).get("rvol"))
    v["no_active_setup"] = (v.get("scanner_count", 0) == 0)
    if v["no_active_setup"]:
        v["no_setup_facts"] = _no_setup_facts(obs)
        # A quiet symbol is ACTIVE (present) but not an alert.
        if v.get("lifecycle_state") == "NEW" and prior_obs is None:
            v["lifecycle_state"] = "ACTIVE"
    return v


def _no_setup_facts(obs: Dict[str, Any]) -> List[str]:
    """Useful facts for a no-setup symbol (Task 22) — only from present data."""
    ind = obs.get("indicators") or {}
    facts: List[str] = []
    chg = _num(ind.get("chg_pct"))
    if chg is not None:
        facts.append(f"{chg:+.1f}% today")
    vsv = _num(ind.get("vs_vwap_pct"))
    if vsv is not None:
        facts.append("Above VWAP" if vsv > 0 else "Below VWAP" if vsv < 0 else "At VWAP")
    rvol = _num(ind.get("rvol"))
    if rvol is not None:
        facts.append(f"RVOL {rvol:.1f}x")
    facts.append("No meaningful change since prior scan")
    return facts


def build_watchlist_views(
    rows: Sequence[Dict[str, Any]], *,
    prior_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
    watchlist_id: Any = None, watchlist_name: Optional[str] = None,
    session: Optional[str] = None, stale: bool = False,
) -> List[Dict[str, Any]]:
    """Build views for ALL rows (no symbol dropped). One bad row is skipped, not
    fatal (Task 26)."""
    prior_by_symbol = prior_by_symbol or {}
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        try:
            sym = _row_symbol(row)
            out.append(build_watchlist_symbol_view(
                row, prior_row=prior_by_symbol.get(sym),
                watchlist_id=watchlist_id, watchlist_name=watchlist_name,
                session=session, stale=stale))
        except Exception:
            continue
    return out


_PRIORITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
_LIFECYCLE_RANK = {"NEW": 5, "STRENGTHENING": 4, "ACTIVE": 3, "WEAKENING": 2,
                   "RESOLVED": 1}


def _attention_score(v: Dict[str, Any]) -> tuple:
    # No-setup symbols sink below any active setup.
    active = 0 if v.get("no_active_setup") else 1
    # WEAKENING is attention-worthy: rank it above plain ACTIVE.
    life = _LIFECYCLE_RANK.get(v.get("lifecycle_state"), 0)
    weak_boost = 1 if v.get("lifecycle_state") == "WEAKENING" else 0
    return (active, _PRIORITY_RANK.get(v.get("alert_priority"), 0), life + weak_boost,
            v.get("scanner_count") or 0)


def attention_sort(views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attention-first default order (Task 4): HIGH+NEW/STRENGTHENING first,
    WEAKENING surfaced, no-setup last. Deterministic tiebreak by symbol."""
    return sorted(views, key=lambda v: (_attention_score(v), _neg_symbol(v)), reverse=True)


def _neg_symbol(v: Dict[str, Any]):
    # For reverse=True, invert symbol so it sorts A→Z within equal attention.
    return tuple(-ord(c) for c in str(v.get("symbol") or ""))


def sort_views(views: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """User-selectable sort (Task 4)."""
    k = str(key or "Attention")
    if k == "Attention":
        return attention_sort(views)
    if k == "Ticker":
        return sorted(views, key=lambda v: str(v.get("symbol") or ""))
    if k == "% Change":
        return sorted(views, key=lambda v: (v.get("change_pct") is None,
                                            -(v.get("change_pct") or 0)))
    if k == "RVOL":
        return sorted(views, key=lambda v: (v.get("rvol") is None, -(v.get("rvol") or 0)))
    if k == "Priority":
        return sorted(views, key=lambda v: -_PRIORITY_RANK.get(v.get("alert_priority"), 0))
    if k == "Setup":
        return sorted(views, key=lambda v: str(v.get("primary_setup") or "~"))
    return attention_sort(views)


def watchlist_summary(views: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact summary counts (Task 8). Only meaningful categories."""
    life = Counter(v.get("lifecycle_state") for v in views)
    needs = sum(1 for v in views if v.get("alert_priority") in ("HIGH", "MEDIUM")
                and not v.get("no_active_setup"))
    return {
        "total": len(views),
        "needs_attention": needs,
        "new_setups": life.get("NEW", 0),
        "strengthening": life.get("STRENGTHENING", 0),
        "weakening": life.get("WEAKENING", 0),
        "no_active_setup": sum(1 for v in views if v.get("no_active_setup")),
    }


_FILTERS = ("All", "Needs Attention", "High Priority", "New", "Strengthening",
            "Weakening", "Active Setups", "No Setup", "Bullish", "Bearish")


def watchlist_filters() -> tuple:
    return _FILTERS


def filter_watchlist(views: Sequence[Dict[str, Any]], flt: str) -> List[Dict[str, Any]]:
    f = str(flt or "All")
    if f in ("All", ""):
        return list(views)
    def _m(v: Dict[str, Any]) -> bool:
        if f == "Needs Attention":
            return v.get("alert_priority") in ("HIGH", "MEDIUM") and not v.get("no_active_setup")
        if f == "High Priority":
            return v.get("alert_priority") == "HIGH"
        if f == "New":
            return v.get("lifecycle_state") == "NEW"
        if f == "Strengthening":
            return v.get("lifecycle_state") == "STRENGTHENING"
        if f == "Weakening":
            return v.get("lifecycle_state") == "WEAKENING"
        if f == "Active Setups":
            return not v.get("no_active_setup")
        if f == "No Setup":
            return bool(v.get("no_active_setup"))
        if f == "Bullish":
            return v.get("direction") == "bullish"
        if f == "Bearish":
            return v.get("direction") == "bearish"
        return True
    return [v for v in views if _m(v)]


def activity_feed(views: Sequence[Dict[str, Any]], *, limit: int = 20) -> List[Dict[str, Any]]:
    """Recent-changes feed from each symbol's changes_since_prior (Task 9). Uses
    only detected changes — never manufactures history."""
    events: List[Dict[str, Any]] = []
    for v in views:
        for c in (v.get("changes_since_prior") or []):
            events.append({"symbol": v.get("symbol"), "change": c,
                           "timestamp": v.get("timestamp"),
                           "lifecycle_state": v.get("lifecycle_state")})
    # Stable, deterministic order: by symbol then change text (no fake clock).
    events.sort(key=lambda e: (str(e["symbol"]), str(e["change"])))
    return events[: max(0, int(limit))]


def empty_watchlist_message() -> str:
    return ("Your watchlist is empty. Add stocks from Scanner, Market Brief, or "
            "Day Trader to start monitoring changes and setups here.")
