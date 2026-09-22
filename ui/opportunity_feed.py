"""Run 40 — Intelligent Alerts & Opportunity Feed (thin Streamlit surface).

Renders the deterministic OpportunityView from analytics.opportunity_view. ALL
logic lives in that pure, tested module; this file only lays out cards. Guarded
so it imports without Streamlit (headless/CI). No scanner/model/DT behavior is
touched — it presents existing evidence.

Mount from a page with:
    from ui.opportunity_feed import render_opportunity_feed
    render_opportunity_feed(observations, watchlist=wl, scanned=cov_expected,
                            detected=len(observations))
where `observations` are canonical Run 36 observations (e.g. from the latest
scan / capture). `prior_by_symbol` optionally supplies each symbol's previous
observation for change detection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from analytics.opportunity_view import (
    FEED_FILTERS,
    build_opportunity_view,
    empty_state_message,
    filter_feed,
    rank_feed,
)

_PRIORITY_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "⚪"}
_FRESH_ICON = {"Fresh": "", "Partial Data": "⚠️ Partial", "Delayed": "⏳ Delayed",
               "Stale": "⚠️ Stale"}


def build_feed_views(
    observations: Sequence[Dict[str, Any]],
    *,
    prior_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
    watchlist: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Build ranked OpportunityViews for a set of observations (pure)."""
    prior_by_symbol = prior_by_symbol or {}
    views = [
        build_opportunity_view(
            o, prior_obs=prior_by_symbol.get(str(o.get("symbol") or "").upper()),
            watchlist=watchlist)
        for o in observations or []
    ]
    return rank_feed(views)


def _render_card(st, v: Dict[str, Any]) -> None:
    icon = _PRIORITY_ICON.get(v.get("alert_priority"), "⚪")
    star = "★ " if v.get("is_watchlist") else ""
    chg = v.get("change_pct")
    chg_txt = f"  ({chg:+.1f}%)" if isinstance(chg, (int, float)) else ""
    fresh = _FRESH_ICON.get(v.get("freshness"), "")
    header = (f"{icon} **{star}{v.get('symbol')}** — {v.get('primary_setup') or 'Signal'}"
              f"  ·  {v.get('alert_priority')}{chg_txt}")
    if fresh:
        header += f"  ·  {fresh}"
    st.markdown(header)
    if v.get("scanner_count", 0) >= 2:
        secondary = " • ".join(v.get("secondary_setups") or [])
        line = f"{v['scanner_count']} scanners agree"
        if secondary:
            line += f" — also: {secondary}"
        st.caption(line + "  ·  confirmation/context, not a predictive edge")
    reasons = (v.get("positive_reasons") or [])[:4]
    if reasons:
        st.markdown("  ".join(f"✅ {r}" for r in reasons))
    risks = (v.get("risk_reasons") or [])[:2]
    if risks:
        st.markdown("  ".join(f"⚠️ {r}" for r in risks))
    changes = (v.get("changes_since_prior") or [])[:3]
    if changes:
        st.caption("Changed: " + " · ".join(changes) + f"  ({v.get('lifecycle_state')})")


def render_opportunity_feed(
    observations: Sequence[Dict[str, Any]],
    *,
    prior_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
    watchlist: Optional[Sequence[str]] = None,
    scanned: Optional[int] = None,
    detected: Optional[int] = None,
    default_filter: str = "All",
) -> None:
    """Render the feed. No-op (returns) when Streamlit is unavailable."""
    try:
        import streamlit as st
    except Exception:
        return

    views = build_feed_views(observations, prior_by_symbol=prior_by_symbol,
                             watchlist=watchlist)
    st.markdown("### 🔔 Intelligent Alerts")
    st.caption("The most interesting things HSF sees right now — and why. "
               "Alert Priority is an attention signal, **not** a prediction of return.")
    flt = st.selectbox("Filter", FEED_FILTERS,
                       index=FEED_FILTERS.index(default_filter)
                       if default_filter in FEED_FILTERS else 0)
    shown = filter_feed(views, flt)
    high_priority = sum(1 for v in views if v.get("alert_priority") == "HIGH")

    if not shown:
        msg = empty_state_message(scanned=scanned,
                                  detected=detected if detected is not None else len(views),
                                  high_priority=0 if flt in ("All", "High Priority") else high_priority)
        st.info(msg or "No opportunities match this filter right now.")
        return

    for v in shown:
        with st.container():
            _render_card(st, v)
            st.divider()
