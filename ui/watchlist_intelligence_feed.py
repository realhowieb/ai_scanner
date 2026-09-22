"""Run 42 — Watchlist Intelligence 2.0 surface (thin Streamlit renderer).

Renders the Run 40/42 WatchlistSymbolView for every watchlist symbol. All logic
lives in analytics.watchlist_view (which reuses analytics.opportunity_view); this
file only lays out cards. Guarded so it never breaks the watchlist page, imports
headless, triggers no full-market scan (one batched metrics fetch), and makes no
predictive claim.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from analytics import watchlist_view as wv

_PRIORITY_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "⚪"}
_LIFECYCLE_ICON = {"NEW": "🆕", "STRENGTHENING": "📈", "ACTIVE": "▪️",
                   "WEAKENING": "📉", "RESOLVED": "✔️"}


def _watchlist_tickers(user: str) -> List[str]:
    try:
        from db.watchlists import get_watchlist_tickers, list_watchlists
        out: List[str] = []
        for wl in (list_watchlists(user) or []):
            out += get_watchlist_tickers(wl.get("id"), user) or []
        return sorted({str(t).upper() for t in out if t})
    except Exception:
        return []


def _metrics_rows(tickers: Sequence[str]) -> List[Dict[str, Any]]:
    """One batched metrics fetch for the whole watchlist (Task 25) — never a
    full-market scan, never per-symbol."""
    if not tickers:
        return []
    try:
        from market_data import build_day_trader_metrics
        return build_day_trader_metrics(list(tickers), with_rvol=True) or []
    except Exception:
        return []


def build_watchlist_feed(
    user: str, *, session: Optional[str] = None,
    prior_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Pure-ish assembly: tickers → batched rows → views + summary. Returns
    {views, summary, rows_by_symbol} (rows kept for next-render change detection)."""
    tickers = _watchlist_tickers(user)
    rows = _metrics_rows(tickers)
    views = wv.build_watchlist_views(rows, prior_by_symbol=prior_by_symbol,
                                     session=session)
    return {"views": views, "summary": wv.watchlist_summary(views),
            "rows_by_symbol": {str(r.get("ticker") or r.get("symbol") or "").upper(): r
                               for r in rows},
            "n_tickers": len(tickers)}


def _render_card(st, v: Dict[str, Any]) -> None:
    pi = _PRIORITY_ICON.get(v.get("alert_priority"), "⚪")
    li = _LIFECYCLE_ICON.get(v.get("lifecycle_state"), "")
    chg = v.get("change_pct")
    chg_txt = f"  ({chg:+.1f}%)" if isinstance(chg, (int, float)) else ""
    if v.get("no_active_setup"):
        st.markdown(f"⚪ **{v.get('symbol')}** — No active setup{chg_txt}")
        facts = v.get("no_setup_facts") or []
        if facts:
            st.caption(" · ".join(facts))
        return
    st.markdown(f"{pi} **{v.get('symbol')}** — {v.get('primary_setup') or 'Setup'} "
                f"· {v.get('alert_priority')} {li} {v.get('lifecycle_state')}{chg_txt}")
    if v.get("scanner_count", 0) >= 2:
        st.caption(f"{v['scanner_count']} scanners agree — "
                   + " • ".join(v.get("scanner_names") or []))
    changes = (v.get("changes_since_prior") or [])[:4]
    if changes:
        st.markdown("**What changed:** " + " · ".join(changes))
    reasons = (v.get("positive_reasons") or [])[:4]
    if reasons:
        st.markdown("  ".join(f"✅ {r}" for r in reasons))
    risks = (v.get("risk_reasons") or [])[:2]
    if risks:
        st.markdown("  ".join(f"⚠️ {r}" for r in risks))


def render_watchlist_intelligence(user: str, *, session: Optional[str] = None) -> None:
    """Mount point for the watchlist page. No-op headless; never raises."""
    try:
        import streamlit as st
    except Exception:
        return
    try:
        prior = st.session_state.get("_watchlist_prior_rows") or {}
        feed = build_watchlist_feed(user, session=session, prior_by_symbol=prior)
        views, summary = feed["views"], feed["summary"]

        if not views:
            st.info(wv.empty_watchlist_message())
            return

        st.markdown("### 🔎 Watchlist Intelligence")
        st.caption("Your personal market monitor. **Alert Priority is an attention "
                   "signal, not a prediction of return.**")
        cols = st.columns(5)
        cols[0].metric("Symbols", summary["total"])
        cols[1].metric("Needs Attention", summary["needs_attention"])
        cols[2].metric("New", summary["new_setups"])
        cols[3].metric("Strengthening", summary["strengthening"])
        cols[4].metric("Weakening", summary["weakening"])

        c1, c2 = st.columns([2, 1])
        flt = c1.selectbox("Filter", wv.watchlist_filters(), key="wl_intel_filter")
        sort_key = c2.selectbox("Sort", ["Attention", "Ticker", "% Change", "RVOL",
                                         "Priority", "Setup"], key="wl_intel_sort")
        shown = wv.sort_views(wv.filter_watchlist(views, flt), sort_key)

        for v in shown:
            with st.container():
                _render_card(st, v)
                st.divider()

        feed_events = wv.activity_feed(views)
        if feed_events:
            with st.expander("Recent changes", expanded=False):
                for e in feed_events:
                    st.markdown(f"- **{e['symbol']}** — {e['change']}")

        # Run 43: Historical Replay entry point (reachable from Watchlist).
        with st.expander("📽️ Signal Timeline (historical replay)", expanded=False):
            try:
                from ui.historical_replay import render_historical_replay
                syms = [v.get("symbol") for v in views if v.get("symbol")]
                pick = st.selectbox("Replay symbol", syms, key="wl_replay_pick") \
                    if syms else None
                if pick:
                    render_historical_replay(pick)
            except Exception:
                pass

        st.session_state["_watchlist_prior_rows"] = feed["rows_by_symbol"]
    except Exception:
        # Optional intelligence layer must never break basic watchlist.
        pass
