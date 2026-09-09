"""Market brief panel: the morning-digest content, surfaced in the app UI.

Renders the same Top gappers / Today's setups / PreBreakout picks that the
morning email sends, computed from the latest daily snapshot. Reuses
scheduler.morning_digest so the email and the UI never drift. Never raises.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _compute_brief() -> Optional[Dict[str, Any]]:
    """Assemble gappers / setups / picks from the latest snapshot, or None."""
    try:
        from scheduler.morning_digest import (
            _earnings_days_map,
            _flag_earnings_rows,
            _latest_snapshot_df,
            _market_gappers,
            _prebreakout_picks,
            _todays_setups,
        )
    except Exception:
        return None
    try:
        df = _latest_snapshot_df()
    except Exception:
        df = None
    if df is None:
        return None

    gappers = _market_gappers(df) or []
    try:
        _flag_earnings_rows(gappers, _earnings_days_map([g.get("ticker") for g in gappers]))
    except Exception:
        pass
    try:
        golden, top_setups = _todays_setups(df)
    except Exception:
        golden, top_setups = [], []
    picks = _prebreakout_picks(df) or []
    try:
        pe = _earnings_days_map([p["symbol"] for p in picks])
        for p in picks:
            days = pe.get(p["symbol"])
            if days is not None:
                p["symbol"] = f"{p['symbol']} ⚠️E{days}d"
    except Exception:
        pass
    return {"gappers": gappers, "golden": golden, "top_setups": top_setups, "picks": picks}


if st is not None:

    @st.cache_data(ttl=300, show_spinner="Building your market brief…")
    def _brief_cached() -> Optional[Dict[str, Any]]:
        return _compute_brief()

else:  # pragma: no cover
    _brief_cached = _compute_brief


def render_market_brief() -> None:
    """Render the market brief (gappers / setups / PreBreakout picks). Never raises."""
    if st is None:
        return
    try:
        if st.button("🔄 Refresh", key="brief_refresh"):
            _brief_cached.clear()
            st.rerun()
        data = _brief_cached()
    except Exception:
        data = None
    if not data:
        st.info(
            "No recent scan snapshot yet — the market brief appears after the "
            "day's first scan runs (it's the same content as your morning email)."
        )
        return

    # 🚀 Top market gappers
    st.markdown("### 🚀 Top market gappers")
    gappers = data.get("gappers") or []
    if gappers:
        rows = []
        for g in gappers:
            chg, gap = g.get("chg_pct"), g.get("gap_pct")
            rows.append({
                "Ticker": g.get("ticker"),
                "Last": g.get("last"),
                "Chg %": f"{chg:+.2f}%" if chg is not None else "—",
                "Gap %": f"{gap:+.2f}%" if gap is not None else "—",
            })
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No gappers in the latest snapshot.")

    # 🎯 Today's setups
    st.markdown("### 🎯 Today's setups")
    golden = data.get("golden") or []
    top_setups = data.get("top_setups") or []
    if golden:
        st.markdown(f"📈 **Fresh EMA 9/21 golden crosses:** {', '.join(golden)}")
    if top_setups:
        ts = ", ".join(f"{t} ({s:g})" for t, s in top_setups)
        st.markdown(f"🚀 **Top breakout scores:** {ts}")
    if not golden and not top_setups:
        st.caption("No fresh setups in the latest snapshot.")
    st.caption("Educational only — not financial advice; confirm setups yourself at the open.")

    # 🧠 PreBreakout picks
    picks = data.get("picks") or []
    if picks:
        st.markdown("### 🧠 PreBreakout picks")
        for p in picks:
            st.markdown(f"- **{p['symbol']}** — {p['prob']}% model confidence")
