"""P1-5 — Today: the daily entry point, readable in ~10 seconds.

Single column, card-based (phone-first):
  market status line → market snapshot → top setups (canonical HSF ranking of
  the latest full-market scan) → new since your last visit → your watchlist in
  today's scan → session recap.

Read-only: everything comes from saved scheduled scans, the persisted health
snapshot and the user's watchlist. No performance numbers, no research state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def watchlist_in_scan(watch: List[str], scan_df: Any) -> Dict[str, Any]:
    """Split watchlist tickers into those in the latest scan (with HSF Score
    when they qualify) and those not in today's ranked list."""
    from ui.headline_score import hsf_scores_by_ticker
    from ui.market_scans import tickers_of

    in_scan = set(tickers_of(scan_df))
    scores = hsf_scores_by_ticker(scan_df.to_dict(orient="records")) if in_scan else {}
    found = [{"ticker": t, "score": scores.get(t)} for t in watch if t in in_scan]
    missing = [t for t in watch if t not in in_scan]
    return {"found": found, "missing": missing}


def _open_button(ticker: str, key: str) -> None:
    if st.button("Open", key=key):
        st.session_state["hsf_stock_ticker"] = ticker
        st.session_state.pop("hsf_stock_opp", None)
        st.switch_page("pages/stock.py")


def _user_watchlist(username: str) -> List[str]:
    tick = st.session_state.get("active_watchlist_tickers")
    if tick:
        return [str(t).strip().upper() for t in tick if str(t).strip()]
    try:
        from db.watchlists import get_default_watchlist_id, get_watchlist_tickers

        wid = get_default_watchlist_id(username)
        return [str(t).strip().upper() for t in (get_watchlist_tickers(wid, username) if wid else [])]
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("your watchlist", exc)
        return []


def _section_top(scan_df: Any) -> None:
    from ui.market_scans import top_setups

    st.markdown("### Top setups right now")
    tops = top_setups(scan_df, n=5)
    if not tops:
        st.caption("No ranked setups in the latest scan.")
        return
    for i, o in enumerate(tops):
        with st.container(border=True):
            setup = f" · {o['primary_setup']}" if o.get("primary_setup") not in (None, "", "Signal") else ""
            st.markdown(f"**{o['ticker']}** · HSF Score **{o['score']}**{setup} · {o['status'].title()}")
            facts = []
            if o.get("last") is not None:
                facts.append(f"${o['last']:,.2f}")
            if o.get("chg_pct") is not None:
                facts.append(f"{o['chg_pct']:+.2f}% today")
            if o.get("rvol") is not None:
                facts.append(f"RVOL {o['rvol']:.2f}×")
            if facts:
                st.caption(" · ".join(facts))
            _open_button(o["ticker"], f"today_top_{i}_{o['ticker']}")
    st.caption("Ranked by HSF Score, an opportunity ranking — not a probability of profit.")


def _section_new(scan_df: Any, run_id: Optional[int]) -> None:
    from ui.last_visit import new_since_last_visit

    new = sorted(new_since_last_visit(scan_df, shown_run_id=run_id))
    st.markdown("### New since your last visit")
    if not new:
        st.caption("Nothing new since you last looked, or this is your first visit on this browser.")
        return
    st.markdown(", ".join(f"**{t}**" for t in new[:15]) + (f" and {len(new) - 15} more" if len(new) > 15 else ""))
    if st.button(f"Show all {len(new)} in the Scanner", key="today_show_new"):
        st.session_state["hsf_lens"] = "new"
        st.switch_page("app.py")


def _section_watchlist(username: str, scan_df: Any) -> None:
    st.markdown("### Your watchlist")
    watch = _user_watchlist(username)
    if not watch:
        st.caption("Your watchlist is empty. Add names from the Scanner or Stock Intelligence.")
        return
    w = watchlist_in_scan(watch, scan_df)
    if w["found"]:
        for i, r in enumerate(w["found"]):
            with st.container(border=True):
                score = f" · HSF Score **{r['score']}**" if r["score"] is not None else ""
                st.markdown(f"**{r['ticker']}** is in the latest scan{score}")
                _open_button(r["ticker"], f"today_wl_{i}_{r['ticker']}")
    else:
        st.caption("None of your watched names are in the latest scan's ranked list.")
    if w["missing"]:
        st.caption("Not in the latest ranked list: " + ", ".join(w["missing"][:20]))


def render_today(username: str) -> None:
    """Render the Today page body. Never raises."""
    if st is None:
        return
    from ui.market_scans import safe_recent_runs, safe_run_df
    from ui.product_copy import TAGLINE

    st.title("Today")
    st.caption(TAGLINE)
    try:
        from ui.trust_banner import render_trust_banner

        render_trust_banner()
    except Exception:
        pass
    runs = safe_recent_runs()
    run_id = int(runs[0]["id"]) if runs else None
    scan_df = safe_run_df(run_id)
    if scan_df is None or scan_df.empty:
        st.info("The latest market scan is unavailable right now. The Scanner and Market Brief "
                "will show results as soon as the next scheduled scan completes.")
        return
    from ui.header import render_market_snapshot

    for name, fn in (("market snapshot", lambda: render_market_snapshot(results_df=scan_df)),
                     ("top setups", lambda: _section_top(scan_df)),
                     ("new since your last visit", lambda: _section_new(scan_df, run_id)),
                     ("your watchlist", lambda: _section_watchlist(username, scan_df))):
        try:
            fn()
        except Exception as exc:
            from ui.safe_errors import show_error

            show_error(name, exc)
    from ui.recap import render_recap

    render_recap()
    st.page_link("app.py", label="Open the Scanner", icon="🔎")
    st.page_link("pages/brief.py", label="Market Brief", icon="📬")
