"""Run 28 first-run onboarding for the existing HSF product workflow.

This module teaches the workflow around persisted watchlists and existing HSF
intelligence. It does not scan, score, freeze, mature outcomes, evaluate alerts,
deliver alerts, or call an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


STARTER_TICKERS = ("AAPL", "NVDA", "MSFT", "TSLA", "AMD")


@dataclass(frozen=True)
class FirstTickerResult:
    status: str
    ticker: str
    message: str
    opportunity: Optional[Dict[str, Any]] = None


def user_is_first_run(watchlist: List[object] | None) -> bool:
    """First-run is inferred from persisted watchlist emptiness."""
    return len(_normalize_tickers(watchlist or [])) == 0


def activation_reached(watchlist: List[object] | None, viewed_personal_intel: bool) -> bool:
    """Canonical Run 28 activation definition."""
    return bool(_normalize_tickers(watchlist or [])) and bool(viewed_personal_intel)


def current_opportunity_for_ticker(
    ticker: str,
    *,
    snapshots_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Read the latest persisted Market Brief opportunity for one ticker."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return None
    snapshots = snapshots_loader() if snapshots_loader else _load_recent_opportunity_snapshots()
    current = snapshots[0] if snapshots else {}
    for row in (current or {}).get("opportunities") or []:
        if _normalize_ticker(row.get("ticker") or row.get("Ticker") or row.get("Symbol")) == symbol:
            return dict(row)
    return None


def add_first_watch_ticker(
    user_id: str,
    ticker: object,
    *,
    existing_watchlist: Optional[List[object]] = None,
    watchlist_loader: Optional[Callable[[str], List[str]]] = None,
    add_fn: Optional[Callable[[str, str], bool]] = None,
    snapshots_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> FirstTickerResult:
    """Validate/add a first-run ticker and compose its immediate HSF context."""
    user = str(user_id or "").strip().lower()
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return FirstTickerResult(
            status="invalid",
            ticker="",
            message="We couldn't recognize that ticker. Try a symbol like NVDA.",
        )

    watched = (
        _normalize_tickers(existing_watchlist)
        if existing_watchlist is not None
        else _normalize_tickers(watchlist_loader(user) if watchlist_loader else _load_user_watchlist(user))
    )
    duplicate = symbol in set(watched)
    if not duplicate:
        ok = add_fn(user, symbol) if add_fn else _add_to_persisted_watchlist(user, symbol)
        if not ok:
            return FirstTickerResult(
                status="save_failed",
                ticker=symbol,
                message="We could not save that ticker right now. Please try again.",
            )

    opportunity = current_opportunity_for_ticker(symbol, snapshots_loader=snapshots_loader)
    if duplicate:
        return FirstTickerResult(
            status="duplicate",
            ticker=symbol,
            message=f"{symbol} is already in your watchlist.",
            opportunity=opportunity,
        )
    return FirstTickerResult(
        status="added",
        ticker=symbol,
        message=f"{symbol} added to your watchlist.",
        opportunity=opportunity,
    )


def render_hsf_onboarding_entry(username: str, *, tier_name: str = "") -> None:
    """Render first-run or returning-user landing value."""
    if st is None:
        return
    user = str(username or "").strip().lower()
    watchlist = _load_user_watchlist(user)
    first_run = user_is_first_run(watchlist)
    st.session_state["hsf_first_run"] = first_run
    if first_run:
        _render_first_run(user)
    else:
        _render_returning_user(user, watchlist, tier_name=tier_name)


def render_market_brief_orientation(username: str) -> None:
    _render_dismissible_orientation(
        username,
        "market_brief",
        "Market Brief",
        "HSF Market Brief summarizes the strongest current opportunities and meaningful market changes.",
    )


def render_scanner_orientation(username: str) -> None:
    _render_dismissible_orientation(
        username,
        "scanner",
        "Scanner",
        "Discover stocks currently matching HSF market conditions. Use filters to narrow results, then watch names you want HSF to track.",
    )


def render_stock_intelligence_orientation(username: str) -> None:
    _render_dismissible_orientation(
        username,
        "stock_intelligence",
        "Reading HSF Intelligence",
        "HSF Score summarizes current strength; HSF status classifies the state; confirming signals explain what supports it; history appears when enough evidence exists.",
    )


def render_alerts_orientation(username: str) -> None:
    _render_dismissible_orientation(
        username,
        "alerts",
        "HSF Intelligence Alerts",
        "Get notified when watched stocks experience meaningful HSF changes such as upgrades, downgrades, fading, entering, or leaving the opportunity ranking.",
    )


def render_intelligence_performance_orientation(username: str) -> None:
    _render_dismissible_orientation(
        username,
        "intelligence_performance",
        "Intelligence Performance",
        "HSF tracks how historical intelligence states evolve so you can see where evidence has accumulated and where history is still limited.",
    )


def _render_first_run(user: str) -> None:
    from ui.design_system import render_page_header

    render_page_header("HSF Market Intelligence", "Know what changed. Understand why it matters. Track the stocks you care about.")
    st.caption(
        "HSF organizes market signals into one workflow: what deserves attention, "
        "what is strengthening, what is fading, and what changed in your watchlist."
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        with st.container(border=True):
            st.markdown("#### Add your first stock")
            st.caption("Add a ticker to start your personalized HSF watchlist.")
            with st.form("hsf_first_ticker_form", clear_on_submit=False):
                ticker = st.text_input("What stocks do you care about?", placeholder="NVDA")
                submitted = st.form_submit_button("Build My Watchlist")
            st.caption("Popular examples: " + "  ·  ".join(STARTER_TICKERS))
            if submitted:
                result = add_first_watch_ticker(user, ticker)
                _remember_added_ticker(result)
                _render_first_value(result)
    with c2:
        st.markdown("#### How HSF works")
        st.markdown("1. **Discover** current market opportunities.")
        st.markdown("2. **Watch** stocks you care about.")
        st.markdown("3. **Understand** HSF status, changes, and context.")
        st.caption("HSF can notify you when watched intelligence changes.")
        st.page_link("pages/brief.py", label="Explore Market", icon="📬")

    with st.expander("HSF Score and status", expanded=False):
        st.markdown(
            """
**HSF Score** is a 0-100 summary of the strength of the current HSF opportunity state.
Higher scores indicate stronger alignment across HSF's current market signals.

**STRONG** means strong alignment across current HSF signals.
**WATCH** means a meaningful setup worth monitoring.
**CAUTION** means the current HSF state has weakened or lacks enough strength.
**FADING** means previously stronger conditions are deteriorating.
"""
        )
        st.caption("HSF Score is market intelligence, not a recommendation to buy or sell.")


def _render_returning_user(user: str, watchlist: List[str], *, tier_name: str = "") -> None:
    try:
        from analytics.watchlist_intelligence import build_watchlist_intelligence

        intel = build_watchlist_intelligence(user, watchlist=watchlist)
    except Exception:
        intel = {"summary": {"tracked": len(watchlist)}, "groups": {}}
    summary = intel.get("summary") or {}
    attention = int(summary.get("needs_attention") or 0)
    strengthening = int(summary.get("strengthening") or 0)
    fading = int(summary.get("fading") or 0)
    tier = f" · {tier_name}" if tier_name else ""
    st.markdown(f"### Your HSF market view{tier}")
    st.caption(
        f"{len(watchlist)} watched stock(s) · {attention} need attention · "
        f"{strengthening} strengthening · {fading} fading"
    )
    c1, c2 = st.columns(2)
    c1.page_link("pages/watchlists.py", label="Open My Watchlist", icon="📋")
    c2.page_link("pages/brief.py", label="Explore Market Brief", icon="📬")


def _render_first_value(result: FirstTickerResult) -> None:
    if result.status == "invalid":
        st.warning(result.message)
        return
    if result.status == "save_failed":
        st.error(result.message)
        return

    st.success(result.message)
    opp = result.opportunity
    if opp:
        score = opp.get("score")
        try:
            score_text = f"{float(score):.0f}"
        except (TypeError, ValueError):
            score_text = "—"
        signals = len(opp.get("signals") or [])
        st.markdown(
            f"**HSF currently classifies {result.ticker} as {opp.get('status') or 'ranked'}**  \n"
            f"HSF Score: **{score_text}** · {signals} confirming signal(s)"
        )
        st.session_state["hsf_stock_ticker"] = result.ticker
        st.session_state["hsf_stock_opp"] = opp
        st.page_link("pages/stock.py", label="View Full Intelligence", icon="🔬")
    else:
        st.markdown(
            f"**{result.ticker} is not currently ranked as an HSF opportunity.**  \n"
            "HSF will continue organizing meaningful changes for watched stocks."
        )
        st.page_link("pages/watchlists.py", label="Open My Watchlist", icon="📋")


def _render_dismissible_orientation(username: str, key: str, title: str, body: str) -> None:
    if st is None:
        return
    session_key = f"hsf_orientation_dismissed::{key}::{str(username or '').strip().lower()}"
    if st.session_state.get(session_key):
        return
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"#### {title}")
        c1.caption(body)
        if c2.button("Got it", key=session_key):
            st.session_state[session_key] = True
            st.rerun()


def _remember_added_ticker(result: FirstTickerResult) -> None:
    if st is None or result.status not in {"added", "duplicate"} or not result.ticker:
        return
    current = {
        str(t).strip().upper()
        for t in (st.session_state.get("active_watchlist_tickers") or [])
        if str(t).strip()
    }
    current.add(result.ticker)
    st.session_state["active_watchlist_tickers"] = sorted(current)
    st.session_state["hsf_onboarding_first_ticker"] = result.ticker
    st.session_state["hsf_first_watchlist_add"] = True
    if activation_reached(sorted(current), True):
        st.session_state["hsf_activation_reached"] = True


def _normalize_ticker(value: object) -> str:
    try:
        from db.watchlists import normalize_watchlist_ticker

        return normalize_watchlist_ticker(value)
    except Exception:
        ticker = str(value or "").strip().upper()
        return ticker if ticker else ""


def _normalize_tickers(values: List[object] | None) -> List[str]:
    try:
        from db.watchlists import normalize_watchlist_tickers

        return normalize_watchlist_tickers(list(values or []))
    except Exception:
        return sorted({t for t in (_normalize_ticker(v) for v in (values or [])) if t})


def _load_user_watchlist(user: str) -> List[str]:
    try:
        from db.watchlists import get_user_watchlist

        return get_user_watchlist(user)
    except Exception:
        return []


def _add_to_persisted_watchlist(user: str, ticker: str) -> bool:
    try:
        from db.watchlists import add_to_watchlist

        return bool(add_to_watchlist(user, ticker))
    except Exception:
        return False


def _load_recent_opportunity_snapshots() -> List[Dict[str, Any]]:
    try:
        from db.opportunity_snapshots import load_recent_snapshots

        return load_recent_snapshots(context="market_brief", limit=1)
    except Exception:
        return []
