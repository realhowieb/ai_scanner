"""⚡ Day Trader — live: dedicated full page for the intraday monitor.

Own page so it can stay open all session (auto-refresh polling only this view,
not the whole scanner) and get maximum width for the table.
"""
from __future__ import annotations

import streamlit as st

from ui.showcase import apply_showcase_styles, initial_sidebar_state

st.set_page_config(page_title="Day Trader — live", page_icon="⚡", layout="wide",
                   initial_sidebar_state=initial_sidebar_state())
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # Run 62/P2: before any sign-in gate
apply_showcase_styles()

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to use the Day Trader monitor.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()


def _can_use_day_trader() -> bool:
    """Day Trader is Pro+ (FEATURE_MIN_TIER["can_day_trader"]). Sessions whose
    entitlements predate the flag fall back to the resolved tier."""
    ss = st.session_state
    if ss.get("is_admin"):
        return True
    ent = ss.get("entitlements") or {}
    if "can_day_trader" in ent:
        return bool(ent["can_day_trader"])
    try:
        from auth.tiering import has_min_tier

        return bool(has_min_tier(ss.get("tier_key") or "basic", "pro"))
    except Exception:
        return False


if not _can_use_day_trader():
    try:
        from ui.design_system import render_page_header
        from ui.header import render_page_logo
        from ui.pricing import upgrade_message

        render_page_logo()
        render_page_header("Day Trader", "Live movers: gappers, VWAP and relative volume, in real time.")
        msg = upgrade_message("can_day_trader")
        if "Day Trader" not in msg:  # stale pricing module mid-redeploy
            msg = "Pro adds the live Day Trader monitor: intraday movers, VWAP and relative volume."
        st.info(msg)
        from ui.app_runtime import _upgrade_button

        _upgrade_button("Upgrade to Pro", "pro", "upgrade_to_pro_day_trader")
    except Exception:
        st.info("The live Day Trader monitor is part of Pro.")
    st.page_link("pages/billing.py", label="Compare all plans", icon="💳")
    st.page_link("app.py", label="← Back to scanner", icon="🏠")
    st.stop()


def _session_watch_tickers() -> list[str]:
    """Use tickers already loaded on the main page; avoid DB work before first paint."""
    tickers = st.session_state.get("active_watchlist_tickers") or []
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


try:
    from ui.day_trader import render_day_trader_panel
    from ui.header import render_page_logo

    render_page_logo()
    render_day_trader_panel(watch_tickers=_session_watch_tickers())
except Exception as e:
    from ui.safe_errors import show_error

    show_error("the Day Trader monitor", e)

st.page_link("app.py", label="← Back to scanner", icon="🏠")
