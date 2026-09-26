"""Shared alerts body (P1-6): used by the Alerts page and the My Stocks page.

Moved verbatim from pages/alerts.py so "My Stocks" can show watchlists and
alerts together while the Alerts URL (the target of every "Alert me on this"
shortcut) keeps working unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st


def session_watch_tickers() -> List[str]:
    """Use tickers already loaded on the main page; avoid DB work before first paint."""
    tickers = st.session_state.get("active_watchlist_tickers") or []
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def tier_limits() -> Tuple[int, bool]:
    """(max_alerts, email_enabled) for the current session's tier.

    Normalize through tier_key(): session state can hold a Tier object or a
    differently-cased value, and a raw string compare silently downgraded
    Premium accounts to Basic limits here.
    """
    raw = st.session_state.get("tier") or st.session_state.get("plan")
    try:
        from ui.app_session import alert_limit_for_tier, tier_key

        key = tier_key(raw) or "basic"
        is_admin = key == "admin"
        max_alerts = 25 if is_admin else alert_limit_for_tier(key)
    except Exception:
        key = str(raw or "basic").strip().lower()
        is_admin = key == "admin"
        max_alerts = 25 if is_admin else 1
    email_ok = key in ("pro", "premium", "admin")
    return int(max_alerts), bool(email_ok)


def render_alerts_body(username: str, *, watch_tickers: Optional[List[str]] = None) -> None:
    """Intelligence alerts + create/manage alerts. Never raises."""
    try:
        from ui.alerts import render_alerts_panel

        # HSF Intelligence Alerts (state-change) sit above the existing static alerts.
        try:
            from ui.intelligence_alerts_ui import render_intelligence_alerts

            st.caption(
                "HSF Intelligence alerts are generated for watched opportunities by background detection; "
                "opening this page only reads saved alert state."
            )
            render_intelligence_alerts(username)
            st.markdown("---")
        except Exception:
            pass
        max_alerts, email_ok = tier_limits()
        render_alerts_panel(
            username,
            watch_tickers=watch_tickers if watch_tickers is not None else session_watch_tickers(),
            max_alerts=max_alerts,
            email_enabled=email_ok,
        )
    except Exception as e:
        from ui.safe_errors import show_error

        show_error("your alerts", e)
