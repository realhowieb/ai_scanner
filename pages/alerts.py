"""🔔 Alerts: dedicated page for alert management.

Create/manage alerts, outcome scorecards, and the triggered feed live here;
the main page keeps the lightweight notification bell and a link. In-context
creation survives via the "Alert me on this" quick action on result details,
which pre-fills the price-alert form before this page renders it.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage alerts.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()


def _session_watch_tickers() -> list[str]:
    """Use tickers already loaded on the main page; avoid DB work before first paint."""
    tickers = st.session_state.get("active_watchlist_tickers") or []
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def _tier_limits() -> tuple[int, bool]:
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


try:
    from ui.alerts import render_alerts_panel

    _max_alerts, _email_ok = _tier_limits()
    render_alerts_panel(
        _username,
        watch_tickers=_session_watch_tickers(),
        max_alerts=_max_alerts,
        email_enabled=_email_ok,
    )
except Exception as e:
    st.error("Alerts failed to load.")
    st.caption(f"{type(e).__name__}: {e}")

st.page_link("app.py", label="← Back to scanner", icon="🏠")
