"""Custom sidebar navigation.

With `client.showSidebarNavigation = false`, Streamlit's auto page list is off,
so we render our own — which lets us curate order, labels, and icons, and keep
the auth-utility pages (reset password / verify email) OUT of the main nav while
they stay reachable by their email-link URLs (and from Settings). Never raises.
"""
from __future__ import annotations

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

# Feature pages only — reset_password / verify_email are intentionally omitted.
_NAV = [
    ("app.py", "Scanner", "🔎"),
    ("pages/brief.py", "Market Brief", "📬"),
    ("pages/day_trader.py", "Day Trader", "⚡"),
    ("pages/watchlists.py", "Watchlists", "📋"),
    ("pages/alerts.py", "Alerts", "🔔"),
    ("pages/journal.py", "Journal", "📓"),
    ("pages/kalshi.py", "Kalshi BTC", "🪙"),
    ("pages/settings.py", "Settings", "⚙️"),
    ("pages/billing.py", "Billing", "💳"),
]


def render_sidebar_nav() -> None:
    """Render the curated sidebar navigation. Safe to call on every page."""
    if st is None:
        return
    try:
        with st.sidebar:
            for path, label, icon in _NAV:
                try:
                    st.page_link(path, label=label, icon=icon)
                except Exception:
                    pass
    except Exception:
        pass
