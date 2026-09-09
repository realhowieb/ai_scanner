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


def _render_identity() -> None:
    """Compact account header (name · plan · log out) from session state.

    Gives sub-pages the same identity block the main app renders, so the sidebar
    is consistent everywhere. Never raises.
    """
    try:
        name = (st.session_state.get("display_name")
                or st.session_state.get("username") or "").strip()
        if "@" in name:
            name = name.split("@")[0]
        if not name:
            return
        is_admin = bool(st.session_state.get("is_admin"))
        plan = "Admin" if is_admin else str(st.session_state.get("tier_key") or "basic").title()
        st.markdown(f"### 👤 {name}")
        st.markdown(f"**Plan:** `{plan}`")
        if st.button("Log out", key="nav_logout"):
            try:
                from ui.auth import logout_and_reset_session

                logout_and_reset_session()
            except Exception:
                pass
        st.divider()
    except Exception:
        pass


def render_sidebar_nav(*, with_header: bool = True) -> None:
    """Render the curated sidebar navigation. Safe to call on every page.

    with_header adds the identity block; the main app passes False because it
    renders its own (richer) account sidebar.
    """
    if st is None:
        return
    try:
        with st.sidebar:
            if with_header:
                _render_identity()
            else:
                # Main app renders its own identity block above us; add the
                # divider here so the nav is visually separated everywhere.
                st.divider()
            for path, label, icon in _NAV:
                try:
                    st.page_link(path, label=label, icon=icon)
                except Exception:
                    pass
    except Exception:
        pass
