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

# P1-1: grouped navigation. Feature pages only — reset_password / verify_email
# are intentionally omitted (reachable by their email-link URLs and Settings).
# Alerts lives inside My Stocks; its own page stays reachable for shortcuts.
_NAV_SECTIONS = [
    ("Today", [
        ("pages/today.py", "Today", "📅"),
        ("pages/brief.py", "Market Brief", "📬"),
    ]),
    ("Discover", [
        ("app.py", "Scanner", "🔎"),
        ("pages/day_trader.py", "Day Trader", "⚡"),
    ]),
    ("Research", [
        ("pages/stock.py", "Stock Intelligence", "🔬"),
        ("pages/methodology.py", "How HSF works", "📖"),
    ]),
    ("My Stocks", [
        ("pages/watchlists.py", "My Stocks", "📋"),
        ("pages/journal.py", "Journal", "📓"),
    ]),
    ("Account", [
        ("pages/settings.py", "Settings", "⚙️"),
        ("pages/billing.py", "Billing", "💳"),
    ]),
    ("Labs", [
        ("pages/kalshi.py", "Kalshi BTC", "🪙"),
    ]),
]
# Flat list kept for callers/tests that only need the set of nav pages.
_NAV = [item for _section, items in _NAV_SECTIONS for item in items]


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
        from ui.chrome import hide_developer_chrome
        from ui.showcase import apply_showcase_styles

        hide_developer_chrome()
        apply_showcase_styles()
    except Exception:
        pass
    try:
        with st.sidebar:
            if with_header:
                _render_identity()
            else:
                # Main app renders its own identity block above us; add the
                # divider here so the nav is visually separated everywhere.
                st.divider()
            for section, items in _NAV_SECTIONS:
                st.caption(section.upper())
                for path, label, icon in items:
                    try:
                        st.page_link(path, label=label, icon=icon)
                    except Exception:
                        pass
    except Exception:
        pass
