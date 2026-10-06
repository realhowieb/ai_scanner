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
        ("pages/custom_scan.py", "Custom Scan", "🧪"),
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
    ("Admin", [
        ("pages/admin.py", "Admin Console", "🛠️"),
    ]),
    ("Labs", [
        ("pages/kalshi.py", "Kalshi BTC", "🪙"),
    ]),
]
# Flat list kept for callers/tests that only need the set of nav pages.
_NAV = [item for _section, items in _NAV_SECTIONS for item in items]

# P2-25 (owner, 2026-09-29): Labs (Kalshi BTC) is admin-only. The page checks
# the database itself; hiding the section here just keeps the menu clean.
ADMIN_ONLY_SECTIONS = frozenset({"Admin", "Labs"})


def _visible_sections():
    is_admin = bool(st is not None and st.session_state.get("is_admin"))
    return [(s, items) for s, items in _NAV_SECTIONS if is_admin or s not in ADMIN_ONLY_SECTIONS]


def _render_identity(*, key_suffix: str = "sidebar") -> None:
    """The shared account / plan card (ui.account_card) — the same component on
    every page; the phone menu gets its compact form. Never raises."""
    try:
        from ui.account_card import render_account_card

        render_account_card(key_suffix=key_suffix, compact=key_suffix == "mobile")
    except Exception:
        pass


TOP_MENU_KEY = "hsf_top_menu"


def render_top_menu() -> None:
    """Run 67: phone-width menu at the top of the page (the sidebar hides behind
    an arrow on phones). CSS in ui.chrome shows it only under 640px. Never raises."""
    if st is None:
        return
    try:
        with st.container(key=TOP_MENU_KEY):
            with st.popover("☰ Menu"):
                _render_identity(key_suffix="mobile")
                for section, items in _visible_sections():
                    st.caption(section.upper())
                    for path, label, icon in items:
                        try:
                            st.page_link(path, label=label, icon=icon)
                        except Exception:
                            pass
    except Exception:
        pass


def render_sidebar_nav(*, with_header: bool = True) -> None:
    """Render the curated sidebar navigation. Safe to call on every page.

    with_header adds the shared account card; the main app passes False because
    its account sidebar renders the same card (ui.account_card) itself.
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
    # Run 72: navigation is for signed-in users; a signed-out visitor (e.g. on
    # the public Plans page) would only get "please sign in" behind every link.
    if not str(st.session_state.get("username") or "").strip():
        return
    render_top_menu()
    try:
        with st.sidebar:
            if with_header:
                _render_identity()
            for section, items in _visible_sections():
                st.caption(section.upper())
                for path, label, icon in items:
                    try:
                        st.page_link(path, label=label, icon=icon)
                    except Exception:
                        pass
    except Exception:
        pass
