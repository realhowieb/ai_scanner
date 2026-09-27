"""⚙️ Settings — account, security, connected accounts, and quick links."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()  # Run 62/P2: before any sign-in gate

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage settings.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

# Sidebar nav (this page doesn't use render_page_logo, so render it directly).
try:
    from ui.nav import render_sidebar_nav

    render_sidebar_nav()
except Exception:
    pass

try:
    from ui.plan_labels import plan_label  # Run 85B: one customer-facing plan name

    tier = plan_label(st.session_state.get("tier_key") or st.session_state.get("tier"),
                      is_admin=bool(st.session_state.get("is_admin")))
except Exception:
    tier = "Free"
verified = None
try:
    from db.email_verification import is_email_verified

    verified = is_email_verified(_username)
except Exception:
    verified = None
try:
    can_paper = bool((st.session_state.get("entitlements") or {}).get("can_paper_trade"))
except Exception:
    can_paper = False

from ui.design_system import render_page_header  # noqa: E402

render_page_header("Settings", "Your account, connections and preferences.")

# --- Account ---
st.markdown("#### 👤 Account")
_display = (st.session_state.get("display_name") or "").strip()
ver_txt = "✅ Verified" if verified else ("— unknown" if verified is None else "❌ Not verified")
lines = []
if _display and _display.lower() != _username:
    lines.append(f"- **Name:** {_display}")
lines.append(f"- **Username:** {_username}")
lines.append(f"- **Plan:** {tier}")
lines.append(f"- **Email:** {ver_txt}")
st.markdown("\n".join(lines))
try:
    st.page_link("pages/billing.py", label="Manage plan & billing", icon="💳")
except Exception:
    pass

st.divider()

# --- Security (auth-utility pages, grouped here instead of the sidebar) ---
st.markdown("#### 🔒 Security")
try:
    st.page_link("pages/reset_password.py", label="Reset password", icon="🔑")
    if not verified:
        st.page_link("pages/verify_email.py", label="Verify email", icon="✉️")
except Exception:
    pass

st.divider()

# --- Connected accounts (Alpaca paper) ---
st.markdown("#### 🔗 Connected accounts")
if can_paper:
    try:
        from ui.paper_trade import render_connect_panel

        render_connect_panel(_username)
    except Exception:
        st.caption("Paper-account connection is unavailable right now.")
else:
    from ui.pricing import upgrade_message

    st.caption(upgrade_message("can_paper_trade"))

st.divider()

# --- Notifications & watchlist ---
st.markdown("#### 🔔 Notifications & data")
st.caption("Morning/evening briefs and email alerts go to Pro+ verified accounts.")
active = st.session_state.get("active_watchlist_tickers") or []
st.caption(
    f"Active watchlist: {len(active)} tickers"
    + (f" — {', '.join(active[:10])}{'…' if len(active) > 10 else ''}" if active else " (none set)")
)
try:
    a, b, c = st.columns(3)
    a.page_link("pages/brief.py", label="Market Brief", icon="📬")
    b.page_link("pages/alerts.py", label="Alerts", icon="🔔")
    c.page_link("pages/watchlists.py", label="My Stocks", icon="📋")
except Exception:
    pass

st.divider()

# --- Home screen (P2-6) ---
# Streamlit Cloud serves the page inside its own frame, so HSF can't ship a web-app
# manifest (installable PWA, offline, push). A home-screen shortcut still gives
# one-tap access; these are the platform's own steps.
st.markdown("#### Add HSF to your home screen")
st.markdown(
    "- **iPhone / iPad (Safari):** tap **Share**, then **Add to Home Screen**.\n"
    "- **Android (Chrome):** open the **⋮** menu, then **Add to Home screen**.\n"
    "- **Desktop (Chrome / Edge):** bookmark the page, or use **Install** / **Create shortcut** "
    "in the browser menu when offered."
)
st.caption("This adds a shortcut to HSF. It is not an offline app and does not send push notifications.")

st.divider()
st.page_link("app.py", label="← Back to scanner", icon="🏠")
