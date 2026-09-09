"""⚙️ Settings — account, security, connected accounts, and quick links."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage settings.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.app_session import tier_key

    tier = (tier_key(st.session_state.get("tier")) or "basic").upper()
except Exception:
    tier = "BASIC"
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

st.markdown("## ⚙️ Settings")

# --- Account ---
st.markdown("#### 👤 Account")
ver_txt = "✅ Verified" if verified else ("— unknown" if verified is None else "❌ Not verified")
st.markdown(
    f"- **Email:** {_username}\n"
    f"- **Plan:** {tier}\n"
    f"- **Email status:** {ver_txt}"
)
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
    st.caption("Connect an Alpaca **paper** account to place practice trades — a "
               "Premium feature.")

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
    c.page_link("pages/watchlists.py", label="Watchlists", icon="📋")
except Exception:
    pass

st.divider()
st.page_link("app.py", label="← Back to scanner", icon="🏠")
