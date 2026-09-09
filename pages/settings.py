"""⚙️ Settings — account, connected accounts, notifications, quick links.

A consolidated control panel that surfaces things previously scattered across
the app (paper-account connection, tier, billing). Reuses existing panels.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

_username = (st.session_state.get("username") or "").strip().lower()
if not _username:
    st.info("Please log in on the main page to manage settings.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

st.markdown("## ⚙️ Settings")

# --- Account ---
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

st.markdown("### 👤 Account")
c1, c2, c3 = st.columns(3)
c1.metric("Email", _username)
c2.metric("Plan", tier)
c3.metric("Email verified", "✅ Yes" if verified else ("—" if verified is None else "❌ No"))
try:
    st.page_link("pages/billing.py", label="Manage plan & billing", icon="💳")
except Exception:
    pass

# --- Notifications ---
st.markdown("### 🔔 Notifications")
st.caption(
    "Morning & evening briefs and email alerts are sent to **Pro+ verified** "
    "accounts. You can also pull the brief any time on the 📬 Brief page."
)
try:
    st.page_link("pages/brief.py", label="Open Market Brief", icon="📬")
    st.page_link("pages/alerts.py", label="Manage alerts", icon="🔔")
except Exception:
    pass

# --- Security ---
st.markdown("### 🔒 Security")
try:
    st.page_link("pages/reset_password.py", label="Reset password", icon="🔑")
    if not verified:
        st.page_link("pages/verify_email.py", label="Verify email", icon="✉️")
except Exception:
    pass

# --- Connected accounts (Alpaca paper) ---
st.markdown("### 🔗 Connected accounts")
try:
    from ui.paper_trade import render_connect_panel

    render_connect_panel(_username)
except Exception:
    pass

# --- Active watchlist ---
st.markdown("### 📋 Watchlist")
active = st.session_state.get("active_watchlist_tickers") or []
if active:
    st.caption(f"Active watchlist: {len(active)} tickers — {', '.join(active[:12])}"
               + ("…" if len(active) > 12 else ""))
else:
    st.caption("No active watchlist yet.")
try:
    st.page_link("pages/watchlists.py", label="Manage watchlists", icon="📋")
except Exception:
    pass

st.markdown("---")
st.page_link("app.py", label="← Back to scanner", icon="🏠")
