"""Intermediate redirect page — calls billing service then meta-refreshes to Stripe."""
from __future__ import annotations

import json
import urllib.request

import streamlit as st

st.set_page_config(page_title="Redirecting to Stripe…", page_icon="💳")


def _get_checkout_url(email: str, plan: str) -> str | None:
    try:
        from config import BILLING_API_BASE
        base = (BILLING_API_BASE or "").rstrip("/")
        if not base:
            return None
        payload = json.dumps({"email": email, "plan": plan}).encode()
        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("url") or data.get("portal_url")
    except Exception:
        return None


plan = (st.session_state.get("_upgrade_plan") or st.query_params.get("plan") or "").strip().lower()
email = st.session_state.get("username", "")

if not plan or plan not in {"pro", "premium"}:
    st.error("Invalid plan. Please go back and try again.")
    st.page_link("app.py", label="← Back")
    st.stop()

if not email:
    st.warning("Session expired. Please log in first.")
    st.page_link("app.py", label="← Log in")
    st.stop()

with st.spinner("Preparing your checkout…"):
    url = _get_checkout_url(email, plan)

if not url:
    st.error("Could not connect to billing service. Please try again.")
    st.page_link("app.py", label="← Back")
    st.stop()

# Meta-refresh in the main Streamlit page (not an iframe) navigates the real browser tab.
st.markdown(
    f"""
    <meta http-equiv="refresh" content="0; url={url}">
    <p>Redirecting to Stripe… <a href="{url}">Click here if not redirected.</a></p>
    """,
    unsafe_allow_html=True,
)
