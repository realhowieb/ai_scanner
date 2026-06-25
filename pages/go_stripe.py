"""Intermediate redirect page — calls billing service then opens Stripe checkout.

Carries a return token (rt) in the Stripe success_url so the session can be
restored after the redirect without relying on browser cookies (which are
unreliable on Streamlit Cloud).
"""
from __future__ import annotations

import json
import urllib.request

import streamlit as st

st.set_page_config(page_title="Redirecting to Stripe…", page_icon="💳")


def _get_checkout_url(
    email: str, plan: str, success_url: str | None, return_url: str | None
) -> tuple[str | None, str | None]:
    try:
        from config import BILLING_API_BASE
        base = (BILLING_API_BASE or "").rstrip("/")
        if not base:
            return None, "BILLING_API_BASE not configured."
        body: dict[str, str] = {"email": email, "plan": plan}
        if success_url:
            body["success_url"] = success_url
        if return_url:
            body["return_url"] = return_url
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        url = data.get("checkout_url") or data.get("url") or data.get("portal_url")
        return url, f"response={data}" if not url else None
    except Exception as e:
        return None, str(e)


def _build_return_urls(username: str) -> tuple[str | None, str | None]:
    """Mint a session for the user; embed its id in both the checkout success
    URL and the portal return URL so the session restores without a cookie."""
    try:
        from config import APP_BASE_URL
        from ui.auth_sessions import create_session
        sid = create_session(username)
        if not sid:
            return None, None
        base = (APP_BASE_URL or "").rstrip("/")
        success = f"{base}/?checkout=success&rt={sid}"
        portal = f"{base}/?portal=return&rt={sid}"
        return success, portal
    except Exception:
        return None, None


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

with st.spinner("Preparing your checkout… (may take 30s if billing service is waking up)"):
    success_url, return_url = _build_return_urls(email)
    url, err = _get_checkout_url(email, plan, success_url, return_url)

if not url:
    st.error(f"Could not connect to billing service: {err}")
    st.caption("The billing service may be sleeping (free Render tier). Wait 30 seconds and try again.")
    if st.button("🔄 Retry"):
        st.rerun()
    st.page_link("app.py", label="← Back")
    st.stop()

st.success("Checkout ready!")
st.link_button("💳 Continue to Stripe", url, use_container_width=True)
st.caption("After payment you'll be returned here automatically — your plan will update.")
