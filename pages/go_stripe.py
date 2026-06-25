"""Intermediate redirect page — calls billing service then meta-refreshes to Stripe."""
from __future__ import annotations

import json
import urllib.request

import streamlit as st

st.set_page_config(page_title="Redirecting to Stripe…", page_icon="💳")


def _get_checkout_url(email: str, plan: str) -> tuple[str | None, str | None]:
    try:
        from config import BILLING_API_BASE
        base = (BILLING_API_BASE or "").rstrip("/")
        if not base:
            return None, "BILLING_API_BASE not configured."
        payload = json.dumps({"email": email, "plan": plan}).encode()
        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        print(f"[go_stripe] billing response: {data}")
        url = data.get("url") or data.get("portal_url")
        return url, f"response={data}" if not url else None
    except Exception as e:
        return None, str(e)


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
    url, err = _get_checkout_url(email, plan)

if not url:
    st.error(f"Could not connect to billing service: {err}")
    st.caption("The billing service may be sleeping (free Render tier). Wait 30 seconds and try again.")
    if st.button("🔄 Retry"):
        st.rerun()
    st.page_link("app.py", label="← Back")
    st.stop()

st.success("Checkout ready!")
st.link_button("💳 Continue to Stripe", url, use_container_width=True)
st.caption("Opens in a new tab. After payment, return to this tab — your plan will update automatically.")
