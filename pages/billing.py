from __future__ import annotations

import os
import time

import requests
import streamlit as st
from requests import exceptions as req_exc

# =========================
# Config
# =========================

BILLING_API_BASE = (os.getenv("BILLING_API_BASE") or "https://ai-scanner-h2c8.onrender.com").strip()

# Render free/idle services can cold-start
HEALTH_TIMEOUT_S = float(os.getenv("BILLING_HEALTH_TIMEOUT", "3"))
POST_TIMEOUT_S = float(os.getenv("BILLING_POST_TIMEOUT", "45"))

# Retry a couple times for cold starts
POST_RETRIES = int(os.getenv("BILLING_POST_RETRIES", "2"))
POST_RETRY_SLEEP_S = float(os.getenv("BILLING_POST_RETRY_SLEEP", "1.0"))


# =========================
# Billing service helpers
# =========================

def _billing_healthcheck() -> bool:
    """Return True if billing service is reachable quickly."""
    try:
        r = requests.get(f"{BILLING_API_BASE}/health", timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except Exception:
        return False


def _billing_post_json(path: str, payload: dict) -> dict:
    """POST JSON to billing service with cold-start-friendly behavior."""
    # Fast preflight: avoids waiting on a long POST when the service is asleep.
    if not _billing_healthcheck():
        raise RuntimeError(
            "Billing service is waking up (cold start). Please wait ~20–40 seconds and try again."
        )

    last_err = None
    for attempt in range(POST_RETRIES + 1):
        try:
            r = requests.post(
                f"{BILLING_API_BASE}{path}",
                json=payload,
                timeout=POST_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json() or {}
        except req_exc.Timeout as e:
            last_err = e
        except req_exc.ConnectionError as e:
            last_err = e
        except req_exc.HTTPError as e:
            # Bubble up useful error details while keeping it user-friendly.
            msg = str(e)
            try:
                msg = (r.text or msg).strip()  # type: ignore[name-defined]
            except Exception:
                pass
            raise RuntimeError(f"Billing request failed: {msg}")

        # retry on cold start-ish failures
        if attempt < POST_RETRIES:
            time.sleep(POST_RETRY_SLEEP_S)

    raise RuntimeError(
        "Billing service is taking longer than usual (cold start). Please try again in a moment."
    ) from last_err


def _portal_return_url(email: str) -> str | None:
    """Mint a session and build a portal return URL carrying the rt token so the
    Customer Portal returns the user logged in (cookie-independent)."""
    try:
        from config import APP_BASE_URL
        from ui.auth_sessions import create_session
        sid = create_session(email)
        if not sid:
            return None
        base = (APP_BASE_URL or "").rstrip("/")
        return f"{base}/?portal=return&rt={sid}"
    except Exception:
        return None


def _create_portal_url(*, email: str) -> str:
    body = {"email": email}
    ret = _portal_return_url(email)
    if ret:
        body["return_url"] = ret
    data = _billing_post_json("/create-portal-session", body)
    url = (data.get("portal_url") or "").strip()
    if not url:
        raise RuntimeError("Billing service did not return portal_url")
    return url


# =========================
# App helpers
# =========================

def _logged_in_email() -> str:
    """Lock billing to the authenticated account. In this app, username is the email."""
    return (st.session_state.get("username") or "").strip().lower()


def _current_plan_label(tier_key: str) -> str:
    t = (tier_key or "basic").strip().lower()
    if t == "admin":
        return "Admin"
    if t == "premium":
        return "Premium"
    if t == "pro":
        return "Pro"
    return "Basic"


def _refresh_tier_from_db(email: str) -> str | None:
    """Fetch latest tier for this user from DB and update session_state.
    Returns normalized tier key ('basic'/'pro'/'premium') if found.
    """
    email_key = (email or "").strip().lower()
    if not email_key:
        return None

    user = None
    try:
        # Most likely helper in your project
        from db.users import get_user_by_username  # type: ignore
        user = get_user_by_username(email_key)
    except Exception:
        try:
            from db.users import get_user  # type: ignore
            user = get_user(email_key)
        except Exception:
            user = None

    if not user:
        return None

    tier = (user.get("tier") or user.get("tier_key") or "").strip().lower()
    if not tier:
        return None

    # Keep common keys in sync
    st.session_state["tier_key"] = tier
    st.session_state["tier"] = tier
    return tier


# =========================
# Stripe open helper (reliable)
# =========================

# =========================
# UI blocks
# =========================

def _pricing_table() -> None:
    st.markdown(
        """
| Feature | Basic | Pro | Premium |
|---|---:|---:|---:|
| Curated Breakout Scans | ✅ | ✅ | ✅ |
| Breakout Score | ✅ | ✅ | ✅ |
| Charts | ✅ | ✅ | ✅ |
| Alerts (Breakout / Watchlist / Price) | 1 | 5 | 25 |
| Email Alert Delivery | ❌ | ✅ | ✅ |
| CSV Export | ❌ | ✅ | ✅ |
| Advanced Filters | ❌ | ✅ | ✅ |
| Earnings Calendar | ❌ | ✅ | ✅ |
| Scan History | ❌ | ✅ | ✅ |
| AI Insights (notes / summary / chat) | ❌ | ❌ | ✅ |
| Early Breakout (ML) | ❌ | ❌ | ✅ |
| Full Universe Mode | ❌ | ❌ | ✅ |
| Diagnostics / Retrain | ❌ | ❌ | ✅ |
"""
    )


def _benefits_block() -> None:
    st.subheader("What’s locked + why")
    st.markdown(
        """
- **Everyone** gets **1 alert** (Breakout / Watchlist / Price), checked automatically a few times a day — delivered in-app.
- **🔒 Pro features** help you *move faster*: **5 alerts with email delivery**, CSV export, advanced filters, earnings calendar, and scan history.
- **🔒 Premium features** help you *get earlier signals*: **25 alerts**, AI insights (notes / summary / chat), ML early breakout candidates, full-universe mode, and deeper diagnostics.
"""
    )


def _upgrade_buttons(current_tier_key: str) -> None:
    focus = (st.session_state.get("pricing_focus") or "").strip().lower()

    email = _logged_in_email()
    if not email or "@" not in email:
        st.warning("Please sign in before upgrading. Upgrades are tied to your account.")
        return

    col_pro, col_premium = st.columns(2)

    email = _logged_in_email()

    def _plan_button(plan: str, key: str) -> None:
        if st.button(
            f"Upgrade to {plan.title()}",
            key=key,
            width="stretch",
            type="primary" if focus == plan else "secondary",
        ):
            try:
                from ui.email_verification_gate import require_verified_for_upgrade
                allowed = require_verified_for_upgrade(email, key_suffix=key)
            except Exception:
                allowed = True
            if allowed:
                with st.spinner("Preparing checkout…"):
                    from ui.checkout import create_checkout_url
                    url, err = create_checkout_url(email, plan)
                st.session_state[f"_checkout_url_{plan}"] = url
                st.session_state[f"_checkout_err_{plan}"] = err
        url = st.session_state.get(f"_checkout_url_{plan}")
        err = st.session_state.get(f"_checkout_err_{plan}")
        if url:
            st.link_button(f"💳 Continue to Stripe ({plan.title()})", url, width="stretch")
        elif err:
            st.caption(f"⚠️ {err}")

    with col_pro:
        st.markdown("### 🚀 Pro")
        st.caption("Unlock exports + advanced filters + scan history")
        if current_tier_key in ("pro", "premium", "admin"):
            st.success("You already have Pro (or higher).")
        else:
            _plan_button("pro", "billing_upgrade_pro")

    with col_premium:
        st.markdown("### ⭐ Premium")
        st.caption("Unlock ML signals + full universe + diagnostics")
        if current_tier_key in ("premium", "admin"):
            st.success("You already have Premium (or Admin).")
        else:
            _plan_button("premium", "billing_upgrade_premium")


# =========================
# Main page
# =========================

def render_billing_page() -> None:
    st.title("💳 Plans & Billing")
    st.caption("Upgrade anytime. Downgrade anytime. No lock-in.")

    tier_key = (st.session_state.get("tier_key") or st.session_state.get("tier") or "basic").strip().lower()
    current_label = _current_plan_label(tier_key)

    st.info(f"You’re currently on **{current_label}**.")

    email = _logged_in_email()
    if not email or "@" not in email:
        st.warning("Please sign in before upgrading. Upgrades are tied to your account.")
        st.stop()

    st.info(f"Upgrading account: **{email}**")

    # Stripe redirect status (e.g., ?checkout=success or ?checkout=cancel)
    checkout_status = (st.query_params.get("checkout") or "").strip().lower()

    if checkout_status == "cancel":
        st.info("Checkout canceled — no charges were made. You can upgrade anytime.")
        try:
            st.query_params.pop("checkout", None)
        except Exception:
            pass
        st.divider()

    if checkout_status == "success":
        st.success("🎉 Upgrade complete! Syncing your plan…")

        # Avoid infinite reruns while webhook/DB catches up
        if not st.session_state.get("post_checkout_refreshed"):
            st.session_state["post_checkout_refreshed"] = True
            new_tier = _refresh_tier_from_db(email)
            if new_tier:
                try:
                    st.query_params.pop("checkout", None)
                except Exception:
                    pass
                st.session_state.pop("stripe_redirect_url", None)
                st.rerun()
            else:
                st.info("Still syncing… wait a few seconds, then click ‘Refresh plan now’.")

        if st.button("🔄 Refresh plan now", key="billing_refresh_plan", width="stretch"):
            new_tier = _refresh_tier_from_db(email)
            if new_tier:
                try:
                    st.query_params.pop("checkout", None)
                except Exception:
                    pass
                st.session_state.pop("stripe_redirect_url", None)
                st.rerun()
            else:
                st.warning("Plan not updated yet. Confirm Stripe webhook delivered successfully.")

        st.divider()

    # Customer portal for paid tiers
    if current_label in ("Pro", "Premium"):
        if st.button("Manage subscription", key="billing_manage", width="stretch"):
            try:
                st.session_state["_portal_url"] = _create_portal_url(email=email)
            except Exception as e:
                st.error(str(e))
                st.caption("Tip: The billing service may be waking up. Try again in ~30 seconds.")

        portal_url = st.session_state.get("_portal_url")
        if portal_url:
            st.success("Customer Portal ready!")
            st.link_button("🔧 Open Stripe Customer Portal", portal_url, width="stretch")
            st.caption("Opens in a new tab. After managing your plan, return here — your changes apply automatically.")

    _pricing_table()
    _benefits_block()

    st.divider()
    _upgrade_buttons(tier_key)

    st.divider()
    if st.button("← Back to Scanner", key="billing_back", width="stretch"):
        st.session_state.pop("show_pricing", None)
        st.session_state.pop("pricing_focus", None)
        st.session_state.pop("stripe_redirect_url", None)
        st.session_state.pop("stripe_redirect_kind", None)
        st.session_state.pop("post_checkout_refreshed", None)
        try:
            st.switch_page("app.py")
        except Exception:
            # If switch_page isn't available or path differs, just nudge
            st.info("Use the sidebar to return to the Scanner page.")


# Run page
render_billing_page()