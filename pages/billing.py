import os
import streamlit as st
import requests

# Stripe billing backend (Render)

BILLING_API_BASE = os.getenv("BILLING_API_BASE", "https://ai-scanner-h2c8.onrender.com").strip()


def _refresh_tier_from_db(email: str) -> str | None:
    """Fetch latest tier for this user from DB and update session_state.

    Returns the normalized tier key (e.g., 'basic'/'pro'/'premium') if found.
    """
    email_key = (email or "").strip().lower()
    if not email_key:
        return None

    user = None
    try:
        # Primary lookup used throughout this project
        from db.users import get_user_by_username  # type: ignore

        user = get_user_by_username(email_key)
    except Exception:
        try:
            # Fallback name (older versions)
            from db.users import get_user  # type: ignore

            user = get_user(email_key)
        except Exception:
            user = None

    if not user:
        return None

    tier = (user.get("tier") or user.get("tier_key") or "").strip().lower()
    if not tier:
        return None

    # Keep both keys in sync (some parts of the app read either)
    st.session_state["tier_key"] = tier
    st.session_state["tier"] = tier

    # Also refresh entitlements if app.py already computed them this session
    # (If not present yet, app.py will recompute on next navigation.)
    ent = st.session_state.get("entitlements")
    if isinstance(ent, dict):
        # no-op: leave to app.py recompute; we just ensure tier reflects latest
        pass

    return tier


def _logged_in_email() -> str:
    # Lock billing to the authenticated account. Do NOT fall back to any other session keys.
    # In this app, username is the email.
    return (st.session_state.get("username") or "").strip().lower()


def _create_checkout_url(*, email: str, plan: str) -> str:
    r = requests.post(
        f"{BILLING_API_BASE}/create-checkout-session",
        json={"email": email, "plan": plan},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json() or {}
    url = (data.get("checkout_url") or "").strip()
    if not url:
        raise RuntimeError("Billing service did not return checkout_url")
    return url


def _create_portal_url(*, email: str) -> str:
    r = requests.post(
        f"{BILLING_API_BASE}/create-portal-session",
        json={"email": email},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json() or {}
    url = (data.get("portal_url") or "").strip()
    if not url:
        raise RuntimeError("Billing service did not return portal_url")
    return url


def _current_plan_label(tier_key: str) -> str:
    t = (tier_key or "basic").strip().lower()
    if t == "admin":
        return "Admin"
    if t == "premium":
        return "Premium"
    if t == "pro":
        return "Pro"
    return "Basic"


def _pricing_table() -> None:
    st.markdown(
        """
| Feature | Basic | Pro | Premium |
|---|---:|---:|---:|
| Curated Breakout Scans | ✅ | ✅ | ✅ |
| Breakout Score | ✅ | ✅ | ✅ |
| Charts | ✅ | ✅ | ✅ |
| CSV Export | ❌ | ✅ | ✅ |
| Advanced Filters | ❌ | ✅ | ✅ |
| Scan History | ❌ | ✅ | ✅ |
| Early Breakout (ML) | ❌ | ❌ | ✅ |
| Full Universe Mode | ❌ | ❌ | ✅ |
| Diagnostics / Retrain | ❌ | ❌ | ✅ |
"""
    )


def _benefits_block() -> None:
    st.subheader("What’s locked + why")
    st.markdown(
        """
- **🔒 Pro features** help you *move faster*: export results, use advanced filters, and review scan history.
- **🔒 Premium features** help you *get earlier signals*: ML early breakout candidates, full-universe mode, and deeper diagnostics.
"""
    )


def _upgrade_buttons(current_tier: str) -> None:
    focus = (st.session_state.get("pricing_focus") or "").strip().lower()

    email = _logged_in_email()
    if not email or "@" not in email:
        st.warning("Please sign in (create an account) before upgrading. Upgrades are tied to your account.")
        return

    # If we already created a checkout link in this session, show it.
    checkout_url = (st.session_state.get("checkout_url") or "").strip()
    if checkout_url:
        st.success("Checkout created ✅")
        st.link_button("➡️ Continue to Stripe Checkout", checkout_url, use_container_width=True)
        st.caption("After payment, return to the app and refresh (or log out/in) to see your updated plan.")
        if st.button("Clear checkout link", key="billing_clear_checkout", use_container_width=True):
            st.session_state.pop("checkout_url", None)
            st.rerun()
        st.divider()

    col_pro, col_premium = st.columns(2)

    with col_pro:
        st.markdown("### 🚀 Pro")
        st.caption("Unlock exports + advanced filters + scan history")
        if current_tier in ("pro", "premium", "admin"):
            st.success("You already have Pro (or higher).")
        else:
            if st.button(
                "Upgrade to Pro",
                key="billing_upgrade_pro",
                use_container_width=True,
                type="primary" if focus == "pro" else "secondary",
            ):
                try:
                    st.session_state["pricing_focus"] = "pro"
                    st.session_state["checkout_url"] = _create_checkout_url(email=email, plan="pro")
                    st.toast("Checkout created — continue to Stripe.", icon="💳")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not start checkout: {e}")

    with col_premium:
        st.markdown("### ⭐ Premium")
        st.caption("Unlock ML signals + full universe + diagnostics")
        if current_tier in ("premium", "admin"):
            st.success("You already have Premium (or Admin).")
        else:
            if st.button(
                "Upgrade to Premium",
                key="billing_upgrade_premium",
                use_container_width=True,
                type="primary" if focus == "premium" else "secondary",
            ):
                try:
                    st.session_state["pricing_focus"] = "premium"
                    st.session_state["checkout_url"] = _create_checkout_url(email=email, plan="premium")
                    st.toast("Checkout created — continue to Stripe.", icon="💳")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not start checkout: {e}")


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

    # Auto-refresh plan after Stripe redirect (e.g., ?checkout=success)
    checkout_status = (st.query_params.get("checkout") or "").strip().lower()
    if checkout_status == "success":
        st.success("Payment received ✅ Syncing your plan…")

        # Avoid infinite reruns if DB/webhook is still catching up
        if not st.session_state.get("post_checkout_refreshed"):
            st.session_state["post_checkout_refreshed"] = True
            new_tier = _refresh_tier_from_db(email)
            if new_tier:
                # Clear the query param so we don't re-trigger on every reload
                try:
                    st.query_params.pop("checkout", None)
                except Exception:
                    pass
                st.rerun()
            else:
                st.info("Still syncing… If this persists, refresh in a few seconds or log out/in.")

        # Give users a manual retry button as well
        if st.button("🔄 Refresh plan now", key="billing_refresh_plan", use_container_width=True):
            new_tier = _refresh_tier_from_db(email)
            if new_tier:
                try:
                    st.query_params.pop("checkout", None)
                except Exception:
                    pass
                st.rerun()
            else:
                st.warning("Plan not updated yet. Make sure the Stripe webhook delivered successfully.")

        st.divider()

    if current_label in ("Pro", "Premium"):
        if st.button("Manage subscription", key="billing_manage", use_container_width=True):
            try:
                portal_url = _create_portal_url(email=email)
                st.link_button("Open Stripe Customer Portal", portal_url, use_container_width=True)
            except Exception as e:
                st.error(f"Could not open billing portal: {e}")

    _pricing_table()
    _benefits_block()

    st.divider()
    _upgrade_buttons(tier_key)

    st.divider()
    if st.button("← Back to Scanner", key="billing_back", use_container_width=True):
        # Clear pricing open flags so the sidebar doesn't keep showing the cue
        st.session_state.pop("show_pricing", None)
        st.session_state.pop("pricing_focus", None)
        st.session_state.pop("checkout_url", None)
        st.session_state.pop("post_checkout_refreshed", None)
        st.switch_page("app.py")


# Streamlit runs scripts top-to-bottom; render immediately.
render_billing_page()
