from __future__ import annotations

import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components
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


def _create_checkout_or_portal(*, email: str, plan: str) -> tuple[str, str]:
    """Create a Stripe redirect for this plan.

    Backend may return either:
      - {checkout_url: ..., mode: "checkout"}
      - {portal_url: ..., mode: "portal"}  (existing subscriber should manage upgrade in portal)
    Returns (mode, url).
    """
    data = _billing_post_json("/create-checkout-session", {"email": email, "plan": plan})

    portal_url = (data.get("portal_url") or "").strip()
    checkout_url = (data.get("checkout_url") or "").strip()
    mode = (data.get("mode") or "").strip().lower()

    if portal_url:
        return ("portal", portal_url)
    if checkout_url:
        return ("checkout", checkout_url)

    # Fallback if mode is present but URL key differs
    if mode == "portal" and portal_url:
        return ("portal", portal_url)
    if mode == "checkout" and checkout_url:
        return ("checkout", checkout_url)

    raise RuntimeError("Billing service did not return a checkout_url or portal_url")


def _create_portal_url(*, email: str) -> str:
    data = _billing_post_json("/create-portal-session", {"email": email})
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

def _open_checkout_same_tab(url: str, *, kind: str = "Checkout") -> None:
    """Reliable Streamlit-friendly checkout open.

    Streamlit Cloud can intercept plain <a target=_self> links, sometimes resulting in a blank
    Streamlit loading screen. The most reliable approach is a USER CLICK inside a components
    iframe that redirects the TOP window.
    """
    u = (url or "").strip()
    if not u:
        return

    st.success(f"Stripe {kind} is ready ✅")
    st.caption(f"Click the button below to open Stripe {kind}. If it doesn’t open, use the fallback link.")

    # Primary: user-click button that redirects the TOP window.
    components.html(
        f"""
        <div style="width:100%;">
          <button id="stripeGo"
            style="width:100%; padding:0.65rem 1rem; border-radius:0.6rem; border:1px solid rgba(255,255,255,0.18);
                   background: rgba(255,255,255,0.06); color: inherit; font-size: 1rem; cursor: pointer;">
            💳 Open Stripe (same tab)
          </button>
        </div>
        <script>
          (function() {{
            var url = {u!r};
            var btn = document.getElementById('stripeGo');
            if (!btn) return;
            btn.addEventListener('click', function() {{
              try {{
                window.top.location.href = url;
              }} catch (e) {{
                try {{ window.parent.location.href = url; }} catch (e2) {{ window.location.href = url; }}
              }}
            }});
          }})();
        </script>
        """,
        height=72,
    )

    # Fallback if the browser still forces a new tab
    st.link_button(f"Open Stripe {kind} (fallback)", u, width="stretch")
    st.stop()


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


def _upgrade_buttons(current_tier_key: str) -> None:
    focus = (st.session_state.get("pricing_focus") or "").strip().lower()

    email = _logged_in_email()
    if not email or "@" not in email:
        st.warning("Please sign in before upgrading. Upgrades are tied to your account.")
        return

    # Back-compat cleanup: older sessions may have used checkout_url
    if st.session_state.get("checkout_url"):
        st.session_state["stripe_redirect_url"] = st.session_state.pop("checkout_url")
        st.session_state.setdefault("stripe_redirect_kind", "Checkout")

    # If we already created a Stripe redirect URL in this session, show same-tab open UI
    stripe_url = (st.session_state.get("stripe_redirect_url") or "").strip()
    stripe_kind = (st.session_state.get("stripe_redirect_kind") or "Checkout").strip() or "Checkout"
    if stripe_url:
        _open_checkout_same_tab(stripe_url, kind=stripe_kind)

    col_pro, col_premium = st.columns(2)

    with col_pro:
        st.markdown("### 🚀 Pro")
        st.caption("Unlock exports + advanced filters + scan history")

        if current_tier_key in ("pro", "premium", "admin"):
            st.success("You already have Pro (or higher).")
        else:
            if st.button(
                "Upgrade to Pro",
                key="billing_upgrade_pro",
                width="stretch",
                type="primary" if focus == "pro" else "secondary",
            ):
                try:
                    st.session_state["pricing_focus"] = "pro"
                    mode, url = _create_checkout_or_portal(email=email, plan="pro")
                    st.session_state["stripe_redirect_kind"] = "Customer Portal" if mode == "portal" else "Checkout"
                    st.session_state["stripe_redirect_url"] = url
                    _open_checkout_same_tab(url, kind=st.session_state["stripe_redirect_kind"])
                except Exception as e:
                    st.error(str(e))
                    st.caption("Tip: On free Render, billing may need ~30 seconds to wake up.")

    with col_premium:
        st.markdown("### ⭐ Premium")
        st.caption("Unlock ML signals + full universe + diagnostics")

        if current_tier_key in ("premium", "admin"):
            st.success("You already have Premium (or Admin).")
        else:
            if st.button(
                "Upgrade to Premium",
                key="billing_upgrade_premium",
                width="stretch",
                type="primary" if focus == "premium" else "secondary",
            ):
                try:
                    st.session_state["pricing_focus"] = "premium"
                    mode, url = _create_checkout_or_portal(email=email, plan="premium")
                    st.session_state["stripe_redirect_kind"] = "Customer Portal" if mode == "portal" else "Checkout"
                    st.session_state["stripe_redirect_url"] = url
                    _open_checkout_same_tab(url, kind=st.session_state["stripe_redirect_kind"])
                except Exception as e:
                    st.error(str(e))
                    st.caption("Tip: On free Render, billing may need ~30 seconds to wake up.")


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
                portal_url = _create_portal_url(email=email)
                # Prefer same-tab open
                components.html(
                    f"""
                    <div style=\"width:100%;\">
                      <button id=\"portalGo\"
                        style=\"width:100%; padding:0.65rem 1rem; border-radius:0.6rem; border:1px solid rgba(255,255,255,0.18);
                               background: rgba(255,255,255,0.06); color: inherit; font-size: 1rem; cursor: pointer;\">
                        🔧 Open Stripe Customer Portal (same tab)
                      </button>
                    </div>
                    <script>
                      (function() {{
                        var url = {portal_url!r};
                        var btn = document.getElementById('portalGo');
                        if (!btn) return;
                        btn.addEventListener('click', function() {{
                          try {{
                            window.top.location.href = url;
                          }} catch (e) {{
                            try {{ window.parent.location.href = url; }} catch (e2) {{ window.location.href = url; }}
                          }}
                        }});
                      }})();
                    </script>
                    """,
                    height=72,
                )
                st.link_button("Open Stripe Customer Portal (fallback)", portal_url, width="stretch")
            except Exception as e:
                st.error(str(e))
                st.caption("Tip: The billing service may be waking up. Try again in ~30 seconds.")

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