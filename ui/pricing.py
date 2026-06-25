"""Pricing sidebar UI module.

Provides:
    - pricing_sidebar(current_username, users)
"""

from typing import Dict, Optional

import streamlit as st

from config import STRIPE_MONTHLY_LINKS, STRIPE_YEARLY_LINKS, TIERS_CONFIG

try:
    from config import BILLING_API_BASE
except ImportError:
    BILLING_API_BASE = None


def _create_checkout_url(email: str, plan: str, period: str) -> str | None:
    """Call billing service to create a Stripe checkout session. Returns URL or None."""
    base = (BILLING_API_BASE or "").rstrip("/")
    if not base or not email:
        return None
    try:
        import json
        import urllib.request
        payload = json.dumps({"email": email, "plan": plan, "period": period}).encode()
        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("checkout_url") or data.get("url") or data.get("portal_url")
    except Exception:
        return None


def pricing_sidebar(current_username: Optional[str], users: Dict[str, Dict[str, str]], email: str | None = None):
    """Show upgrade options only for tiers above the user's current plan.

    - Basic users see Pro + Premium
    - Pro users see Premium
    - Premium users see a small thank-you message (no upgrade cards)
    - Includes a Monthly / Yearly toggle that switches Stripe links.
    """
    tiers_order = ["basic", "pro", "premium"]
    current_key = users.get(current_username or "", {}).get("tier", "basic")
    # Normalize tier key: handle case/whitespace and unknown values gracefully.
    current_key = (current_key or "basic").strip().lower()
    try:
        start_idx = tiers_order.index(current_key) + 1
    except ValueError:
        start_idx = 1  # default: treat as basic if unknown

    upsell_keys = tiers_order[start_idx:]

    st.sidebar.markdown("## 💳 Plan & Upgrade")
    current_label = current_key.capitalize()

    # If no higher tiers available (Premium), show a simple confirmation message.
    if not upsell_keys:
        st.sidebar.success(
            "⭐ You're on the top **Premium** plan. All features are unlocked, "
            "including EZ 3-Step AI Scanner, Early Breakout Candidates, Scan History, "
            "advanced filters, and more."
        )
        return

    # Otherwise, show the current plan and upsell cards for higher tiers.
    st.sidebar.caption(f"Current plan: **{current_label}**")

    # Billing period toggle
    billing_period = st.sidebar.radio(
        "Billing period",
        ["Monthly", "Yearly"],
        index=0,
        horizontal=True,
    )

    cols = st.sidebar.columns(len(upsell_keys))
    for i, key in enumerate(upsell_keys):
        cfg = TIERS_CONFIG.get(key, {})
        name = cfg.get("name", key.title())
        monthly_price = cfg.get("price_monthly", 0)
        yearly_price = cfg.get("price_yearly", 0)
        features = cfg.get("features", [])

        with cols[i]:
            # Center-align everything in this card
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

            st.markdown(f"**{name}**")

            # Best value badge for Pro on Yearly
            if key == "pro" and billing_period == "Yearly":
                st.markdown(
                    "<div style='font-size: 0.8rem; color: #22c55e; font-weight: 600; margin-bottom: 0.2rem;'>"
                    "⭐ Best value"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Show only the selected billing period price
            if billing_period == "Monthly":
                st.markdown(
                    f"<div style='font-size: 1.1rem; font-weight: 700;'>${monthly_price}/mo</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size: 1.1rem; font-weight: 700; color: #22c55e;'>${yearly_price}/yr</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size: 0.8rem; color: #9ca3af;'>≈ 2 months free vs monthly.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"- SP500: {'✅' if ('SP500 Scan' in features or 'SP500' in features) else '❌'}")
            st.markdown(f"- NASDAQ: {'✅' if 'NASDAQ' in features else '❌'}")
            st.markdown(f"- Export: {'✅' if 'ExportCSV' in features else '❌'}")
            if "AI Notes" in features:
                st.markdown("- AI Notes: ✅")
            else:
                st.markdown("- AI Notes: ❌")

            btn_label = f"Subscribe {name} ({billing_period})"
            btn_key = f"upgrade_{key}_{billing_period}"

            # Use dynamic checkout session (same-tab redirect) when billing service is available.
            if BILLING_API_BASE and email:
                if st.button(btn_label, key=btn_key):
                    with st.spinner("Preparing checkout…"):
                        url = _create_checkout_url(email, key, billing_period.lower())
                    if url:
                        st.markdown(
                            f'<meta http-equiv="refresh" content="0; url={url}">',
                            unsafe_allow_html=True,
                        )
                        st.info("Redirecting to Stripe checkout…")
                        st.stop()
                    else:
                        st.error("Could not create checkout session. Try again or contact support.")
            else:
                # Fallback: static payment links (open in same tab via link_button)
                fallback_url = (
                    STRIPE_MONTHLY_LINKS if billing_period == "Monthly" else STRIPE_YEARLY_LINKS
                ).get(key)
                if fallback_url:
                    st.link_button(btn_label, fallback_url)
                else:
                    st.caption("Stripe link not configured yet for this plan/period.")