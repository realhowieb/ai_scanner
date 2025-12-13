import streamlit as st


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

    col_pro, col_premium = st.columns(2)

    with col_pro:
        st.markdown("### 🚀 Pro")
        st.caption("Unlock exports + advanced filters + scan history")
        if current_tier in ("pro", "premium", "admin"):
            st.success("You already have Pro (or higher).")
        else:
            if st.button("Upgrade to Pro", key="billing_upgrade_pro", use_container_width=True, type="primary" if focus == "pro" else "secondary"):
                # Day 6: UI only. Day 7+: wire to Stripe or upgrade codes.
                st.session_state["upgrade_to"] = "pro"
                st.toast("Pro upgrade flow coming next (billing wiring).", icon="💳")

    with col_premium:
        st.markdown("### ⭐ Premium")
        st.caption("Unlock ML signals + full universe + diagnostics")
        if current_tier in ("premium", "admin"):
            st.success("You already have Premium (or Admin).")
        else:
            if st.button("Upgrade to Premium", key="billing_upgrade_premium", use_container_width=True, type="primary" if focus == "premium" else "secondary"):
                # Day 6: UI only. Day 7+: wire to Stripe or upgrade codes.
                st.session_state["upgrade_to"] = "premium"
                st.toast("Premium upgrade flow coming next (billing wiring).", icon="💳")


def render_billing_page() -> None:
    st.title("💳 Plans & Billing")
    st.caption("Upgrade anytime. Downgrade anytime. No lock-in.")

    tier_key = (st.session_state.get("tier_key") or st.session_state.get("tier") or "basic").strip().lower()
    current_label = _current_plan_label(tier_key)

    st.info(f"You’re currently on **{current_label}**.")

    _pricing_table()
    _benefits_block()

    st.divider()
    _upgrade_buttons(tier_key)

    st.divider()
    if st.button("← Back to Scanner", key="billing_back", use_container_width=True):
        # Clear pricing open flags so the sidebar doesn't keep showing the cue
        st.session_state.pop("show_pricing", None)
        st.session_state.pop("pricing_focus", None)
        st.switch_page("app.py")


# If this file is used as a Streamlit multipage app page, render immediately.
if __name__ == "__main__" or st._is_running_with_streamlit:
    render_billing_page()
