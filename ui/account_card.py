"""Shared account / plan card — the one sidebar account block on every page.

Scanner (app.py), every sub-page sidebar (ui.nav) and the phone "☰ Menu" all
render this component, so the name, plan, plan summary, upgrade CTA,
"Compare all plans" and "Log out" can never drift between pages.

It is presentation only: it reads the account context that app.py resolves once
after sign-in (username, display_name, tier_key, is_admin — B4) and never looks
anything up. Plan names come from ui.plan_labels; plan facts from the canonical
entitlement numbers (alert limits) and the Run 73/85 packaging.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _alerts(tier: str) -> int:
    from ui.app_session import ALERT_LIMIT_BY_TIER

    return int(ALERT_LIMIT_BY_TIER.get(tier, 1))


def _premium_value() -> str:
    return (f"{_alerts('premium')} alerts, AI scan summaries and results chat, setup notes, "
            "Early Breakout research, full-market custom scans and paper trading")


def plan_card(tier: Any, *, is_admin: bool = False) -> Dict[str, Optional[str]]:
    """Copy for the card. Keys: label, headline, summary, cta_label, cta_plan, cta_note, compare."""
    from ui.plan_labels import plan_label

    label = plan_label(tier, is_admin=is_admin)
    if label == "Admin":
        return {"label": label, "headline": "Admin access",
                "summary": "All features are enabled for testing and operations.",
                "cta_label": None, "cta_plan": None, "cta_note": None, "compare": None}
    if label == "Premium":
        return {"label": label, "headline": "You're on Premium",
                "summary": f"Research and workflow: {_premium_value()}.",
                "cta_label": None, "cta_plan": None, "cta_note": None, "compare": "Compare all plans"}
    if label == "Pro":
        return {"label": label, "headline": "You're on Pro",
                "summary": (f"Monitor and investigate: {_alerts('pro')} alerts with email delivery, interactive "
                            "results and CSV export, Nasdaq and combined scans, earnings and scan history."),
                "cta_label": "Upgrade to Premium", "cta_plan": "premium",
                "cta_note": f"Premium adds research and workflow: {_premium_value()}.",
                "compare": "Compare all plans"}
    return {"label": "Free", "headline": "You're on Free",
            "summary": "Discover today's market opportunities with HSF Score and basic Stock Intelligence.",
            "cta_label": "Upgrade to Pro", "cta_plan": "pro",
            "cta_note": (f"Pro adds monitoring and investigation: {_alerts('pro')} alerts, email delivery, "
                         "interactive results, exports and history."),
            "compare": "Compare all plans"}


def account_name(display_name: Any, username: Any) -> str:
    raw_display = str(display_name or "").strip()
    raw_username = str(username or "").strip()
    if raw_display:
        return raw_display.split("@")[0] if "@" in raw_display else raw_display
    if "@" in raw_username:
        return raw_username.split("@")[0]
    return raw_username or "Account"


def render_account_card(*, key_suffix: str = "sidebar", compact: bool = False) -> None:
    """Render the card in the current container (sidebar or phone menu). Never raises.
    `compact` (phone menu) drops the longer descriptions but keeps plan, CTA,
    "Compare all plans" and "Log out"."""
    if st is None:
        return
    try:
        ss = st.session_state
        username = str(ss.get("username") or "").strip()
        if not username:
            return
        is_admin = bool(ss.get("is_admin"))
        card = plan_card(ss.get("tier_key") or ss.get("tier"), is_admin=is_admin)
        name = account_name(ss.get("display_name"), username)
        st.markdown(f"**👤 {name}**" if compact else f"### 👤 {name}")
        st.markdown(f"**Plan:** `{card['label']}`")
        with st.container(border=True, key=f"hsf_account_card_{key_suffix}"):
            st.markdown(f"**{card['headline']}**")
            if not compact:
                st.caption(card["summary"])
            if card["cta_label"]:
                from ui.app_runtime import _upgrade_button

                _upgrade_button(card["cta_label"], card["cta_plan"], f"upgrade_to_{card['cta_plan']}_{key_suffix}")
                if card["cta_note"] and not compact:
                    st.caption(card["cta_note"])
            if card["compare"]:
                st.page_link("pages/billing.py", label=card["compare"], icon="💳")
        if st.button("Log out", key=f"nav_logout_{key_suffix}"):
            try:
                from ui.auth import logout_and_reset_session

                logout_and_reset_session()
            except Exception:
                pass
        st.divider()
    except Exception:
        pass
