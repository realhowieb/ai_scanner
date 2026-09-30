"""Returning from the Stripe customer portal (P1-27 finding, 2026-09-29).

The portal can upgrade, downgrade, switch or cancel, so a portal return must not
wait for an *upgrade* the way a checkout return does (that showed "Activating
your plan upgrade…" for ~20 s after a downgrade to Free, then "Upgrade is still
processing"). Read the plan once, show it, and clear the return flags.
"""
from __future__ import annotations

from typing import Callable, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def handle_portal_return(username: str, resolve_tier: Callable[[str], Optional[str]]) -> None:
    """Refresh the plan from the database and say what it is now. Never raises."""
    if st is None:
        return
    try:
        tier = resolve_tier(username)
    except Exception:
        tier = None
    if tier:
        st.session_state["tier"] = tier
        st.session_state["plan"] = tier
    for key in ("portal", "rt"):
        try:
            st.query_params.pop(key, None)
        except Exception:
            pass
    st.session_state.pop("_tier_poll_attempt", None)
    try:
        from ui.plan_labels import plan_label

        label = plan_label(tier or "basic")
    except Exception:
        label = str(tier or "basic").title()
    st.info(
        f"Billing updated. Your plan: **{label}**. "
        "Changes can take a few seconds to show; refresh if it hasn't updated yet."
    )
