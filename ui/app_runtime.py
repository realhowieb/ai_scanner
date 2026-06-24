"""Runtime helpers for the Streamlit app shell."""
from __future__ import annotations

import json
from datetime import datetime, time
from json import JSONDecodeError
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def get_market_session(now: datetime | None = None) -> str:
    """Return premarket, regular, afterhours, or closed for US/Eastern time."""
    tz = ZoneInfo("US/Eastern")
    if now is None:
        now = datetime.now(tz)
    else:
        now = now.astimezone(tz)

    if now.weekday() >= 5:
        return "closed"

    current_time = now.time()
    if time(4, 0) <= current_time < time(9, 30):
        return "premarket"
    if time(9, 30) <= current_time < time(16, 0):
        return "regular"
    if time(16, 0) <= current_time < time(20, 0):
        return "afterhours"
    return "closed"


def normalize_results_to_df(obj: object) -> pd.DataFrame | None:
    """Normalize load_run_results output to a DataFrame or None."""
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        return None if obj.empty else obj

    if isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
        except (TypeError, ValueError):
            return None
        return None if df.empty else df

    if isinstance(obj, dict):
        try:
            df = pd.DataFrame([obj])
        except (TypeError, ValueError):
            return None
        return None if df.empty else df

    if isinstance(obj, str):
        raw = obj.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (JSONDecodeError, TypeError):
            return None
        return normalize_results_to_df(parsed)

    return None


def render_active_filters_summary(
    *,
    universe,
    min_price: float,
    max_price: float,
    min_dollar_vol: float,
    top_n: int,
    premarket: bool,
    afterhours: bool,
    include_ta: bool,
    unusual_vol: bool,
    apply_gap_filter: bool,
    min_gap: float,
    max_nasdaq_scan: int,
    max_combo_scan: int,
) -> None:
    """Render a compact summary of the active scan filters."""
    chips: list[str] = []

    if universe:
        chips.append(f"Universe: {universe}")

    chips.append(f"Price: ${min_price:g}-${max_price:g}")

    if min_dollar_vol and min_dollar_vol > 0:
        chips.append(f"Min $Vol: {int(min_dollar_vol):,}")

    chips.append(f"Top N: {int(top_n)}")
    chips.append(f"NASDAQ cap: {int(max_nasdaq_scan):,}")
    chips.append(f"Combo cap: {int(max_combo_scan):,}")

    if premarket:
        chips.append("Session: Premarket")
    elif afterhours:
        chips.append("Session: After-hours")
    else:
        chips.append("Session: Regular")

    if include_ta:
        chips.append("TA: ON")
    if unusual_vol:
        chips.append("Unusual Vol: ON")

    if apply_gap_filter:
        chips.append(f"Gap Filter: ON (>= {float(min_gap):g}%)")

    st.markdown("#### Active Filters")
    st.caption(" | ".join(chips))


def render_onboarding_hint(username: str, *, tier_name: str) -> None:
    """Render a one-time quick-start hint per session per user."""
    key = f"onboarding_dismissed::{(username or '').strip().lower()}"
    if st.session_state.get(key):
        return

    with st.expander("Quick start", expanded=True):
        st.markdown(
            f"""
**Welcome!** You're signed in on **{tier_name}**.

**Fast workflow:**
1. Set filters in the sidebar
2. Click **Run Scan** (SP500 / NASDAQ / Combo)
3. Use **Save as my default settings** once you like your setup
4. Use **Reset to saved profile** anytime to revert

Tip: turn on **Apply Gap Filter** to enforce **Min Gap %**.
"""
        )
        if st.button("Got it", key=f"onboarding_got_it::{username}"):
            st.session_state[key] = True
            st.rerun()


def render_sidebar_upgrade_card(
    tier_obj: object | None,
    *,
    has_min_tier: Callable[[object | None, str], bool],
) -> None:
    """Show the upgrade CTA card for Basic users in the sidebar."""
    try:
        # Admins never see the upgrade card regardless of tier_obj key.
        import streamlit as _st
        if _st.session_state.get("is_admin"):
            return
        if has_min_tier(tier_obj, "pro"):
            return
    except (AttributeError, KeyError, TypeError, ValueError):
        return

    with st.sidebar.container(border=True):
        st.markdown("### You're on Basic")
        st.caption(
            "You're seeing a limited scan.\n"
            "Upgrade to unlock advanced filters, exports, and AI signals."
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Upgrade to Pro", key="upgrade_to_pro", width="stretch"):
                st.session_state["pricing_focus"] = "pro"
                st.switch_page("pages/billing.py")
        with c2:
            if st.button("Upgrade to Premium", key="upgrade_to_premium", width="stretch"):
                st.session_state["pricing_focus"] = "premium"
                st.switch_page("pages/billing.py")

        st.caption(
            "Pro unlocks exports and advanced filters. Premium unlocks full-universe and Early Breakout."
        )
