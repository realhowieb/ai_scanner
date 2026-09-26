"""Runtime helpers for the Streamlit app shell."""
from __future__ import annotations

import json
from datetime import datetime, time
from json import JSONDecodeError
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def _upgrade_button(label: str, plan: str, key: str) -> None:
    """Create a checkout session inline and show the Stripe link."""
    if st.button(label, key=key, width="stretch"):
        email = st.session_state.get("username", "")
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
        st.link_button(f"💳 Continue to Stripe ({plan.title()})", url)
    elif err:
        st.caption(f"⚠️ {err}")


def get_market_session(now: datetime | None = None) -> str:
    """Return premarket, regular, afterhours, or closed for US/Eastern time."""
    tz = ZoneInfo("US/Eastern")
    if now is None:
        now = datetime.now(tz)
    else:
        now = now.astimezone(tz)

    if now.weekday() >= 5:
        return "closed"

    # Run 70 (P2-9): use the NYSE calendar (read-only) so holidays read "closed"
    # and early-close days end the regular session early — matching the trust
    # banner. Outside the calendar's years, fall back to the fixed hours.
    close = time(16, 0)
    try:
        from analytics import market_calendar as mc

        if mc.calendar_covered(now.date()):
            if not mc.is_trading_day(now.date()):
                return "closed"
            close = mc.close_time_et(now.date())
    except Exception:
        pass

    current_time = now.time()
    if time(4, 0) <= current_time < time(9, 30):
        return "premarket"
    if time(9, 30) <= current_time < close:
        return "regular"
    if close <= current_time < time(20, 0):
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
    """Render a one-time quick-start hint, dismissable permanently per user."""
    key = f"onboarding_dismissed::{(username or '').strip().lower()}"
    if st.session_state.get(key):
        return

    # Persisted dismissal: once a user clicks "Got it" it stays gone across
    # sessions. Lazy + guarded so a stale module / DB hiccup falls back to the
    # session-only behavior instead of crashing.
    try:
        from db.user_settings import get_onboarding_dismissed

        if get_onboarding_dismissed(username):
            st.session_state[key] = True
            return
    except Exception:
        pass

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
            try:
                from db.user_settings import set_onboarding_dismissed

                set_onboarding_dismissed(username, True)
            except Exception:
                pass
            st.rerun()


def render_sidebar_upgrade_card(
    tier_obj: object | None,
    *,
    has_min_tier: Callable[[object | None, str], bool],
) -> None:
    """Show the primary Pro upgrade path for Free users in the sidebar."""
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
        st.markdown("### You're on Free")
        st.caption(
            "Discover today's market opportunities with HSF Score and basic Stock Intelligence."
        )
        _upgrade_button("Upgrade to Pro", "pro", "upgrade_to_pro")
        st.caption(
            "Pro adds monitoring and investigation: 5 alerts, email delivery, interactive results, exports and history."
        )
        st.page_link("pages/billing.py", label="Compare all plans")
