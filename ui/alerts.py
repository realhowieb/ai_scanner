"""In-app UI for creating and managing per-user alerts.

Three launch alert types: Breakout, Watchlist, and Price Above/Below. Renders a
create form per type, the user's existing alerts (toggle/delete), and a
'Recently triggered' feed populated by the scheduled alert runner.

Gracefully degrades to a caption when ALERTS_ENABLED is off or Neon is
unavailable — it never raises into the main app.
"""
from __future__ import annotations

from typing import List

import streamlit as st


def _fmt_alert(a: dict) -> str:
    """One-line human description of an alert row."""
    t = a.get("alert_type")
    if t == "breakout":
        scope = "watchlist only" if a.get("watchlist_only") else "all tickers"
        return f"🚀 Breakout ≥ {float(a.get('threshold') or 0):g} ({scope})"
    if t == "watchlist":
        return "📋 Watchlist — any holding appears in scan results"
    if t == "price":
        return (
            f"💲 {a.get('ticker')} price {a.get('direction') or 'above'} "
            f"{float(a.get('threshold') or 0):g}"
        )
    return str(t)


def render_alerts_panel(
    user_id: str,
    watch_tickers: List[str] | None = None,
    *,
    max_alerts: int = 1,
    email_enabled: bool = False,
) -> None:
    """Render the alerts management block in the main column.

    max_alerts: per-tier cap on how many alerts the user may create.
    email_enabled: whether this user's tier (Pro+) gets email delivery; Basic
    still gets in-app alerts but not email.
    """
    try:
        from config import ALERTS_ENABLED
    except Exception:
        return
    if not ALERTS_ENABLED:
        return
    if not user_id:
        return

    try:
        from db.alerts import (
            create_alert,
            delete_alert,
            list_alerts,
            list_recent_events,
            set_alert_enabled,
        )
    except Exception:
        return

    st.markdown("## 🔔 Alerts")
    if email_enabled:
        st.caption(
            "Get notified (in-app + email) when your conditions hit. "
            "Checked automatically a few times a day."
        )
    else:
        st.caption(
            "Get notified in-app when your conditions hit, checked automatically "
            "a few times a day. 📧 **Email alerts are a Pro feature** — upgrade to "
            "get them in your inbox."
        )

    try:
        existing = list_alerts(user_id)
    except Exception as e:
        st.caption("Alerts require Neon DB (cloud) and are currently unavailable.")
        with st.expander("Alert error details", expanded=False):
            st.code(f"{type(e).__name__}: {e}\n{repr(e)}")
        return

    used = len(existing)
    if used >= int(max_alerts):
        st.warning(
            f"You're using all **{used}/{max_alerts}** alerts on your plan. "
            "Upgrade for more alert slots."
        )
    else:
        st.caption(f"Using {used} of {max_alerts} alert slots on your plan.")

    with st.expander("➕ Create an alert", expanded=True):
        tab_break, tab_watch, tab_price = st.tabs(
            ["🚀 Breakout", "📋 Watchlist", "💲 Price"]
        )

        # NOTE: plain widgets (not st.form) — st.form rendered empty inside the
        # expander → tabs nesting on the deployed Streamlit, so the inputs and
        # submit button never appeared. Keyed widgets + st.button work reliably.
        with tab_break:
            st.caption("Fire when a ticker's breakout score crosses your threshold.")
            thr = st.number_input(
                "Breakout score ≥",
                min_value=0.0,
                value=8.0,
                step=0.5,
                key="alert_break_thr",
            )
            wl_only = st.checkbox(
                "Limit to my watchlist tickers", value=False, key="alert_break_wl"
            )
            if st.button("Create breakout alert", key="alert_break_btn"):
                _guarded_create(
                    existing,
                    max_alerts,
                    lambda: create_alert(
                        user_id,
                        "breakout",
                        threshold=float(thr),
                        watchlist_only=bool(wl_only),
                    ),
                )

        with tab_watch:
            st.caption(
                "Fire when any ticker on your watchlist shows up in the scan results."
            )
            if not watch_tickers:
                st.info("Add tickers to a watchlist first to use this alert.")
            if st.button("Create watchlist alert", key="alert_watch_btn"):
                _guarded_create(
                    existing,
                    max_alerts,
                    lambda: create_alert(user_id, "watchlist"),
                )

        with tab_price:
            st.caption("Fire when a specific ticker crosses a price you set.")
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                tk = st.text_input(
                    "Ticker", value="", placeholder="e.g. AAPL", key="alert_price_tk"
                )
            with c2:
                direction = st.selectbox(
                    "Direction", ["above", "below"], key="alert_price_dir"
                )
            with c3:
                target = st.number_input(
                    "Price", min_value=0.0, value=0.0, step=1.0, key="alert_price_val"
                )
            if st.button("Create price alert", key="alert_price_btn"):
                if not tk.strip():
                    st.warning("Enter a ticker symbol.")
                elif float(target) <= 0:
                    st.warning("Enter a price greater than 0.")
                else:
                    _guarded_create(
                        existing,
                        max_alerts,
                        lambda: create_alert(
                            user_id,
                            "price",
                            ticker=tk.strip().upper(),
                            threshold=float(target),
                            direction=direction,
                        ),
                    )

    # --- Existing alerts ---
    if existing:
        st.markdown("**Your alerts**")
        for a in existing:
            cols = st.columns([6, 2, 2])
            label = _fmt_alert(a)
            if not a.get("enabled"):
                label = f"~~{label}~~ (paused)"
            cols[0].markdown(label)
            new_enabled = cols[1].toggle(
                "On", value=bool(a.get("enabled")), key=f"alert_tog_{a['id']}"
            )
            if new_enabled != bool(a.get("enabled")):
                try:
                    set_alert_enabled(a["id"], user_id, new_enabled)
                    st.rerun()
                except Exception:
                    st.warning("Could not update alert.")
            if cols[2].button("Delete", key=f"alert_del_{a['id']}"):
                try:
                    delete_alert(a["id"], user_id)
                    st.rerun()
                except Exception:
                    st.warning("Could not delete alert.")

    # --- Recently triggered ---
    try:
        events = list_recent_events(user_id, limit=10)
    except Exception:
        events = []
    if events:
        with st.expander(f"🔥 Recently triggered ({len(events)})", expanded=False):
            for ev in events:
                when = ev.get("fired_at")
                when_s = when.strftime("%Y-%m-%d %H:%M UTC") if hasattr(when, "strftime") else ""
                st.markdown(f"**{when_s}** — {ev.get('message', '')}")


def _guarded_create(existing: list, max_per_user: int, do_create) -> None:
    """Enforce the per-user cap, run the create, and refresh."""
    if len(existing) >= int(max_per_user):
        st.warning(f"You've reached the maximum of {max_per_user} alerts.")
        return
    try:
        do_create()
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        # Surface the real reason instead of a vague message so DB/permission
        # issues are diagnosable.
        st.error(f"Could not create alert: {type(e).__name__}: {e}")
        with st.expander("Create error details", expanded=True):
            st.code(repr(e))
        return
    # toast survives the rerun so the user gets confirmation even though the
    # script restarts immediately to refresh the alert list.
    try:
        st.toast("✅ Alert created")
    except Exception:
        pass
    st.rerun()
