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
    if t == "move":
        return f"⚡ {a.get('ticker')} moves ±{float(a.get('threshold') or 0):g}% today (live)"
    if t == "rvol":
        return f"📊 {a.get('ticker')} RVOL ≥ {float(a.get('threshold') or 0):g}× (live)"
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
        if used > int(max_alerts):
            st.caption(
                f"Only your newest {int(max_alerts)} alert(s) can actively fire on this plan. "
                "Older alerts stay saved so you can delete them or upgrade later."
            )
    else:
        st.caption(f"Using {used} of {max_alerts} alert slots on your plan.")

    with st.expander("➕ Create an alert", expanded=True):
        tab_break, tab_watch, tab_price, tab_move, tab_rvol = st.tabs(
            ["🚀 Breakout", "📋 Watchlist", "💲 Price", "⚡ % Move", "📊 RVOL"]
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
            # Smart create: show the observed score distribution and how often
            # this threshold would have fired, so dead/spam thresholds are
            # caught before the alert exists. Guarded — insight is optional.
            try:
                from ui.alert_preview import render_breakout_threshold_insight

                render_breakout_threshold_insight(float(thr))
            except Exception:
                pass
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
            # Immediate-fire check: if the condition is already true at the
            # current price, say so before the user creates it.
            if tk.strip() and float(target) > 0:
                try:
                    from market_data import get_latest_quotes

                    q = (get_latest_quotes([tk.strip().upper()]) or {}).get(tk.strip().upper())
                    last = q.get("last") if isinstance(q, dict) else None
                    if last is not None:
                        already = (direction == "above" and last >= float(target)) or (
                            direction == "below" and last <= float(target)
                        )
                        note = " — this would fire on the next check" if already else ""
                        st.caption(f"{tk.strip().upper()} last: **{last:,.2f}**{note}")
                except Exception:
                    pass
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

        with tab_move:
            st.caption(
                "Fire when a ticker moves more than ±X% vs yesterday's close. "
                "Checked live (~60s) during extended hours."
            )
            mc1, mc2 = st.columns([2, 2])
            mv_tk = mc1.text_input("Ticker", value="", placeholder="e.g. NVDA", key="alert_move_tk")
            mv_thr = mc2.number_input(
                "Move % ≥", min_value=0.5, value=5.0, step=0.5, key="alert_move_thr"
            )
            if st.button("Create % move alert", key="alert_move_btn"):
                if not mv_tk.strip():
                    st.warning("Enter a ticker symbol.")
                else:
                    _guarded_create(
                        existing,
                        max_alerts,
                        lambda: create_alert(
                            user_id, "move", ticker=mv_tk.strip().upper(), threshold=float(mv_thr)
                        ),
                    )

        with tab_rvol:
            st.caption(
                "Fire when a ticker trades at X times its 20-day average volume. "
                "Checked live (~60s) during extended hours."
            )
            rc1, rc2 = st.columns([2, 2])
            rv_tk = rc1.text_input("Ticker", value="", placeholder="e.g. TSLA", key="alert_rvol_tk")
            rv_thr = rc2.number_input(
                "RVOL ≥", min_value=1.0, value=2.0, step=0.5, key="alert_rvol_thr"
            )
            if st.button("Create RVOL alert", key="alert_rvol_btn"):
                if not rv_tk.strip():
                    st.warning("Enter a ticker symbol.")
                else:
                    _guarded_create(
                        existing,
                        max_alerts,
                        lambda: create_alert(
                            user_id, "rvol", ticker=rv_tk.strip().upper(), threshold=float(rv_thr)
                        ),
                    )

    # --- Existing alerts ---
    if existing:
        # Outcome scorecards: how each alert's recent fires actually played out.
        # Shown only with >=3 scored fires so one lucky/unlucky fire can't
        # masquerade as a track record.
        try:
            from db.alert_outcomes import HIT_TARGET_PCT, HORIZON_DAYS, scorecards_for_user

            scorecards = scorecards_for_user(user_id)
        except Exception:
            scorecards, HIT_TARGET_PCT, HORIZON_DAYS = {}, 5.0, 3

        st.markdown("**Your alerts**")
        for index, a in enumerate(existing):
            cols = st.columns([6, 2, 2])
            label = _fmt_alert(a)
            if not a.get("enabled"):
                label = f"~~{label}~~ (paused)"
            elif index >= int(max_alerts):
                label = f"{label} _(over plan limit)_"
            cols[0].markdown(label)
            card = scorecards.get(a.get("id"))
            if card and card.get("fires", 0) >= 3:
                avg = card.get("avg_return_pct")
                avg_s = f" · avg {avg:+.1f}%/{HORIZON_DAYS}d" if avg is not None else ""
                cols[0].caption(
                    f"🎯 {card['hits']}/{card['fires']} recent fires hit "
                    f"+{HIT_TARGET_PCT:g}% within {HORIZON_DAYS}d{avg_s}"
                )
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
        events = list_recent_events(user_id, limit=25)
    except Exception:
        events = []
    if events:
        # Collapse repeats: manual/forced runs can record the same fire several
        # times in a day; show it once with a repeat count instead of a wall of
        # identical entries.
        deduped: list = []
        for ev in events:
            msg = ev.get("message", "")
            if deduped and deduped[-1][0].get("message", "") == msg:
                deduped[-1][1] += 1
            else:
                deduped.append([ev, 1])
        deduped = deduped[:10]
        with st.expander(f"🔥 Recently triggered ({len(deduped)})", expanded=False):
            for ev, count in deduped:
                when = ev.get("fired_at")
                when_s = when.strftime("%Y-%m-%d %H:%M UTC") if hasattr(when, "strftime") else ""
                repeat = f" _(×{count} today)_" if count > 1 else ""
                st.markdown(f"**{when_s}** — {ev.get('message', '')}{repeat}")


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
