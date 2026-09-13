"""Compact HSF Intelligence Alerts UI (preferences + recent feed).

Read/write user preferences and show the recent state-change feed. Detection and
delivery are owned by the background pipeline — this page never evaluates or
delivers alerts (read-only except the explicit 'Save preferences' action).
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_PREF_ROWS = [
    ("new", "New HSF opportunity"),
    ("upgrade", "Becomes stronger (status upgrade)"),
    ("downgrade", "Weakens (status downgrade)"),
    ("fading", "Starts fading"),
    ("dropped", "Drops from HSF opportunities"),
    ("rising", "HSF Score rising"),
    ("falling", "HSF Score falling"),
    ("signal_added", "Confirming signal added"),
    ("signal_removed", "Confirming signal removed"),
]
_SEV_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "⚪"}


def render_intelligence_alerts(username: str) -> None:
    """Render preferences + recent feed for one user. Never raises."""
    if st is None or not username:
        return
    try:
        from analytics.opportunity_events import DEFAULT_PREFERENCES
        from db.intelligence_alerts import (
            get_hsf_alert_prefs,
            list_recent_intelligence_alerts,
            set_hsf_alert_prefs,
        )
    except Exception:
        return

    st.markdown("### 🔔 HSF Intelligence Alerts")
    st.caption("Get notified when something meaningful changes about the HSF "
               "opportunities on your watchlist — not on every scan. Detection "
               "runs in the background.")

    prefs = {**DEFAULT_PREFERENCES, **(get_hsf_alert_prefs(username) or {})}
    with st.expander("⚙️ Alert preferences", expanded=False):
        st.caption("Notify me when a watched ticker:")
        new_prefs: Dict[str, Any] = {}
        cols = st.columns(2)
        for i, (key, label) in enumerate(_PREF_ROWS):
            new_prefs[key] = cols[i % 2].checkbox(label, value=bool(prefs.get(key)),
                                                  key=f"hsf_pref_{key}")
        if st.button("Save preferences", key="hsf_pref_save"):
            if set_hsf_alert_prefs(username, new_prefs):
                st.success("Preferences saved.")
            else:
                st.caption("Couldn't save preferences right now.")

    st.markdown("#### Recent HSF intelligence alerts")
    try:
        recent = list_recent_intelligence_alerts(username, limit=30)
    except Exception:
        recent = []
    if not recent:
        st.caption("No HSF intelligence alerts yet. HSF will record meaningful "
                   "state changes for the opportunities you follow.")
        return
    _feed_filter = st.radio("Filter", ["All", "Upgrades", "Downgrades", "New", "Fading", "Dropped"],
                            horizontal=True, key="hsf_feed_filter", label_visibility="collapsed")
    keep = {
        "Upgrades": {"STATUS_UPGRADE"}, "Downgrades": {"STATUS_DOWNGRADE"},
        "New": {"NEW_OPPORTUNITY"}, "Fading": {"FADING"}, "Dropped": {"DROPPED"},
    }.get(_feed_filter)
    shown = 0
    for r in recent:
        if keep and r.get("event_type") not in keep:
            continue
        shown += 1
        icon = _SEV_ICON.get(str(r.get("severity") or "").upper(), "")
        try:
            ts = r["detected_at"].strftime("%b %d %I:%M %p") if hasattr(r.get("detected_at"), "strftime") else str(r.get("detected_at"))
        except Exception:
            ts = str(r.get("detected_at"))
        copy = (r.get("copy") or f"{r.get('ticker')} · {r.get('event_type')}").replace("\n", " · ")
        status = r.get("delivery_status")
        tail = f"  ·  _{status.lower()}_" if status and status != "DELIVERED" else ""
        st.markdown(f"{icon} **{ts}** — {copy}{tail}")
        if r.get("ticker") and st.button(f"🔬 {r['ticker']} intel", key=f"hsf_feed_intel_{shown}_{r['ticker']}"):
            st.session_state["hsf_stock_ticker"] = str(r["ticker"]).upper()
            try:
                st.switch_page("pages/stock.py")
            except Exception:
                st.caption("Open 'Stock Intel' from the sidebar.")
    if keep and shown == 0:
        st.caption(f"No {_feed_filter.lower()} alerts in the recent feed.")

    # Admin-only operational health + alert quality (read-only; no evaluation,
    # no writes, no delivery, no Claude). Health and quality stay separate.
    try:
        if (st.session_state.get("entitlements") or {}).get("can_diagnostics"):
            _render_health()
            _render_quality()
            _render_opportunity_outcomes()
    except Exception:
        pass


_HEALTH_ICON = {"HEALTHY": "🟢", "DEGRADED": "🟠", "STALE": "🟠", "UNKNOWN": "⚪"}


def _render_health() -> None:
    try:
        from db.intelligence_alerts import get_intelligence_health, list_recent_evaluation_runs

        h = get_intelligence_health()
        runs = list_recent_evaluation_runs(limit=15)
    except Exception:
        return
    with st.expander("🩺 HSF Intelligence health (admin)", expanded=False):
        icon = _HEALTH_ICON.get(str(h.get("status")), "")
        st.markdown(f"**Status: {icon} {h.get('status')}**"
                    + (f" — {h['reason']}" if h.get("reason") else ""))
        c1, c2, c3 = st.columns(3)
        c1.caption(f"Last run: {h.get('last_run_at')}")
        c2.caption(f"Last success: {h.get('last_success_at')}")
        c3.caption(f"Since success: {h.get('minutes_since_last_success')} min"
                   if h.get("minutes_since_last_success") is not None else "Since success: —")
        lm = h.get("latest_metrics") or {}
        if lm:
            st.caption(
                f"Latest — events {lm.get('events_detected')} · users {lm.get('users_evaluated')} · "
                f"matched {lm.get('notifications_matched')} · delivered {lm.get('delivered')} · "
                f"deduped {lm.get('deduped')} · filtered {lm.get('filtered_by_preferences')} · "
                f"failed {lm.get('failed')}")
        if runs:
            st.markdown("**Recent evaluations**")
            st.dataframe(
                [{"Time": r.get("started_at"), "Status": r.get("status"),
                  "Events": r.get("events_detected"), "Matched": r.get("notifications_matched"),
                  "Delivered": r.get("delivered"), "Failed": r.get("failed"),
                  "Dur ms": r.get("duration_ms"), "Stage": r.get("error_stage")} for r in runs],
                hide_index=True, width="stretch")


def _horizon_labels() -> dict:
    """Honest elapsed-time labels from the ONE canonical horizon definition."""
    try:
        from analytics.alert_quality import get_quality_horizons

        return {h["key"]: h["label"] for h in get_quality_horizons()}
    except Exception:
        return {}


def _pct(x) -> str:
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "—"


def _render_quality() -> None:
    """Compact alert-quality panel. Measures HSF state follow-through, never
    price, never 'win rate'. Read-only. Separate from operational health."""
    try:
        from db.intelligence_alerts import get_alert_quality_summary

        q = get_alert_quality_summary()
    except Exception:
        return
    with st.expander("📐 HSF Intelligence alert quality (admin)", expanded=False):
        if not q.get("available"):
            st.caption("Alert quality data is unavailable right now.")
            return
        st.caption("Did alerts identify meaningful subsequent HSF state changes? "
                   "Measured from frozen alert-time state vs later canonical HSF "
                   "observations — not price, not investment performance.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Alerts", q.get("alerts_total"))
        c2.metric("Matured", q.get("matured"))
        c3.metric("Pending", q.get("pending"))
        c4.metric("Confirmed", _pct(q.get("confirmation_rate")))
        c5.metric("Reversed", _pct(q.get("reversal_rate")))
        freq = q.get("frequency") or {}
        st.caption(
            f"No follow-up data: {q.get('unavailable')} · "
            f"dedupe rate {_pct(freq.get('dedupe_rate'))} · "
            f"filtered {_pct(freq.get('filter_rate'))} · "
            f"alerts/ticker {freq.get('alerts_per_ticker'):.1f}"
            if isinstance(freq.get("alerts_per_ticker"), (int, float))
            else f"No follow-up data: {q.get('unavailable')} · dedupe rate {_pct(freq.get('dedupe_rate'))}")
        by_event = q.get("by_event_type") or []
        if by_event:
            st.markdown("**Quality by event type**")
            st.dataframe(
                [{"Event": e["event_type"], "Matured": e["matured"],
                  "Confirmed": e["confirmed"], "Reversed": e["reversed"],
                  "Confirmation": _pct(e.get("confirmation_rate")),
                  "Assessment": e["assessment"]} for e in by_event],
                hide_index=True, width="stretch")
        by_h = q.get("by_horizon") or []
        if by_h:
            labels = _horizon_labels()
            st.markdown("**By horizon** (elapsed wall-clock, not trading days)")
            st.dataframe(
                [{"Horizon": labels.get(h["horizon"], h["horizon"]),
                  "Matured": h["matured"], "Confirmed": h["confirmed"],
                  "Reversed": h["reversed"], "Confirmation": _pct(h.get("confirmation_rate")),
                  "Assessment": h["assessment"]} for h in by_h],
                hide_index=True, width="stretch")
        st.caption(f"Rates shown only at ≥ {q.get('min_sample')} matured observations; "
                   "smaller samples read INSUFFICIENT_SAMPLE. Measurement only — "
                   "alert behavior is unchanged.")


_HLBL = {"NEXT": "Next obs", "H24": "24h+", "H72": "72h+", "H120": "120h+"}


def _outcome_rows(rows, key_label="Group") -> list:
    out = []
    for r in rows:
        out.append({
            key_label: r.get("key"), "N": r.get("matured"),
            "Strengthened": r.get("strengthened"), "Persisted": r.get("persisted"),
            "Weakened": r.get("weakened"), "Faded": r.get("faded"),
            "Dropped": r.get("dropped"), "Recovered": r.get("recovered"),
            "Follow-through": (_pct(r.get("follow_through_rate")) if r.get("assessment") == "OK"
                               else "insufficient"),
        })
    return out


def _render_opportunity_outcomes() -> None:
    """Run 25 — what happened to ALL frozen HSF opportunities (not just alerted
    ones). HSF-state persistence only, never price. Read-only. Separate from
    alert quality."""
    try:
        from db.opportunity_outcomes import get_opportunity_outcome_summary

        s = get_opportunity_outcome_summary()
    except Exception:
        return
    with st.expander("🧭 HSF Opportunity outcome intelligence (admin)", expanded=False):
        if not s.get("available") or not s.get("matured"):
            st.caption("No matured opportunity outcomes yet. HSF records the "
                       "subsequent state of every frozen opportunity over time.")
            return
        st.caption("What happened to frozen HSF opportunities after identification "
                   "— subsequent canonical HSF state, not price or returns.")
        c1, c2 = st.columns(2)
        c1.metric("Matured outcomes", s.get("matured"))
        c2.metric("Comparable", s.get("comparable"))
        for title, dim, lbl in [
            ("By initial status", "by_status", "Status"),
            ("By HSF score band", "by_score_band", "Band"),
            ("By confirming-signal count", "by_signal_count", "Signals"),
            ("By horizon (elapsed)", "by_horizon", "Horizon"),
            ("By market regime", "by_regime", "Regime"),
        ]:
            rows = s.get(dim) or []
            if not rows:
                continue
            if dim == "by_horizon":
                for r in rows:
                    r["key"] = _HLBL.get(r.get("key"), r.get("key"))
            st.markdown(f"**{title}**")
            st.dataframe(_outcome_rows(rows, lbl), hide_index=True, width="stretch")
        st.caption(f"Follow-through = strengthened or persisted (held). Recovered is "
                   f"reported separately (it starts from a degraded state). Rates only "
                   f"at ≥ {s.get('min_sample')} comparable (excl. VERSION_CHANGED). "
                   "Observation only — HSF Score and alerts are unchanged.")
