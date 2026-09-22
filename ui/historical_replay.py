"""Run 43 — Historical Replay & Signal Timeline (thin Streamlit surface).

Renders a symbol's session replay from persisted canonical observations. All logic
is in analytics.replay (which reuses Run 40); this file only lays out the timeline,
inspector, and the strictly-separated "What happened after" section. Guarded,
headless-safe, no market scan, no model inference — reads persisted observations
only (bounded per-symbol query).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from analytics import replay

_LIFECYCLE_ICON = {"NEW": "🆕", "STRENGTHENING": "📈", "ACTIVE": "▪️",
                   "WEAKENING": "📉", "RESOLVED": "✔️"}
_IMPORTANCE_ICON = {"MAJOR": "🔴", "NOTABLE": "🟠", "INFO": "⚪"}


def _load_symbol_observations(symbol: str) -> List[Dict[str, Any]]:
    try:
        from db.hsf_observations import load_observations_for_symbol
        return load_observations_for_symbol(symbol, limit=1000) or []
    except Exception:
        return []


def _outcomes_map(observations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Extract each observation's attached outcomes keyed by its timestamp."""
    out: Dict[str, Dict[str, Any]] = {}
    for o in observations:
        ocs = o.get("outcomes") or {}
        if ocs:
            out[str(o.get("scan_timestamp") or o.get("timestamp"))] = ocs
    return out


def _hhmm(ts: str) -> str:
    return str(ts)[11:16] if len(str(ts)) >= 16 else str(ts)


def render_historical_replay(symbol: Optional[str] = None) -> None:
    """Signal timeline for a symbol. No-op headless; never raises."""
    try:
        import streamlit as st
    except Exception:
        return
    try:
        symbol = (symbol or st.session_state.get("_replay_symbol")
                  or st.text_input("Symbol", key="replay_symbol_input")).strip().upper()
        if not symbol:
            st.caption("Select a symbol to replay its signal timeline.")
            return

        observations = _load_symbol_observations(symbol)
        if not observations:
            st.info(f"No historical production observations for {symbol} yet. "
                    "Replay is available once scheduled capture has recorded this "
                    "symbol.")
            return

        dates = replay.available_dates(observations, symbol)
        if not dates:
            st.info(f"No scheduled production replay data for {symbol} yet.")
            return
        st.caption(f"Historical production replay available for {len(dates)} "
                   f"session(s): {dates[0]} → {dates[-1]}.")
        date = st.selectbox("Session date", list(reversed(dates)), key="replay_date")

        session = replay.build_replay_session(
            symbol, date, observations, outcomes=_outcomes_map(observations))
        s = session["summary"]
        st.markdown(f"### 📽️ {symbol} — {date}")
        st.caption(f"Observations: {s['observations']} · first setup: "
                   f"{_hhmm(s['first_setup']) if s['first_setup'] else '—'} · "
                   f"peak agreement: {s['peak_scanner_agreement']} · priority changes: "
                   f"{s['priority_changes']} · resolved: "
                   f"{_hhmm(s['setup_resolved']) if s['setup_resolved'] else '—'}")
        st.caption("Descriptive session facts — not entry/exit or profit signals. "
                   "Alert Priority is an attention signal, not a prediction.")

        # Timeline (collapsed): stable spans + events.
        st.markdown("#### Timeline")
        for item in session["timeline"]:
            if item["kind"] == "stable":
                st.markdown(f"`{_hhmm(item['from'])}–{_hhmm(item['to'])}`  "
                            f"{item.get('state') or 'ACTIVE'} — {item['note']}")
            else:
                v = item["view"]
                li = _LIFECYCLE_ICON.get(v.get("lifecycle_state"), "")
                label = ("No active setup" if v.get("no_active_setup")
                         else f"{v.get('primary_setup') or 'Setup'} · {v.get('alert_priority')}")
                st.markdown(f"`{_hhmm(item['timestamp'])}`  {li} "
                            f"**{v.get('lifecycle_state')}** — {label}")
                for c in (v.get("changes_since_prior") or [])[:4]:
                    st.caption(f"   • {c}")

        # Point-in-time inspector.
        st.markdown("#### Inspect a moment")
        times = [v["timestamp"] for v in session["observations"]]
        if times:
            chosen = st.select_slider("Time", options=times, value=times[-1],
                                      format_func=_hhmm, key="replay_inspect")
            v = replay.state_at(session, chosen)
            if v:
                _render_inspector(st, v)
                _render_afterward(st, replay.outcomes_at(session, chosen))
    except Exception:
        # Optional explainability feature must never break navigation.
        pass


def _render_inspector(st, v: Dict[str, Any]) -> None:
    st.markdown(f"**{_hhmm(v['timestamp'])}** — what HSF knew at this moment")
    c = st.columns(3)
    c[0].metric("Priority", v.get("alert_priority"))
    c[1].metric("Lifecycle", v.get("lifecycle_state"))
    c[2].metric("Scanners", v.get("scanner_count"))
    reasons = v.get("positive_reasons") or []
    if reasons:
        st.markdown("**Why showing:** " + " · ".join(reasons[:5]))
    risks = v.get("risk_reasons") or []
    if risks:
        st.markdown("**Caution:** " + " · ".join(risks[:3]))
    if v.get("market_regime"):
        st.caption(f"Market regime at this time: {v['market_regime']}")
    snap = v.get("_feature_snapshot") or {}
    if snap:
        with st.expander("Feature snapshot (as captured)", expanded=False):
            st.json({k: snap[k] for k in sorted(snap)})


def _render_afterward(st, outcomes: Dict[str, Any]) -> None:
    """Strictly separated future-outcome section (Task 15/16)."""
    st.markdown("#### What happened afterward")
    st.caption("Separate from the point-in-time state above — outcomes never "
               "influence the reconstructed HSF view.")
    if not outcomes.get("available"):
        st.caption("No matured outcomes for this moment yet (pending).")
        return
    cols = st.columns(4)
    for i, h in enumerate(("+5m", "+15m", "+30m", "+60m")):
        hz = outcomes["horizons"].get(h, {})
        if hz.get("status") == "MATURED" and hz.get("raw_return") is not None:
            cols[i].metric(h, f"{hz['raw_return'] * 100:+.2f}%")
        else:
            cols[i].metric(h, "Pending")
    mfe, mae = outcomes.get("mfe"), outcomes.get("mae")
    if mfe is not None or mae is not None:
        st.caption(f"MFE {mfe * 100:+.2f}%   MAE {mae * 100:+.2f}%"
                   if mfe is not None and mae is not None else "")
