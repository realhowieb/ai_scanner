"""Deterministic trade plan per result: entry, stop, targets, position size.

Turns a result row the scanner already produced into an actionable plan using
only fields on the row (no network calls): stop distance derives from observed
20-day volatility (clamped to sane bounds), targets are R-multiples, and share
size comes from the user's account size + risk-per-trade percent.

Pure math in build_trade_plan (Mapping in, dict out — unit-testable headless);
rendering is separate and never raises into the results view.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs; pure math still works
    st = None  # type: ignore[assignment]

MIN_STOP_PCT = 2.0
MAX_STOP_PCT = 8.0
TARGET_R = (1.5, 3.0)


def _num(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        try:
            f = float(row[key])
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            return f
    return None


def build_trade_plan(
    row: Mapping[str, Any],
    *,
    account_size: float = 10_000.0,
    risk_pct: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Plan dict for a result row, or None when there's no usable price.

    Stop: half the 20-day volatility below entry, clamped to 2-8%.
    Targets: 1.5R and 3R above entry. Size: risk budget / per-share risk.
    """
    entry = _num(row, "Last", "Close")
    if entry is None or entry <= 0:
        return None
    vol = _num(row, "Volatility20D%")
    stop_pct = min(MAX_STOP_PCT, max(MIN_STOP_PCT, (vol or 6.0) * 0.5))
    stop = entry * (1 - stop_pct / 100.0)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    risk_budget = max(0.0, float(account_size)) * max(0.0, float(risk_pct)) / 100.0
    shares = int(math.floor(risk_budget / risk_per_share)) if risk_budget > 0 else 0
    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "stop_pct": round(stop_pct, 1),
        "targets": [round(entry + r * risk_per_share, 2) for r in TARGET_R],
        "target_r": list(TARGET_R),
        "risk_per_share": round(risk_per_share, 2),
        "shares": shares,
        "risk_budget": round(risk_budget, 2),
    }


def render_trade_plan(row: Mapping[str, Any], *, locked: bool = False) -> None:
    """Trade-plan block inside the ticker details expander. Never raises."""
    if st is None:
        return
    try:
        st.markdown("**🎯 Trade plan**")
        if locked:
            st.caption(
                "🔒 Entry, stop, targets, and position sizing are a **Pro** "
                "feature — upgrade to see the full plan."
            )
            return
        c1, c2 = st.columns(2)
        account = c1.number_input(
            "Account size ($)", min_value=100.0, value=10_000.0, step=500.0,
            key="tp_account",
        )
        risk = c2.number_input(
            "Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0,
            step=0.25, key="tp_risk",
        )
        plan = build_trade_plan(row, account_size=account, risk_pct=risk)
        if not plan:
            st.caption("Not enough price data for a plan.")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", f"{plan['entry']:,.2f}")
        m2.metric("Stop", f"{plan['stop']:,.2f}", f"-{plan['stop_pct']:g}%")
        m3.metric(f"Target {plan['target_r'][0]:g}R", f"{plan['targets'][0]:,.2f}")
        m4.metric(f"Target {plan['target_r'][1]:g}R", f"{plan['targets'][1]:,.2f}")
        st.caption(
            f"Size: **{plan['shares']} shares** risks ~${plan['risk_budget']:,.0f} "
            f"(${plan['risk_per_share']:,.2f}/share). Educational only — not "
            "financial advice; adjust for your own strategy."
        )
        try:
            from ui.journal import log_trade_button

            log_trade_button(row, shares=plan["shares"], key=f"tp_log_{row.get('Ticker')}")
        except Exception:
            pass
        _render_alert_quick_action(row)
    except Exception:
        pass


def _render_alert_quick_action(row: Mapping[str, Any]) -> None:
    """'Alert me on this' — pre-fill the price-alert form and jump to Alerts.

    Keeps in-context alert creation alive now that alert management lives on
    its own page: sets the form's session keys before that page renders its
    widgets (the established chip-prefill pattern), then switches pages.
    """
    try:
        ticker = str(row.get("Ticker") or "").upper() if hasattr(row, "get") else ""
        last = _num(row, "Last", "Close")
        if not ticker or last is None:
            return
        if st.button(f"🔔 Alert me on {ticker}", key=f"tp_alert_{ticker}"):
            st.session_state["alert_price_tk"] = ticker
            st.session_state["alert_price_val"] = round(float(last), 2)
            st.switch_page("pages/alerts.py")
    except Exception:
        pass
