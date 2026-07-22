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


def score_components(row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Decompose a BreakoutScore into its scoring contributions.

    Mirrors scan/breakout's clipped formula; anything not reconstructable from
    the row lands in 'other' so the parts always sum to the printed score.
    """
    score = _num(row, "BreakoutScore")
    if score is None:
        return None

    def clip(v, lo, hi):
        return max(lo, min(hi, v))

    parts: Dict[str, float] = {}
    pct = _num(row, "PctChange")
    if pct is not None:
        parts["Day move"] = clip(pct, -30.0, 30.0)
    t20 = _num(row, "Trend20D%")
    if t20 is not None and t20 > 0:
        parts["Trend 20d"] = clip(t20, 0.0, 50.0) * 0.4
    t10 = _num(row, "Trend10D%")
    if t10 is not None and t10 > 0:
        parts["Trend 10d"] = clip(t10, 0.0, 50.0) * 0.6
    pos = _num(row, "BreakoutPos20D")
    if pos is not None:
        parts["Near 20d high"] = (pos - 0.9) * 50.0
    gap = _num(row, "GapPct")
    if gap is not None:
        parts["Gap"] = clip(gap, -30.0, 30.0) * 0.5
    rv = _num(row, "VolRel20", "RelVol")
    if rv is not None and rv > 0:
        parts["Rel volume"] = (clip(rv, 1.0, 5.0) - 1.0) * 10.0
    if row.get("IsBreakout"):
        parts["Breakout bonus"] = 5.0
    residual = score - sum(parts.values())
    if abs(residual) >= 0.5:
        parts["Other"] = residual
    return {k: round(v, 1) for k, v in parts.items()}


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


def render_trade_plan(row: Mapping[str, Any], *, locked: bool = False, key: str = "main") -> None:
    """Trade-plan block inside the ticker details expander. Never raises.

    ``key`` namespaces the account/risk inputs so the plan can render on more
    than one tab in a single script run (Latest results + Scan History) without
    a duplicate-widget-id collision.
    """
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
            key=f"tp_account_{key}",
        )
        risk = c2.number_input(
            "Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0,
            step=0.25, key=f"tp_risk_{key}",
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
        _render_plan_chart(row, plan, key=key)
        _render_score_waterfall(row, key=key)
        try:
            from ui.journal import log_trade_button

            log_trade_button(row, shares=plan["shares"], key=f"tp_log_{key}_{row.get('Ticker')}")
        except Exception:
            pass
        try:
            from ui.paper_trade import render_paper_trade_button

            render_paper_trade_button(row, shares=plan["shares"], plan=plan, key=key)
        except Exception:
            pass
        _render_alert_quick_action(row, key=key)
    except Exception:
        pass


def _render_alert_quick_action(row: Mapping[str, Any], *, key: str = "main") -> None:
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
        if st.button(f"🔔 Alert me on {ticker}", key=f"tp_alert_{key}_{ticker}"):
            st.session_state["alert_price_tk"] = ticker
            st.session_state["alert_price_val"] = round(float(last), 2)
            st.switch_page("pages/alerts.py")
    except Exception:
        pass


def _render_plan_chart(row: Mapping[str, Any], plan: Dict[str, Any], *, key: str = "main") -> None:
    """Draw the plan as risk geometry: 60d closes + entry/stop/target lines."""
    try:
        ticker = str(row.get("Ticker") or "").upper()
        if not ticker:
            return
        import plotly.graph_objects as go

        bars = _plan_bars(ticker)
        if bars is None or len(bars) < 10:
            return
        closes = bars["Close"].dropna()
        fig = go.Figure(
            go.Scatter(
                x=list(closes.index), y=closes.tolist(), mode="lines",
                line=dict(color="#94a3b8", width=2), name=ticker,
                hovertemplate="%{x|%b %d}: %{y:,.2f}<extra></extra>",
            )
        )
        for label, y, color in (
            ("entry", plan["entry"], "#60a5fa"),
            ("stop", plan["stop"], "#dc2626"),
            (f"{plan['target_r'][0]:g}R", plan["targets"][0], "#16a34a"),
            (f"{plan['target_r'][1]:g}R", plan["targets"][1], "#16a34a"),
        ):
            fig.add_hline(
                y=y, line_color=color, line_dash="dot", line_width=1,
                annotation_text=f"{label} {y:,.2f}", annotation_font_size=11,
                annotation_font_color=color,
            )
        fig.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                        key=f"tp_plan_chart_{key}_{row.get('Ticker')}")
    except Exception:
        pass


def _plan_bars(ticker: str):
    """60d daily bars for one symbol, cached (Alpaca)."""
    try:
        return _plan_bars_cached(ticker)
    except Exception:
        return None


if st is not None:

    @st.cache_data(ttl=300, show_spinner=False)
    def _plan_bars_cached(ticker: str):
        from data.price_alpaca import download_multi_alpaca

        frames = download_multi_alpaca([ticker], period="60d", interval="1d",
                                       prepost=False, timeout_s=10.0)
        return frames.get(ticker)

else:  # pragma: no cover
    def _plan_bars_cached(ticker: str):
        return None


def _render_score_waterfall(row: Mapping[str, Any], *, key: str = "main") -> None:
    """Where the BreakoutScore came from, as a waterfall."""
    try:
        parts = score_components(row)
        if not parts or len(parts) < 2:
            return
        import plotly.graph_objects as go

        labels = list(parts.keys()) + ["Score"]
        values = list(parts.values()) + [0]
        fig = go.Figure(
            go.Waterfall(
                x=labels,
                y=values,
                measure=["relative"] * len(parts) + ["total"],
                increasing=dict(marker=dict(color="#16a34a")),
                decreasing=dict(marker=dict(color="#dc2626")),
                totals=dict(marker=dict(color="#60a5fa")),
                connector=dict(line=dict(color="rgba(128,128,128,0.3)", width=1)),
                text=[f"{v:+.1f}" for v in parts.values()] + [""],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text="Where the score comes from", font=dict(size=13)),
            height=220, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                        key=f"tp_waterfall_{key}_{row.get('Ticker')}")
    except Exception:
        pass
