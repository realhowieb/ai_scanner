"""Kalshi BTC scanner panel: directional read for up/down BTC event contracts.

Fetches BTC bars (data.crypto_btc), runs the pure signal engine
(scan.kalshi_signal), and renders the Buy Up / Buy Down call with confidence, a
heuristic probability, the indicator readings, and entry / exit zones. Read-only
— no Kalshi account, no orders. Never raises into the page.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_TIMEFRAMES = ["1m", "5m", "15m", "1h", "6h", "1d"]


def _cache():
    return st.cache_data(ttl=60, show_spinner=False) if st is not None else (lambda f: f)


if st is not None:

    @st.cache_data(ttl=60, show_spinner="Fetching BTC bars…")
    def _bars_cached(timeframe: str):
        from data.crypto_btc import fetch_btc_bars

        return fetch_btc_bars(timeframe)

else:  # pragma: no cover
    def _bars_cached(timeframe: str):
        from data.crypto_btc import fetch_btc_bars

        return fetch_btc_bars(timeframe)


def render_kalshi_scanner() -> None:
    """Render the Kalshi BTC scanner. Never raises."""
    if st is None:
        return
    try:
        st.markdown("## 🪙 Kalshi BTC Scanner")
        st.caption(
            "A directional read on Bitcoin for Kalshi's up/down BTC event "
            "contracts. Educational only — not financial advice; the probability "
            "is a heuristic, not a forecast."
        )
        c1, c2 = st.columns([1, 1])
        timeframe = c1.selectbox(
            "Timeframe", _TIMEFRAMES, index=1, key="kalshi_tf",
            help="Bar size for the indicators. Match it to the contract's window "
                 "(e.g. 5m/15m for hourly BTC markets, 1h/6h for daily).",
        )
        if c2.button("🔄 Refresh", key="kalshi_refresh"):
            _bars_cached.clear()
            st.rerun()

        df = _bars_cached(timeframe)
        if df is None or len(df) < 35:
            st.warning(
                "Couldn't fetch enough BTC data right now (Coinbase). Try another "
                "timeframe or refresh."
            )
            return

        from scan.kalshi_signal import compute_kalshi_signal

        sig = compute_kalshi_signal(df)
        if not sig:
            st.warning("Not enough data to compute a signal on this timeframe.")
            return

        _render_call(sig)
        _render_indicators(sig)
        _render_plan(sig)
        _render_reasons(sig)
        _render_chart(df, sig, timeframe)
        st.caption(
            "Buy Up = bet BTC finishes higher (YES on an up-market); Buy Down = "
            "the opposite. Educational only — not financial advice."
        )
    except Exception:
        pass


def _render_call(sig: Dict[str, Any]) -> None:
    up = sig["direction"] == "up"
    badge = "🟢 **Buy Up**" if up else "🔴 **Buy Down**"
    color = "#16a34a" if up else "#dc2626"
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:10px;border:1px solid {color};"
        f"background:{color}22'>"
        f"<span style='font-size:22px'>{badge}</span> &nbsp; "
        f"<span style='color:#94a3b8'>BTC ${sig['price']:,.0f}</span></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{sig['confidence']}/100")
    m2.metric("Est. probability", f"{sig['probability']}%")
    m3.metric("Volume", f"{sig['rvol']:.2f}× avg" + (" 🔥" if sig["volume_spike"] else ""))


def _render_indicators(sig: Dict[str, Any]) -> None:
    st.markdown("**Indicators**")
    rows = [
        {"Indicator": "EMA 9 / 21", "Reading":
            f"{sig['ema9']:,.0f} / {sig['ema21']:,.0f}",
         "Bias": "▲ up" if sig["ema9"] >= sig["ema21"] else "▼ down"},
        {"Indicator": "RSI(14)", "Reading": f"{sig['rsi']:.0f}",
         "Bias": "▲ up" if sig["rsi"] >= 50 else "▼ down"},
        {"Indicator": "MACD hist", "Reading": f"{sig['macd_hist']:+.1f}",
         "Bias": "▲ up" if sig["macd_hist"] >= 0 else "▼ down"},
        {"Indicator": "VWAP", "Reading":
            (f"{sig['vwap']:,.0f}" if sig["vwap"] is not None else "—"),
         "Bias": ("▲ up" if (sig["vwap"] is not None and sig["price"] >= sig["vwap"])
                  else "▼ down" if sig["vwap"] is not None else "—")},
        {"Indicator": "Support / Resistance",
         "Reading": f"{sig['support']:,.0f} / {sig['resistance']:,.0f}", "Bias": "—"},
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_plan(sig: Dict[str, Any]) -> None:
    lo, hi = sig["entry_zone"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Entry zone", f"${lo:,.0f}–${hi:,.0f}")
    p2.metric("Target / cash-out", f"${sig['target']:,.0f}")
    p3.metric("Stop", f"${sig['stop']:,.0f}")


def _render_reasons(sig: Dict[str, Any]) -> None:
    reasons = sig.get("reasons") or []
    if not reasons:
        return
    with st.expander("Why this call", expanded=False):
        for r in reasons:
            st.markdown(f"- {r}")


def _render_chart(df, sig: Dict[str, Any], timeframe: str) -> None:
    try:
        import plotly.graph_objects as go

        closes = df["Close"].tail(120)
        fig = go.Figure(
            go.Scatter(x=list(closes.index), y=closes.tolist(), mode="lines",
                       line=dict(color="#f59e0b", width=2), name="BTC")
        )
        for label, y, col in (
            ("target", sig["target"], "#16a34a"),
            ("entry", sig["entry_zone"][1], "#60a5fa"),
            ("stop", sig["stop"], "#dc2626"),
        ):
            fig.add_hline(y=y, line_color=col, line_dash="dot", line_width=1,
                          annotation_text=f"{label} {y:,.0f}",
                          annotation_font_color=col, annotation_font_size=11)
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                        key=f"kalshi_chart_{timeframe}")
    except Exception:
        pass
