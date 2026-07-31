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
# Auto-refresh cadences. Kept short because the 15-min BTC contracts churn fast.
_AUTO = {"Off": 0, "15s": 15, "30s": 30, "60s": 60}


if st is not None:

    # Short TTLs so an auto-refresh tick (fragment run_every) actually re-pulls
    # fresh data instead of serving a stale cache entry.
    @st.cache_data(ttl=30, show_spinner="Fetching BTC bars…")
    def _bars_cached(timeframe: str):
        from data.crypto_btc import fetch_btc_bars

        return fetch_btc_bars(timeframe)

    @st.cache_data(ttl=15, show_spinner=False)
    def _markets_cached():
        from data.kalshi_markets import fetch_btc_markets

        return fetch_btc_markets()

    @st.cache_data(ttl=10, show_spinner=False)
    def _btc15_cached(spot_bucket: float):
        from data.kalshi_markets import fetch_btc_15min

        return fetch_btc_15min(spot_bucket)

else:  # pragma: no cover
    def _bars_cached(timeframe: str):
        from data.crypto_btc import fetch_btc_bars

        return fetch_btc_bars(timeframe)

    def _markets_cached():
        from data.kalshi_markets import fetch_btc_markets

        return fetch_btc_markets()

    def _btc15_cached(spot_bucket: float):
        from data.kalshi_markets import fetch_btc_15min

        return fetch_btc_15min(spot_bucket)


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
        c1, c2, c3 = st.columns([1, 1, 1])
        timeframe = c1.selectbox(
            "Timeframe", _TIMEFRAMES, index=1, key="kalshi_tf",
            help="Bar size for the indicators. Match it to the contract's window "
                 "(e.g. 5m/15m for hourly BTC markets, 1h/6h for daily).",
        )
        auto = c2.selectbox(
            "Auto-refresh", list(_AUTO.keys()), index=0, key="kalshi_auto",
            help="Re-pull BTC bars + Kalshi markets on a timer — handy for the "
                 "fast-churning 15-minute BTC contracts.",
        )
        with c3:
            st.write("")
            if st.button("🔄 Refresh", key="kalshi_refresh"):
                _bars_cached.clear()
                _markets_cached.clear()
                st.rerun()

        interval = _AUTO.get(auto, 0)

        def _body() -> None:
            df = _bars_cached(timeframe)
            if df is None or len(df) < 35:
                st.warning(
                    "Couldn't fetch enough BTC data right now (Coinbase). Try "
                    "another timeframe or refresh."
                )
                return
            from scan.kalshi_signal import compute_kalshi_signal

            # Higher-timeframe confirmation (#4): the 1h trend. Skipped implicitly
            # if the fetch fails (htf_df=None).
            htf = _bars_cached("1h") if timeframe not in ("1h", "6h", "1d") else None
            sig = compute_kalshi_signal(df, htf_df=htf)
            if not sig:
                st.warning("Not enough data to compute a signal on this timeframe.")
                return
            _render_call(sig)
            _render_15min(sig)
            _render_indicators(sig)
            if sig.get("tradeable") and sig.get("entry_zone"):
                _render_plan(sig)
            _render_reasons(sig)
            _render_chart(df, sig, timeframe)
            _render_kalshi_markets(sig)
            st.caption(
                "A decision engine, not a signal: it says No Trade when signals "
                "don't align. Educational only — not financial advice."
            )

        # Auto-refresh re-renders just this block on a timer (short cache TTLs
        # above make each tick a fresh pull). Falls back to a full-page timer
        # when st.fragment isn't available.
        if interval and hasattr(st, "fragment"):
            st.fragment(run_every=f"{interval}s")(_body)()
        else:
            if interval:
                try:
                    from streamlit_autorefresh import st_autorefresh

                    st_autorefresh(interval=interval * 1000, key="kalshi_autorefresh")
                except Exception:
                    st.caption("Auto-refresh unavailable here; use 🔄 Refresh.")
            _body()
    except Exception:
        pass


def _render_call(sig: Dict[str, Any]) -> None:
    rec = sig["recommendation"]
    direction = sig["direction"]
    if direction == "up":
        icon, color = "🟢", "#16a34a"
    elif direction == "down":
        icon, color = "🔴", "#dc2626"
    else:
        icon, color = "⚪", "#64748b"
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:10px;border:1px solid {color};"
        f"background:{color}22'>"
        f"<span style='font-size:22px'>{icon} <b>{rec}</b></span> &nbsp; "
        f"<span style='color:#94a3b8'>BTC ${sig['price']:,.0f}</span></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{sig['confidence']}/100")
    m2.metric("Win prob", f"{sig['win_probability']}%" if sig.get("win_probability") else "—")
    m3.metric("Size", sig.get("position_size") or "—")
    m4.metric("Volume", f"{sig['rvol']:.2f}×" + (" 🔥" if sig["volume_spike"] else ""))
    # Gate checklist — what the decision required.
    def _mk(ok):
        return "✅" if ok else "❌"
    htf = sig.get("htf_agree")
    htf_txt = "—" if htf is None else (_mk(True) if htf else _mk(False))
    st.caption(
        f"Trend aligned {_mk(sig['trend_aligned'])} · "
        f"Volume ≥ 0.7× {_mk(sig['volume_ok'])} · "
        f"Volatility ok {_mk(sig['volatility_ok'])} · "
        f"1h agrees {htf_txt}"
    )
    if not sig.get("tradeable") and sig.get("gate_reasons"):
        st.caption("⚪ No Trade — " + "; ".join(sig["gate_reasons"]))


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
    contribs = sig.get("contributions") or []
    if not contribs:
        return
    with st.expander("What drove this — indicator contributions", expanded=False):
        for name, pts in contribs:
            arrow = "＋" if pts >= 0 else "－"
            st.markdown(f"- {arrow} **{name}**: {pts:+.0f} pts")
        st.caption("Positive points push conviction up; negative (e.g. a 1h-trend "
                   "disagreement) pull it down toward No Trade.")


def _render_15min(sig: Dict[str, Any]) -> None:
    """Focused 'next 15 minutes: BTC higher?' read on the KXBTC15M market."""
    try:
        # Bucket spot to $50 so the cache key is stable between ticks.
        spot = float(sig["price"])
        m = _btc15_cached(round(spot / 50.0) * 50.0)
        st.markdown("### ⏱️ Next 15 minutes — BTC higher?")
        if not m:
            st.caption("No open KXBTC15M market right now (between windows).")
            return
        up_p = m.get("up_prob_pct")
        strike = m.get("floor_strike")
        gap = m.get("strike_vs_spot")
        closes = _fmt_close(m.get("close_time"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Contract", f"BTC ≥ ${strike:,.0f}" if strike else "—",
                  f"{gap:+,.0f} vs spot" if gap is not None else None)
        c2.metric("Market: BTC up", f"{up_p:.0f}%" if up_p is not None else "—",
                  help="YES implied probability that BTC is at/above the level at "
                       "the window close — i.e. the market's 'up' odds.")
        c3.metric("Closes in", closes)

        # --- Expected value vs Kalshi's price (#8, #10) ---
        # If the scanner has No Trade, there's no directional read to price.
        if not sig.get("tradeable"):
            st.caption("⚪ Scanner says **No Trade** — no edge to price against the "
                       "market this window.")
            return
        from scan.kalshi_signal import evaluate_ev

        ev = evaluate_ev(sig["direction"], sig.get("win_probability"), up_p)
        if not ev:
            st.caption("Market price unavailable — can't compute EV.")
            return
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Our win prob", f"{ev['win_prob_pct']:.0f}%")
        e2.metric(f"Kalshi {ev['side']} price", f"{ev['entry_price_pct']:.0f}%")
        e3.metric("Edge (EV)", f"{ev['edge_pts']:+.0f} pts", f"{ev['ev_return_pct']:+.0f}% /$")
        e4.metric("Call", "BUY" if ev["recommend"] else "PASS")
        if ev["recommend"]:
            st.success(
                f"✅ **BUY {ev['side']}** — our {ev['win_prob_pct']:.0f}% win prob "
                f"beats the {ev['entry_price_pct']:.0f}% you'd pay (edge "
                f"{ev['edge_pts']:+.0f} pts). Size: {sig.get('position_size')}."
            )
        else:
            st.info(
                f"➖ **PASS** — {'the market already prices this' if ev['edge_pts'] < 4 else 'thin edge'} "
                f"(our {ev['win_prob_pct']:.0f}% vs {ev['entry_price_pct']:.0f}% price, "
                f"edge {ev['edge_pts']:+.0f} pts). Directionally {sig['direction']}, "
                f"but not worth it at this price."
            )
        st.caption("EV = our win probability − the price you pay. Educational only "
                   "— not financial advice.")
    except Exception:
        pass


def _fmt_close(ct) -> str:
    try:
        from datetime import datetime, timezone

        mins = (ct - datetime.now(timezone.utc)).total_seconds() / 60.0
        if mins < 60:
            return f"{mins:.0f}m"
        if mins < 60 * 24:
            return f"{mins / 60:.1f}h"
        return f"{mins / 1440:.0f}d"
    except Exception:
        return "—"


def _render_kalshi_markets(sig: Dict[str, Any]) -> None:
    """Live open Kalshi BTC markets, paired with the scanner's directional read."""
    try:
        markets = _markets_cached()
        st.markdown("### 📈 Live Kalshi BTC markets")
        if not markets:
            st.caption("No open Kalshi BTC markets returned right now.")
            return

        up = sig["direction"] == "up"
        # 'above' contracts: YES is bullish. 'below': YES is bearish. So the
        # scanner's read maps to YES on 'above' when up, YES on 'below' when down.
        aligned_kind = "above" if up else "below"
        atm = None
        try:
            from data.kalshi_markets import nearest_the_money

            atm = nearest_the_money(markets, sig["price"])
        except Exception:
            atm = None
        if atm:
            label = atm.get("threshold") or atm.get("title")
            closes = _fmt_close(atm.get("close_time"))
            yesp = atm.get("yes_prob_pct")
            if atm.get("kind") == "range":
                st.caption(
                    f"BTC is currently in **{label}** — the market gives that "
                    f"bucket ~{yesp}% (closes in {closes}). Scanner read: "
                    f"**{sig['action']}**. For a clean up/down bet, see the "
                    f"15-minute panel above."
                )
            else:
                side = "YES" if atm.get("kind") == aligned_kind else "NO"
                st.caption(
                    f"Scanner read **{sig['action']}**. Nearest directional "
                    f"contract: *{label}* (YES ~{yesp}%, closes in {closes}). "
                    f"A **{sig['direction']}** read favors **{side}** on this "
                    f"**{atm.get('kind')}** contract."
                )

        # Show the contracts nearest the current price (most relevant), not a
        # far-OTM soonest-close slice — that's what made the table show $73k
        # buckets while spot was ~$64k.
        price = float(sig["price"])
        near = sorted(
            (m for m in markets if m.get("floor_strike") is not None),
            key=lambda m: abs(m["floor_strike"] - price),
        )[:12]
        near.sort(key=lambda m: m["floor_strike"], reverse=True)  # high→low ladder

        kind_icon = {"above": "▲ above", "below": "▼ below", "range": "◆ range"}
        rows = []
        for m in near:
            rows.append({
                "Contract": m.get("threshold") or m.get("title"),
                "Type": kind_icon.get(m.get("kind"), m.get("kind")),
                "Strike": f"${m['floor_strike']:,.0f}" if m.get("floor_strike") else "—",
                "YES %": f"{m['yes_prob_pct']:.0f}%" if m.get("yes_prob_pct") is not None else "—",
                "Spread": f"{m['spread_cents']:.0f}¢" if m.get("spread_cents") is not None else "—",
                "Closes in": _fmt_close(m.get("close_time")),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "YES % is the market's implied probability (mid of YES bid/ask). "
            "▲ above = YES bullish, ▼ below = YES bearish, ◆ range = a price bucket. "
            "Live from Kalshi's public API — read-only, not a recommendation."
        )
    except Exception:
        pass


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
