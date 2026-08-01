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

    @st.cache_data(ttl=60, show_spinner=False)
    def _outcome_stats_cached():
        from db.btc_outcomes import outcome_stats

        return outcome_stats()

    @st.cache_data(ttl=8, show_spinner=False)
    def _btc_stats_cached():
        from data.crypto_btc import btc_24h_stats

        return btc_24h_stats()

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

    def _outcome_stats_cached():
        from db.btc_outcomes import outcome_stats

        return outcome_stats()

    def _btc_stats_cached():
        from data.crypto_btc import btc_24h_stats

        return btc_24h_stats()


def _render_price_header() -> None:
    """Big live BTC-USD price + 24h change. Updates each auto-refresh tick."""
    try:
        s = _btc_stats_cached() or {}
        price = s.get("price")
        if price is None:
            from data.crypto_btc import latest_btc_price

            price = latest_btc_price()
        if price is None:
            return
        chg = s.get("change_pct")
        color = "#16a34a" if (chg is None or chg >= 0) else "#dc2626"
        arrow = "▲" if (chg is not None and chg >= 0) else "▼" if chg is not None else ""
        chg_txt = f"{arrow} {chg:+.2f}% 24h" if chg is not None else ""
        st.markdown(
            "<div style='display:flex;align-items:baseline;gap:14px;margin:2px 0 6px'>"
            "<span style='font-size:15px;color:#94a3b8'>₿ BTC-USD</span>"
            f"<span style='font-size:30px;font-weight:700'>${price:,.0f}</span>"
            f"<span style='font-size:16px;font-weight:600;color:{color}'>{chg_txt}</span>"
            "<span style='font-size:11px;color:#64748b'>● live</span></div>",
            unsafe_allow_html=True,
        )
        if s.get("high_24h") and s.get("low_24h"):
            st.caption(f"24h range ${s['low_24h']:,.0f} – ${s['high_24h']:,.0f}")
    except Exception:
        pass


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
            _render_price_header()
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

        # Slow-changing; render once (outside the auto-refresh fragment).
        _render_outcome_log()
    except Exception:
        pass


def _render_outcome_log() -> None:
    """Live readout of the 15-min outcome dataset + the engine's realized hit rate."""
    try:
        s = _outcome_stats_cached()
        if not s or not s.get("logged"):
            return
        acc = s.get("accuracy")
        acc_txt = f"{acc * 100:.0f}%" if acc is not None else "—"
        st.markdown("### 📊 Outcome log — model training data")
        m1, m2, m3 = st.columns(3)
        m1.metric("Windows logged", s["logged"])
        m2.metric("Settled", s["settled"])
        m3.metric("Engine correct",
                  f"{s['correct']}/{s['decided']}" if s["decided"] else "—",
                  acc_txt if s["decided"] else None)

        # Paper P&L — the honest scoreboard: does buying at Kalshi's price pay?
        bets = s.get("bets") or 0
        if bets:
            pnl = s.get("pnl") or 0.0
            roi = s.get("roi")
            roi_txt = f"{roi * 100:+.1f}%" if roi is not None else "—"
            p1, p2, p3 = st.columns(3)
            p1.metric("Paper trades", f"{s.get('bet_wins', 0)}/{bets} won")
            p2.metric("P&L", f"{pnl:+.2f} u", help="Units of $1 stake per trade.")
            p3.metric("ROI", roi_txt)
            st.caption(
                "Paper P&L assumes a $1 stake per BUY at Kalshi's price, settled "
                "$1/$0. This is the real test — accuracy ≠ profit against a sharp "
                "market. Educational only."
            )
        else:
            st.caption(
                "Every 15-min window's features + prediction, settled against "
                "Kalshi's result. Paper-P&L appears once the engine places its "
                "first BUY. Building toward a calibrated model — needs a few "
                "hundred settled windows."
            )
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
        f"Volume ≥ 0.5× {_mk(sig['volume_ok'])} · "
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
    """Live BTC candlestick with EMA9/21 + VWAP overlays and a volume subplot."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from scan.indicators import ema

        st.markdown(f"### 🟠 Live BTC — {timeframe} candles")
        d = df.tail(120)
        x = list(d.index)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.03,
        )
        # --- price candles ---
        fig.add_trace(
            go.Candlestick(
                x=x, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
                name="BTC", showlegend=False,
            ),
            row=1, col=1,
        )
        # --- EMA 9 / 21 + running VWAP overlays ---
        ema9 = ema(df["Close"], 9).tail(120)
        ema21 = ema(df["Close"], 21).tail(120)
        fig.add_trace(go.Scatter(x=x, y=ema9, line=dict(color="#60a5fa", width=1),
                                 name="EMA9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ema21, line=dict(color="#a78bfa", width=1),
                                 name="EMA21"), row=1, col=1)
        tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
        denom = d["Volume"].cumsum()
        vwap = (tp * d["Volume"]).cumsum() / denom.where(denom > 0)
        fig.add_trace(go.Scatter(x=x, y=vwap, line=dict(color="#f59e0b", width=1, dash="dot"),
                                 name="VWAP"), row=1, col=1)
        # --- plan lines (only when tradeable) ---
        if sig.get("tradeable") and sig.get("entry_zone"):
            for label, y, col in (
                ("target", sig["target"], "#16a34a"),
                ("entry", sig["entry_zone"][1], "#60a5fa"),
                ("stop", sig["stop"], "#dc2626"),
            ):
                fig.add_hline(y=y, line_color=col, line_dash="dot", line_width=1,
                              annotation_text=f"{label} {y:,.0f}",
                              annotation_font_color=col, annotation_font_size=10,
                              row=1, col=1)
        # --- volume, colored by candle direction ---
        vcolors = ["#16a34a" if c >= o else "#dc2626"
                   for o, c in zip(d["Open"], d["Close"])]
        fig.add_trace(go.Bar(x=x, y=d["Volume"], marker_color=vcolors, name="Vol",
                             showlegend=False), row=2, col=1)
        fig.update_layout(
            height=440, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.12)")
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch",
                        key=f"kalshi_chart_{timeframe}")
    except Exception:
        pass
