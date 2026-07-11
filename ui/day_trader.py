"""⚡ Day Trader live monitor: gappers, VWAP, RVOL, today's move.

A self-contained intraday panel built on Alpaca snapshots (one cached call per
refresh — no heavy bar downloads). It answers the questions a day trader has at
market open: what's gapping, what's in play (RVOL), and is price above/below
VWAP. Auto-refreshes on an interval so the app becomes a monitor, not a report.

Degrades gracefully: hides itself when disabled or Alpaca is unconfigured, and
never raises into the main app.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st


def _default_symbols(watch_tickers: List[str] | None) -> str:
    if watch_tickers:
        return ", ".join(dict.fromkeys(t.upper() for t in watch_tickers if str(t).strip()))
    # A liquid, high-beta default so the panel is useful before a watchlist exists.
    return "AAPL, TSLA, NVDA, AMD, SPY, QQQ, META, AMZN, MSFT, GOOGL"


def _parse_symbols(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in (raw or "").replace("\n", ",").split(",")]
    return [p for p in dict.fromkeys(parts) if p]


def render_day_trader_panel(
    watch_tickers: List[str] | None = None,
    *,
    max_symbols: int = 60,
) -> None:
    """Render the live day-trader monitor (gappers / VWAP / RVOL)."""
    try:
        from config import DAY_TRADER_ENABLED
    except Exception:
        return
    if not DAY_TRADER_ENABLED:
        return

    try:
        from market_data import build_day_trader_metrics
    except Exception:
        return

    st.markdown("## ⚡ Day Trader — live")
    st.caption(
        "Intraday snapshot: today's move, gap, VWAP, and relative volume (RVOL). "
        "Auto-refreshes so you can keep it open at the open."
    )

    # --- Controls row ---
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        raw = st.text_input(
            "Symbols",
            value=_default_symbols(watch_tickers),
            key="dt_symbols",
            help="Comma-separated. Defaults to your watchlist.",
        )
    with c2:
        refresh_label = st.selectbox(
            "Auto-refresh",
            ["Off", "15s", "30s", "60s"],
            index=2,
            key="dt_refresh",
        )
    with c3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh now", key="dt_refresh_btn"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    # --- Auto-refresh (best-effort; requires streamlit-autorefresh) ---
    interval_ms = {"Off": 0, "15s": 15000, "30s": 30000, "60s": 60000}.get(refresh_label, 0)
    if interval_ms:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=interval_ms, key="dt_autorefresh")
        except Exception:
            st.caption("Auto-refresh unavailable; use 🔄 Refresh now.")

    symbols = _parse_symbols(raw)
    if not symbols:
        st.info("Enter one or more symbols to monitor.")
        return
    if len(symbols) > max_symbols:
        st.caption(f"Showing the first {max_symbols} of {len(symbols)} symbols.")
        symbols = symbols[:max_symbols]

    try:
        rows = build_day_trader_metrics(symbols)
    except Exception as e:
        st.caption("Live data is temporarily unavailable.")
        with st.expander("Details", expanded=False):
            st.code(f"{type(e).__name__}: {e}")
        return

    if not rows:
        st.caption(
            "No live data (market closed, Alpaca not configured, or symbols not found)."
        )
        return

    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "ticker": "Ticker",
            "last": "Last",
            "chg_pct": "Chg %",
            "gap_pct": "Gap %",
            "vwap": "VWAP",
            "vs_vwap_pct": "vs VWAP %",
            "volume": "Volume",
            "rvol": "RVOL",
        }
    )
    df["vs VWAP"] = df["vs VWAP %"].apply(
        lambda v: "▲ above" if (v is not None and v >= 0) else ("▼ below" if v is not None else "—")
    )

    ordered = ["Ticker", "Last", "Chg %", "Gap %", "VWAP", "vs VWAP", "vs VWAP %", "RVOL", "Volume"]
    df = df[[c for c in ordered if c in df.columns]]

    def _style(frame):
        def color_pct(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return ""
            return "color: #16a34a" if val >= 0 else "color: #dc2626"

        styler = frame.style
        for col in ("Chg %", "Gap %", "vs VWAP %"):
            if col in frame.columns:
                styler = styler.map(color_pct, subset=[col])
        return styler

    try:
        st.dataframe(_style(df), hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, width="stretch")

    now = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
    st.caption(
        f"As of {now} · Chg %/Gap % vs prior close · RVOL = today's volume ÷ 20-day avg · "
        "IEX feed — verify before trading."
    )
