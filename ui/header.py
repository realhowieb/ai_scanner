from __future__ import annotations

from datetime import date

import streamlit as st

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

# Optional (Alpaca / custom) quote provider; safe to be missing
try:
    from market_data import get_latest_quotes  # type: ignore
except ImportError:  # pragma: no cover
    get_latest_quotes = None


HEADER_PROVIDER_ERRORS = (RuntimeError, TimeoutError, ConnectionError, OSError, TypeError, ValueError)


def render_header() -> None:
    """Render a compact, left-aligned MarketPulse AI header logo."""

    # Logo column + spacer (prevents full-width stretch)
    logo_col, _ = st.columns([1.2, 4])

    with logo_col:
        st.markdown(
            "<div style='padding-top:0.5rem; padding-bottom:0.5rem;'>",
            unsafe_allow_html=True,
        )
        st.image(
            "assets/market_ai_logo_tighter.png",
            width="content",
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Price ticker strip ----------------
TICKER_STRIP = ["SPY", "QQQ", "IWM", "DIA", "VIX", "AAPL", "MSFT", "NVDA", "TSLA"]


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_ticker_quotes(symbols: list[str]) -> list[dict[str, float]]:
    """Fetch {symbol,last,change_pct} for a list of tickers.

    Uses the centralized quote provider, which is Alpaca/custom-provider backed
    and cached. Header rendering must never call yfinance directly.
    """
    if not symbols:
        return []

    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    # de-dupe while preserving order
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    results: list[dict[str, float]] = []

    # Provider (Alpaca/custom)
    quotes = {}
    if callable(get_latest_quotes):
        try:
            quotes = get_latest_quotes(symbols) or {}
        except HEADER_PROVIDER_ERRORS:
            quotes = {}

    if quotes:
        for sym in symbols:
            q = quotes.get(sym)
            if not isinstance(q, dict):
                continue
            last = q.get("last")
            prev = q.get("prev_close")
            try:
                if last is None:
                    continue
                last_f = float(last)
                prev_f = float(prev) if prev is not None else last_f
                change_pct = 0.0 if not prev_f else ((last_f - prev_f) / prev_f) * 100.0
                results.append({"symbol": sym, "last": last_f, "change_pct": change_pct})
            except HEADER_PROVIDER_ERRORS:
                continue

    return results


def render_price_ticker(symbols: list[str] | None = None) -> None:
    """Render the scrolling ticker strip."""
    data = _fetch_ticker_quotes(symbols or TICKER_STRIP)
    if not data:
        return

    items_html = ""
    for row in data:
        sym = row["symbol"]
        last = row["last"]
        cpct = row["change_pct"]
        color = "#2ecc71" if cpct >= 0 else "#e74c3c"
        items_html += (
            f"<span class='ticker__item'>"
            f"{sym}&nbsp;{last:.2f}&nbsp;"
            f"<span style='color:{color};'>{cpct:+.2f}%</span>"
            f"</span>"
        )

    ticker_html = f"""
    <style>
    .ticker-wrap {{
        width: 100%;
        overflow: hidden;
        background: #0f1117;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
        padding: 0;
        margin-top: 0;
        margin-bottom: 0;
        line-height: 1.0;
    }}
    .ticker {{
        display: inline-block;
        white-space: nowrap;
        animation: ticker-move 35s linear infinite;
        font-size: 14px;
        will-change: transform;
    }}
    .ticker__item {{
        display: inline-block;
        margin-right: 40px;
        color: #e0e0e0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    @keyframes ticker-move {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-50%, 0, 0); }}
    }}
    </style>
    <div class="ticker-wrap">
      <div class="ticker">
        {items_html} {items_html}
      </div>
    </div>
    """

    st.markdown(
        f"<div style='margin:0;padding:0;'>{ticker_html}</div>",
        unsafe_allow_html=True,
    )


# ---------------- Market snapshot ----------------
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_index_snapshot(symbol: str) -> tuple[float | None, float | None]:
    """Return (last, prev_close) for an index proxy symbol like SPY/QQQ."""
    symbol = str(symbol).strip().upper()

    quotes = {}
    if callable(get_latest_quotes):
        try:
            quotes = get_latest_quotes([symbol]) or {}
        except HEADER_PROVIDER_ERRORS:
            quotes = {}

    if quotes:
        q = quotes.get(symbol)
        if isinstance(q, dict):
            last = q.get("last")
            prev = q.get("prev_close")
            try:
                if last is not None:
                    last_f = float(last)
                    prev_f = float(prev) if prev is not None else last_f
                    return last_f, prev_f
            except HEADER_PROVIDER_ERRORS:
                pass

    return None, None


def render_market_snapshot(results_df=None) -> None:
    """Render a lightweight market snapshot.

    Pass `results_df` (scan results) to enable Top Gainer / Most Active.
    """
    st.markdown("### 🔎 Today's Market Snapshot")

    df = results_df

    # If caller didn't pass results_df, try to pull it from session_state (set by app after scan)
    if df is None:
        try:
            df = st.session_state.get("latest_results_df")
        except Exception:
            df = None

    try:
        if df is not None and getattr(df, "empty", False) is True:
            df = None
    except Exception:
        df = None

    c1, c2, c3, c4 = st.columns(4)

    def _render_index(col, label: str, symbol: str) -> None:
        with col:
            last, prev = _fetch_index_snapshot(symbol)
            if last is None or prev is None or not prev:
                st.metric(label, "—", "—")
                return
            pct = ((last - prev) / prev) * 100.0
            st.metric(label, f"{last:.2f}", f"{pct:+.2f}%")

    _render_index(c1, "S&P 500 (SPY)", "SPY")
    _render_index(c2, "NASDAQ 100 (QQQ)", "QQQ")

    with c3:
        try:
            if df is None or pd is None:
                st.metric("Top Gainer", "—", "—")
            else:
                # Pick a best-effort numeric change column
                lower_map = {col: str(col).lower() for col in df.columns}
                metric_col = None

                # Prefer explicit percent-change style columns
                pct_keys = [
                    "pct_change",
                    "pct change",
                    "change%",
                    "change %",
                    "% change",
                    "%chg",
                    "chg%",
                    "return",
                    "daily_change",
                ]
                for col, lower in lower_map.items():
                    if any(k in lower for k in pct_keys):
                        metric_col = col
                        break

                # Fallback: generic change columns
                if metric_col is None:
                    for col, lower in lower_map.items():
                        if any(k in lower for k in ["chg", "change"]):
                            metric_col = col
                            break

                # Last fallback: gap columns
                if metric_col is None:
                    for col, lower in lower_map.items():
                        if "gap" in lower:
                            metric_col = col
                            break

                if metric_col is None:
                    st.metric("Top Gainer", "—", "—")
                else:
                    numeric_series = pd.to_numeric(df[metric_col], errors="coerce")
                    idx = numeric_series.idxmax()
                    if pd.isna(numeric_series.loc[idx]):
                        st.metric("Top Gainer", "—", "—")
                    else:
                        top = df.loc[idx]
                        ticker = None
                        if hasattr(top, "get"):
                            for k in ["Ticker", "Symbol", "symbol", "ticker"]:
                                v = top.get(k)
                                if v:
                                    ticker = v
                                    break
                        ticker = ticker or "—"
                        raw_val = float(numeric_series.loc[idx])
                        st.metric("Top Gainer", ticker, f"{raw_val:+.2f}%")
        except Exception:
            st.metric("Top Gainer", "—", "—")

    with c4:
        try:
            if df is None or pd is None:
                st.metric("Most Active", "—", "—")
            else:
                vol_col = None

                # Prefer dollar volume / liquidity style columns
                vol_candidates = [
                    "DollarVol20",
                    "DollarVol",
                    "DollarVol20D",
                    "DollarVol_20",
                    "DollarVolume",
                    "AvgDollarVol",
                    "AvgDollarVol20",
                    "MinDollarVol",
                ]
                for c in vol_candidates:
                    if c in df.columns:
                        vol_col = c
                        break

                # Fallback to share volume columns
                if not vol_col:
                    vol_candidates2 = ["Volume", "Vol", "AvgVol", "AvgVol20", "Vol20", "Volume20"]
                    for c in vol_candidates2:
                        if c in df.columns:
                            vol_col = c
                            break

                if not vol_col:
                    st.metric("Most Active", "—", "—")
                else:
                    numeric_vol = pd.to_numeric(df[vol_col], errors="coerce")
                    idx = numeric_vol.idxmax()
                    if pd.isna(numeric_vol.loc[idx]):
                        st.metric("Most Active", "—", "—")
                    else:
                        row = df.loc[idx]
                        ticker = None
                        if hasattr(row, "get"):
                            for k in ["Ticker", "Symbol", "symbol", "ticker"]:
                                v = row.get(k)
                                if v:
                                    ticker = v
                                    break
                        ticker = ticker or "—"
                        val_millions = float(numeric_vol.loc[idx]) / 1_000_000
                        suffix = "M" if vol_col in vol_candidates else "M sh"
                        st.metric("Most Active", ticker, f"{val_millions:.1f}{suffix}")
        except Exception:
            st.metric("Most Active", "—", "—")
