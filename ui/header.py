from __future__ import annotations

import streamlit as st

from datetime import date

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Optional (Alpaca / custom) quote provider; safe to be missing
try:
    from market_data import get_latest_quotes  # type: ignore
except Exception:  # pragma: no cover
    get_latest_quotes = None


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

    - Prefers `get_latest_quotes` if available.
    - Falls back to yfinance.
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
        except Exception:
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
            except Exception:
                continue

    if results:
        return results

    # yfinance fallback
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []

    hist = None
    try:
        hist = yf.download(
            symbols,
            period="2d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        hist = None

    if hist is not None and getattr(hist, "empty", True) is False:
        for sym in symbols:
            try:
                if pd is not None and isinstance(hist.columns, pd.MultiIndex):
                    if ("Close", sym) not in hist.columns:
                        continue
                    closes = hist[("Close", sym)].dropna()
                else:
                    if "Close" not in getattr(hist, "columns", []):
                        continue
                    closes = hist["Close"].dropna()

                if closes is None or len(closes) == 0:
                    continue

                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) > 1 else last
                change_pct = 0.0 if not prev else ((last - prev) / prev) * 100.0
                results.append({"symbol": sym, "last": last, "change_pct": change_pct})
            except Exception:
                continue

        if results:
            return results

    # slow fallback per ticker
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            h = t.history(period="2d")
            if h is None or getattr(h, "empty", True) is True or "Close" not in h.columns:
                continue
            closes = h["Close"].dropna().tolist()
            if not closes:
                continue
            last = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else last
            change_pct = 0.0 if not prev else ((last - prev) / prev) * 100.0
            results.append({"symbol": sym, "last": last, "change_pct": change_pct})
        except Exception:
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
        except Exception:
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
            except Exception:
                pass

    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None, None

    try:
        hist = yf.download(symbol, period="2d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        hist = None

    if hist is not None and getattr(hist, "empty", True) is False and "Close" in hist.columns:
        close_block = hist["Close"]
        if pd is not None and isinstance(close_block, pd.DataFrame):
            close_block = close_block.iloc[:, 0]
        closes = close_block.dropna().to_list()
        if closes:
            last = float(closes[-1])
            prev = float(closes[-2]) if len(closes) > 1 else last
            return last, prev

    try:
        t = yf.Ticker(symbol)
        h = t.history(period="2d")
        if h is None or getattr(h, "empty", True) is True or "Close" not in h.columns:
            return None, None
        closes = h["Close"].dropna().tolist()
        if not closes:
            return None, None
        last = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else last
        return last, prev
    except Exception:
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
