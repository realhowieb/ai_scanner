import math
from typing import Dict, Iterable, Optional


import pandas as pd

# Optional yfinance for ultra-fast batch downloads (single call)
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

# Optional TA helpers. If not present, TA columns will be skipped even when include_ta=True
try:
    from ai_scanner.scan.indicators import ema, rsi  # type: ignore
except Exception:  # pragma: no cover
    ema = None  # type: ignore
    rsi = None  # type: ignore


def _classify_breakout(pct: float) -> str:
    """Bucket a breakout magnitude into readable classes.

    Parameters
    ----------
    pct : float
        Breakout percentage (e.g., 3.2 == +3.2%).

    Returns
    -------
    str
        One of: "Weak", "Moderate", "Strong", "Explosive".
    """
    if pct is None or (isinstance(pct, float) and (math.isnan(pct) or math.isinf(pct))):
        return "Weak"
    if pct >= 8:
        return "Explosive"
    if pct >= 5:
        return "Strong"
    if pct >= 2:
        return "Moderate"
    return "Weak"


def breakout_scanner(
    price_data: Dict[str, pd.DataFrame],
    min_price: float,
    max_price: float,
    include_ta: bool = False,
    spy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Scan price series for 20-day high breakouts.

    A symbol is considered a breakout when its latest close is above the max of the
    previous 20 closes (excluding the most recent bar). The function returns a
    DataFrame of matches with useful metrics.

    Parameters
    ----------
    price_data : dict[str, DataFrame]
        Mapping of ticker -> OHLCV DataFrame (must contain Close; Volume optional).
    min_price, max_price : float
        Filter symbols whose latest close is outside this range.
    include_ta : bool, default False
        If True and TA helpers are available, append RSI(14) and EMA(20).
    spy_df : DataFrame, optional
        If provided, compute RS_20d_vs_SPY (% outperformance over last 20 bars).

    Returns
    -------
    DataFrame
        Columns: [Ticker, Price, Breakout %, Prev 20d Max, Volume, Class,
                   RS 20d vs SPY (%), RSI14, EMA20]
    """
    rows: list[dict] = []

    # Pre-compute SPY 20d change if requested and available
    spy_20d_pct = None
    if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns and len(spy_df) >= 21:
        spy_close_now = float(pd.Series(spy_df["Close"]).iloc[-1])
        spy_close_20 = float(pd.Series(spy_df["Close"]).iloc[-21])
        if spy_close_20 != 0:
            spy_20d_pct = (spy_close_now / spy_close_20 - 1.0) * 100.0

    for ticker, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        closes = pd.Series(df["Close"]).dropna()
        if len(closes) < 21:  # need at least 21 bars to look back 20
            continue

        latest_close = float(closes.iloc[-1])
        if not (min_price <= latest_close <= max_price):
            continue

        prev_max = float(closes.iloc[-21:-1].max())
        if prev_max <= 0:
            continue

        breakout_pct = (latest_close / prev_max - 1.0) * 100.0
        if breakout_pct <= 0:  # not actually above prior 20d max
            continue

        vol_val = None
        if "Volume" in df.columns and len(df["Volume"]) == len(df["Close"]):
            try:
                vol_val = int(pd.Series(df["Volume"]).iloc[-1])
            except Exception:
                vol_val = None

        row = {
            "Ticker": ticker,
            "Price": round(latest_close, 4),
            "Prev 20d Max": round(prev_max, 4),
            "Breakout %": round(breakout_pct, 2),
            "Volume": vol_val,
            "Class": _classify_breakout(breakout_pct),
        }

        # Relative Strength vs SPY over 20 bars
        if spy_20d_pct is not None and len(closes) >= 21:
            sym_20d = float(closes.iloc[-1]) / float(closes.iloc[-21]) - 1.0
            row["RS 20d vs SPY (%)"] = round((sym_20d * 100.0) - spy_20d_pct, 2)

        # Optional TA
        if include_ta and ema is not None and rsi is not None:
            try:
                row["EMA20"] = float(ema(closes, 20).iloc[-1])
                row["RSI14"] = float(rsi(closes, 14).iloc[-1])
            except Exception:
                # If TA fails, skip adding
                pass

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "Ticker", "Price", "Prev 20d Max", "Breakout %", "Volume",
            "Class", "RS 20d vs SPY (%)", "EMA20", "RSI14",
        ])

    out = pd.DataFrame(rows)

    # Order columns if some are missing due to optional TA
    preferred_cols: Iterable[str] = (
        "Ticker", "Price", "Prev 20d Max", "Breakout %", "Volume",
        "Class", "RS 20d vs SPY (%)", "EMA20", "RSI14",
    )
    cols = [c for c in preferred_cols if c in out.columns] + [
        c for c in out.columns if c not in preferred_cols
    ]
    out = out[cols]

    # Sort by strongest breakout first, then price as tiebreaker
    if "Breakout %" in out.columns:
        out = out.sort_values(["Breakout %", "Price"], ascending=[False, False])

    out.reset_index(drop=True, inplace=True)
    return out


def run_sp500_scan(
    min_price: float = 1.0,
    max_price: float = 10_000.0,
    include_ta: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: load S&P 500 universe, fetch prices, and run breakout_scanner.

    Parameters
    ----------
    min_price, max_price : float
        Price filters applied to the latest close.
    include_ta : bool
        If True, append RSI14 and EMA20 when TA helpers are available.

    Returns
    -------
    DataFrame
        Breakout results for S&P 500.
    """
    try:
        from ai_scanner.data.universe import load_sp500_tickers  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import load_sp500_tickers") from e

    # Prefer parallel price fetcher if available
    price_data = {}
    try:
        from ai_scanner.data.prices import fetch_price_data_parallel  # type: ignore
        tickers = list(load_sp500_tickers())
        price_data, _skipped = fetch_price_data_parallel(tickers)
    except Exception:
        # Fallback: try batch fetcher
        try:
            from ai_scanner.data.prices import fetch_price_data_batch  # type: ignore
            tickers = list(load_sp500_tickers())
            price_data, _skipped = fetch_price_data_batch(tickers)
        except Exception as e2:
            raise RuntimeError("Could not fetch price data for SP500 universe") from e2

    spy_df = price_data.get("SPY")
    return breakout_scanner(
        price_data=price_data,
        min_price=min_price,
        max_price=max_price,
        include_ta=include_ta,
        spy_df=spy_df,
    )


# Backwards-compat alias (in case older code imports the misspelled name)
# NOTE: do not advertise this; keep for smooth migration only.
_breakout_scanner_typo_alias = breakout_scanner


# Streamlit-app-friendly wrapper for scanning a list of tickers
def run_breakout_scan(
    tickers: Iterable[str],
    min_price: float = 1.0,
    max_price: float = 10_000.0,
    include_ta: bool = False,
    spy_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> pd.DataFrame:
    """App-friendly breakout scan.

    The Streamlit app passes a list of tickers plus filter kwargs. This wrapper
    fetches price data for those tickers and then calls `breakout_scanner`.

    Speed optimization: if yfinance is available, we download all tickers in a
    single batched request, which is much faster than per-ticker calls.
    """
    tickers = [t for t in list(tickers) if t]
    if not tickers:
        return pd.DataFrame()

    # Ensure SPY is present for RS calculation
    tickers_plus_spy = tickers + (["SPY"] if "SPY" not in tickers else [])

    price_data: Dict[str, pd.DataFrame] = {}

    # 1) Fastest path: single yfinance batch download
    if yf is not None:
        try:
            batch = yf.download(
                " ".join(tickers_plus_spy),
                period="3mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if batch is not None and not batch.empty:
                # When multiple tickers, columns are MultiIndex: (Field, Ticker) or (Ticker, Field)
                if isinstance(batch.columns, pd.MultiIndex):
                    # Normalize to dict[ticker] -> OHLCV df
                    # Try both orientations
                    lvl0 = batch.columns.levels[0]
                    lvl1 = batch.columns.levels[1]
                    # If lvl0 looks like fields (Open/High/Low/Close/Adj Close/Volume)
                    field_names = {str(x).lower() for x in lvl0}
                    fields_like = {"open", "high", "low", "close", "adj close", "volume"}
                    if field_names & fields_like:
                        for t in tickers_plus_spy:
                            try:
                                sub = batch.xs(t, level=1, axis=1, drop_level=True)
                                if sub is not None and not sub.empty:
                                    price_data[t] = sub.dropna(how="all")
                            except Exception:
                                continue
                    else:
                        # Otherwise assume lvl0 is ticker
                        for t in tickers_plus_spy:
                            try:
                                sub = batch.xs(t, level=0, axis=1, drop_level=True)
                                if sub is not None and not sub.empty:
                                    price_data[t] = sub.dropna(how="all")
                            except Exception:
                                continue
                else:
                    # Single ticker download returns flat columns
                    # If only one ticker requested, assign it
                    if len(tickers_plus_spy) == 1:
                        price_data[tickers_plus_spy[0]] = batch.dropna(how="all")
        except Exception:
            price_data = {}

    # 2) Fallbacks if batch download failed
    if not price_data:
        try:
            from ai_scanner.data.prices import fetch_price_data_parallel  # type: ignore
            price_data, _skipped = fetch_price_data_parallel(tickers_plus_spy)
        except Exception:
            try:
                from ai_scanner.data.prices import fetch_price_data_batch  # type: ignore
                price_data, _skipped = fetch_price_data_batch(tickers_plus_spy)
            except Exception:
                price_data = {}

    if not price_data:
        return pd.DataFrame()

    # Choose SPY df for RS calc
    if spy_df is None:
        spy_df = price_data.get("SPY")

    return breakout_scanner(
        price_data=price_data,
        min_price=min_price,
        max_price=max_price,
        include_ta=include_ta,
        spy_df=spy_df,
    )