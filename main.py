#!/usr/bin/env python3
"""
ai_scanner.py — Webull-style scanners in Python (Penny Runners, Breakout Momentum, Premarket Gappers)
- Reads tickers from universe.txt (one per line)
- Fetches recent data via yfinance
- Calculates RSI, MA9, MA21, Relative Volume
- Outputs ranked results to the console and CSVs

Usage:
  python ai_scanner.py --days 30 --penny --breakout --premarket
  python ai_scanner.py --penny
  python ai_scanner.py --breakout --min_price 5 --max_price 50

Notes:
- If you hit rate limits, split your universe or run with smaller batches.
- BRK.B on Webull == BRK-B in yfinance.
"""

import argparse
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------ Indicators ------------------------ #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def relative_volume(df: pd.DataFrame, lookback_days: int = 10) -> float:
    """
    Simple RVOL: today's volume / average volume of prior N days
    Assumes df is daily OHLCV with 'Volume'
    """
    if len(df) < lookback_days + 1:
        return np.nan
    today_vol = df["Volume"].iloc[-1]
    avg_prior = df["Volume"].iloc[-(lookback_days+1):-1].mean()
    return today_vol / avg_prior if avg_prior and avg_prior > 0 else np.nan

# ------------------------ Data ------------------------ #
def load_universe(path: str = "universe.txt") -> list[str]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Missing {path}. Create it with tickers in TXT or CSV format.")
        sys.exit(1)

    tickers = []
    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p)
            if "Ticker" not in df.columns:
                logger.error("CSV universe file must contain a 'Ticker' column.")
                sys.exit(1)
            tickers = df["Ticker"].dropna().astype(str).tolist()
        except Exception as e:
            logger.error(f"Failed to read universe CSV: {e}")
            sys.exit(1)
    else:
        # default TXT fallback
        tickers = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    # Map common Webull symbols to yfinance where needed
    mapping = {
        "BRK.B": "BRK-B",
        "BF.B": "BF-B",
    }
    return [mapping.get(t, t) for t in tickers]

def download_index_tickers(index: str) -> list[str]:
    import pandas as pd
    import requests

    urls = {
        "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "nasdaq100": "https://en.wikipedia.org/wiki/NASDAQ-100",
        "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    }
    url = urls.get(index)
    if not url:
        logger.error(f"Unknown index {index}")
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.5845.188 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html = response.text
        tables = pd.read_html(html)
        if index == "sp500":
            df = tables[0]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).str.upper().tolist()
        elif index == "nasdaq100":
            df = tables[4]
            tickers = df['Ticker'].str.upper().tolist()
        elif index == "dow30":
            df = tables[1]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).str.upper().tolist()
    except Exception as e:
        logger.error(f"Failed to fetch index table for {index}: {e}")
        return []

    tickers = [t for t in tickers if t.isalpha() or '-' in t]
    logger.info(f"Downloaded {len(tickers)} tickers from {index}")
    return tickers

def fetch_history(tickers: list[str], period_days: int = 60) -> dict[str, pd.DataFrame]:
    """
    Fetch daily data for all tickers. Returns dict[ticker] -> DataFrame
    """
    hist = {}
    # yfinance can take a space-delimited string, but batching helps avoid large calls
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                " ".join(batch),
                period=f"{period_days}d",
                interval="1d",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            # If multiple tickers, columns are multi-index
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    sub = data.xs(t, axis=1, level=1, drop_level=False).droplevel(1, axis=1)
                    if not sub.empty:
                        hist[t] = sub.dropna()
            else:
                # Single ticker fallback
                t = batch[0]
                if not data.empty:
                    hist[t] = data.dropna()
        except Exception as e:
            logger.warning(f"Batch fetch failed ({batch[0]}..): {e}")
            # small pause on failures to be kind to API
            time.sleep(1.0)
    return hist

def latest_close_pct_change(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return np.nan
    prev_close = df["Close"].iloc[-2]
    last_close = df["Close"].iloc[-1]
    if prev_close == 0 or math.isnan(prev_close) or math.isnan(last_close):
        return np.nan
    return (last_close / prev_close - 1.0) * 100.0

# ------------------------ Scanners ------------------------ #
def scan_penny_runners(hist: dict[str, pd.DataFrame],
                       min_price=0.5, max_price=5.0,
                       min_change_pct=5.0, min_rvol=1.5,
                       max_mktcap=None) -> pd.DataFrame:
    """
    Penny runners: sub-$5, big movers, high rvol
    """
    rows = []
    for t, df in hist.items():
        try:
            last = df.iloc[-1]
            price = float(last["Close"])
            if not (min_price <= price <= max_price):
                continue

            chg = latest_close_pct_change(df)
            if pd.isna(chg) or chg < min_change_pct:
                continue

            rvol = relative_volume(df, lookback_days=10)

            # MarketCap not available via yfinance ticker fast; optional skip or fetch per-ticker
            mcap = np.nan

            rows.append({
                "Ticker": t,
                "Price": round(price, 4),
                "%Chg_d": round(chg, 2),
                "RVOL10": round(rvol, 2) if not pd.isna(rvol) else np.nan,
                "Vol": int(df["Volume"].iloc[-1]),
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["%Chg_d", "RVOL10"], ascending=[False, False]).reset_index(drop=True)

def scan_breakout(hist: dict[str, pd.DataFrame],
                  min_price=5.0, max_price=50.0,
                  rsi_min=55, rsi_max=70,
                  require_ma9=True, require_ma21=True) -> pd.DataFrame:
    """
    Breakout: price between bands, RSI sweet spot, above MA9 & MA21
    Includes EMA9, EMA21, ATR14, historical % changes, trend labels, formatted volume.
    """
    rows = []
    for t, df in hist.items():
        if len(df) < 25:
            continue
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

        price = float(close.iloc[-1])
        if not (min_price <= price <= max_price):
            continue

        rsi_series = rsi(close, period=14)
        rsi_val = float(rsi_series.iloc[-1])
        if not (rsi_min <= rsi_val <= rsi_max):
            continue

        ma9 = sma(close, 9).iloc[-1]
        ma21 = sma(close, 21).iloc[-1]
        cond_ma9 = price > ma9 if not pd.isna(ma9) else False
        cond_ma21 = price > ma21 if not pd.isna(ma21) else False

        if require_ma9 and not cond_ma9:
            continue
        if require_ma21 and not cond_ma21:
            continue

        chg = latest_close_pct_change(df)
        rvol = relative_volume(df, 10)
        prev_close = close.iloc[-2] if len(close) >= 2 else np.nan
        chg_dollar = price - prev_close if not pd.isna(prev_close) else np.nan
        pct_above_ma9 = ((price / ma9 - 1) * 100) if not pd.isna(ma9) and ma9 != 0 else np.nan
        pct_above_ma21 = ((price / ma21 - 1) * 100) if not pd.isna(ma21) and ma21 != 0 else np.nan

        # EMA calculations
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]

        # ATR14
        atr14 = (high.rolling(14).max() - low.rolling(14).min()).iloc[-1]

        # Historical percent changes
        pct_5 = (price / close.shift(5).iloc[-1] - 1) * 100 if len(close) >= 6 else np.nan
        pct_10 = (price / close.shift(10).iloc[-1] - 1) * 100 if len(close) >= 11 else np.nan
        pct_20 = (price / close.shift(20).iloc[-1] - 1) * 100 if len(close) >= 21 else np.nan

        # Trend label
        if rsi_val >= 67:
            trend = "Strong Up"
        elif rsi_val >= 60:
            trend = "Up"
        else:
            trend = "Neutral"

        rows.append({
            "Ticker": t,
            "Price": round(price, 2),
            "RSI14": round(rsi_val, 2),
            "AboveMA9": bool(cond_ma9),
            "AboveMA21": bool(cond_ma21),
            "Chg_$": round(chg_dollar, 2) if not pd.isna(chg_dollar) else np.nan,
            "%Chg_d": round(chg, 2) if not pd.isna(chg) else np.nan,
            "PctAboveMA9": round(pct_above_ma9, 2) if not pd.isna(pct_above_ma9) else np.nan,
            "PctAboveMA21": round(pct_above_ma21, 2) if not pd.isna(pct_above_ma21) else np.nan,
            "EMA9": round(ema9, 2),
            "EMA21": round(ema21, 2),
            "ATR14": round(atr14, 2),
            "%Chg_5d": round(pct_5, 2) if not pd.isna(pct_5) else np.nan,
            "%Chg_10d": round(pct_10, 2) if not pd.isna(pct_10) else np.nan,
            "%Chg_20d": round(pct_20, 2) if not pd.isna(pct_20) else np.nan,
            "Trend": trend,
            "RVOL10": round(rvol, 2) if not pd.isna(rvol) else np.nan,
            "Vol": int(vol.iloc[-1]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Format Vol with commas
    out["Vol"] = out["Vol"].apply(lambda x: f"{x:,}")
    # Sort by %Chg_d descending, then RVOL10 descending
    out = out.sort_values(by=["%Chg_d", "RVOL10"], ascending=[False, False]).reset_index(drop=True)
    return out

def scan_premarket_gappers(tickers: list[str],
                           minutes=120,
                           min_gap_pct=5.0,
                           min_price=0.5,
                           max_price=30.0,
                           min_volume=100_000) -> pd.DataFrame:
    """
    Premarket gappers using 1m data (prepost=True). Limited by yfinance availability.
    """
    rows = []
    now_utc = datetime.now(timezone.utc)
    period = "1d"
    interval = "1m"

    # Small batches to be gentle
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, prepost=True, progress=False, threads=False)
            if df.empty:
                continue
            df = df.tz_localize("UTC", level=None, nonexistent="shift_forward", ambiguous="NaT", copy=False)
            # Approx premarket = last N minutes before 13:30 UTC (9:30 ET). We'll simply look back 'minutes'.
            cutoff = now_utc - timedelta(minutes=minutes)
            recent = df[df.index >= cutoff]
            if recent.empty:
                continue
            pm_first = recent["Close"].iloc[0]
            pm_last = recent["Close"].iloc[-1]
            price = pm_last
            if not (min_price <= price <= max_price):
                continue

            gap_pct = (pm_last / pm_first - 1.0) * 100.0 if pm_first else np.nan
            volume = int(recent["Volume"].sum())

            if (not pd.isna(gap_pct)) and gap_pct >= min_gap_pct and volume >= min_volume:
                rows.append({
                    "Ticker": t,
                    "PM_Price": round(float(pm_last), 4),
                    "PM_%Chg": round(float(gap_pct), 2),
                    "PM_Vol": volume,
                })
        except Exception as e:
            # yfinance minute data can be flaky per-ticker
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["PM_%Chg", "PM_Vol"], ascending=[False, False]).reset_index(drop=True)

# ------------------------ CLI ------------------------ #
def main():
    parser = argparse.ArgumentParser(description="AI-style stock scanners (Webull-like) in Python")
    parser.add_argument("--universe", default="universe.txt", help="path to tickers list")
    parser.add_argument("--days", type=int, default=60, help="history days for daily scans")
    parser.add_argument("--penny", action="store_true", help="run penny runners scan")
    parser.add_argument("--breakout", action="store_true", help="run breakout scan")
    parser.add_argument("--premarket", action="store_true", help="run premarket gapper scan (1m data)")
    parser.add_argument("--min_price", type=float, default=None, help="override min price")
    parser.add_argument("--max_price", type=float, default=None, help="override max price")
    parser.add_argument("--index", type=str, choices=["sp500","nasdaq100","dow30"], help="Download tickers from index instead of universe file")
    parser.add_argument("--save", action="store_true", help="Save downloaded tickers into universe file")
    parser.add_argument("--append", action="store_true", help="Append downloaded tickers instead of overwriting")
    args = parser.parse_args()

    tickers = []

    # Download from index or load from universe
    if args.index:
        logger.info(f"Downloading tickers for index: {args.index}")
        tickers = download_index_tickers(args.index)
        if not tickers:
            logger.error(f"No tickers downloaded for index {args.index}. Exiting.")
            sys.exit(1)

        if args.save:
            p = Path(args.universe)
            existing = []

            # Load existing tickers if appending
            if args.append and p.exists():
                try:
                    if p.suffix.lower() == ".csv":
                        df_existing = pd.read_csv(p)
                        if "Ticker" in df_existing.columns:
                            existing = df_existing["Ticker"].dropna().astype(str).tolist()
                    else:
                        existing = [line.strip() for line in p.read_text().splitlines() if line.strip()]
                except Exception as e:
                    logger.warning(f"Failed to read existing universe file for append: {e}")

            # Merge new tickers with existing and remove duplicates
            combined = list(dict.fromkeys(existing + tickers))

            # Save to TXT or CSV
            try:
                if p.suffix.lower() == ".csv":
                    df_save = pd.DataFrame({"Ticker": combined})
                    df_save.to_csv(p, index=False)
                else:
                    p.write_text("\n".join(combined) + "\n")
                logger.info(f"Saved {len(combined)} tickers to {p}")
            except Exception as e:
                logger.error(f"Failed to save tickers to {p}: {e}")
                sys.exit(1)
    else:
        tickers = load_universe(args.universe)

    # Final validation before scanning
    tickers = [t for t in tickers if t.isalpha() and 1 <= len(t) <= 5]

    # Fetch history for penny/breakout scans
    hist = {}
    if args.penny or args.breakout:
        logger.info(f"Downloading {args.days}d daily history for {len(tickers)} tickers...")
        hist = fetch_history(tickers, period_days=args.days)

    outputs = []

    if args.penny:
        min_p = args.min_price if args.min_price is not None else 0.5
        max_p = args.max_price if args.max_price is not None else 5.0
        df_penny = scan_penny_runners(hist, min_price=min_p, max_price=max_p)
        logger.info("=== Penny Runners ===")
        print(df_penny.head(50).to_string(index=False))
        outputs.append(("penny_runners.csv", df_penny))

    if args.breakout:
        min_p = args.min_price if args.min_price is not None else 5.0
        max_p = args.max_price if args.max_price is not None else 50.0
        df_break = scan_breakout(hist, min_price=min_p, max_price=max_p)
        logger.info("=== Breakout Momentum ===")
        print(df_break.head(50).to_string(index=False))
        outputs.append(("breakout_momentum.csv", df_break))

    if args.premarket:
        logger.info("Scanning premarket gappers (1m, last 120 min by default)...")
        df_pm = scan_premarket_gappers(tickers, minutes=120)
        logger.info("=== Premarket Gappers ===")
        print(df_pm.head(50).to_string(index=False))
        outputs.append(("premarket_gappers.csv", df_pm))

    # Save CSV outputs
    for name, df in outputs:
        if df is not None and not df.empty:
            df.to_csv(name, index=False)
            logger.info(f"Saved {name}")

if __name__ == "__main__":
    main()