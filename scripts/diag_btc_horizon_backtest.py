"""Does BTC technical analysis predict direction at LONGER horizons?

The 15-min calibration showed TA features are noise intraday. This backtests the
same core features on ~300 daily BTC bars (Coinbase, multi-month/multi-regime)
against forward-return direction at 1/2/3/5/10-day horizons. If any feature
shows real correlation at some horizon, the scanner should be repointed there
(daily KXBTCD contracts); if none do, BTC direction isn't a TA edge at all.

Read-only, no DB — just Coinbase. Run via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZONS = (1, 2, 3, 5, 10)


def _pearson(xs, ys):
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main() -> None:
    import pandas as pd

    from data.crypto_btc import fetch_btc_bars
    from scan.indicators import atr, ema, macd, rsi

    df = fetch_btc_bars("1d")
    print("=" * 72)
    print("BTC daily TA backtest — do features predict forward direction?")
    print("=" * 72)
    if df is None or len(df) < 60:
        print(f"not enough daily bars ({0 if df is None else len(df)})")
        return
    close = df["Close"].astype(float)
    print(f"{len(df)} daily bars, {df.index[0].date()} → {df.index[-1].date()}")
    up_days = float((close.pct_change().dropna() > 0).mean())
    print(f"base rate: up day {up_days:.0%}\n")

    ema9, ema21 = ema(close, 9), ema(close, 21)
    _ml, _sig, hist = macd(close)
    feats = {
        "ema_sep_pct": (ema9 - ema21) / close * 100.0,
        "price_vs_ema9_pct": (close - ema9) / close * 100.0,
        "rsi_minus_50": rsi(close, 14) - 50.0,
        "macd_hist_x1k": hist / close * 1000.0,
        "ema9_slope_3": ema9.diff(3) / close * 100.0,
        "mom_5d_pct": close.pct_change(5) * 100.0,
        "mom_10d_pct": close.pct_change(10) * 100.0,
        "atr_pct": atr(df, 14) / close * 100.0,
    }

    # Correlation of each feature with forward-up (1/0) at each horizon.
    print("CORRELATION with forward-up direction (|r|>~0.15 = worth a look):")
    header = "  " + f"{'feature':<20}" + "".join(f"{f'{h}d':>8}" for h in HORIZONS)
    print(header)
    best = (None, None, 0.0)
    for name, series in feats.items():
        cells = []
        for h in HORIZONS:
            fwd_up = (close.shift(-h) / close - 1.0) > 0
            merged = pd.concat([series, fwd_up.astype(float)], axis=1).dropna()
            c = _pearson(merged.iloc[:, 0].tolist(), merged.iloc[:, 1].tolist())
            cells.append(c)
            if c is not None and abs(c) > abs(best[2]):
                best = (name, h, c)
        row = "".join((f"{c:+8.2f}" if c is not None else f"{'—':>8}") for c in cells)
        print(f"  {name:<20}{row}")

    print(f"\nstrongest signal: {best[0]} @ {best[1]}d  r={best[2]:+.2f}")

    # Practical read: EMA-trend-aligned hit rate vs base rate, per horizon.
    print("\nEMA9>EMA21 (uptrend) → forward-up hit rate vs base rate:")
    bull = ema9 > ema21
    for h in HORIZONS:
        fwd_up = (close.shift(-h) / close - 1.0) > 0
        m = pd.concat([bull, fwd_up], axis=1).dropna()
        b = m[m.iloc[:, 0]]
        if len(b) >= 10:
            hit = float(b.iloc[:, 1].mean())
            base = float(m.iloc[:, 1].mean())
            edge = hit - base
            print(f"  {h}d: {hit:.0%} up when bullish vs {base:.0%} base "
                  f"(edge {edge:+.0%}, n={len(b)})")

    print("\nReading: correlations near 0 across all horizons → TA doesn't predict "
          "BTC direction (pivot to monitoring-only or the stock scanner). A clear "
          "|r|>0.15 at some horizon → repoint the engine there.")


if __name__ == "__main__":
    main()
