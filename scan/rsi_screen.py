"""Daily RSI screens for custom scans. No prices fetched and no score changes."""
import math

import pandas as pd


def daily_rsi14(closes):
    """Wilder RSI, seeded with the mean of the first 14 changes; no short-history guesses."""
    values = pd.to_numeric(closes, errors="coerce")
    out = pd.Series(float("nan"), index=values.index, dtype=float)
    if len(values) < 15 or not all(math.isfinite(v) and v > 0 for v in values):
        return out
    changes = values.diff()
    gain = float(changes.iloc[1:15].clip(lower=0).mean())
    loss = float((-changes.iloc[1:15].clip(upper=0)).mean())
    for i in range(14, len(values)):
        if i > 14:
            delta = float(changes.iloc[i])
            gain = (gain * 13 + max(delta, 0)) / 14
            loss = (loss * 13 + max(-delta, 0)) / 14
        out.iloc[i] = 50.0 if gain == loss == 0 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    return out


def rsi_screen(df, *, min_rsi=None, max_rsi=None, rebound=False, relative_volume=None):
    """Inclusive bounds. Rebound: RSI <=30 in prior 5 bars, now (30,50], up close, RVOL >=1."""
    series = daily_rsi14(df["Close"])
    value = float(series.iloc[-1]) if len(series) else float("nan")
    value = value if math.isfinite(value) else None
    prior = series.iloc[-6:-1]
    candidate = bool(value is not None and 30 < value <= 50 and len(prior) == 5 and prior.notna().all()
                     and prior.min() <= 30 and df["Close"].iloc[-1] > df["Close"].iloc[-2]
                     and relative_volume is not None and math.isfinite(relative_volume) and relative_volume >= 1)
    matches = ((min_rsi is None and max_rsi is None) or
               (value is not None and (min_rsi is None or value >= min_rsi) and (max_rsi is None or value <= max_rsi)))
    return value, candidate, bool(matches and (not rebound or candidate))
