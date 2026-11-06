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


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def indicator_evidence(df, *, relative_volume=None):
    """Versioned evidence from these bars only; no fetches or future observations."""
    closes = pd.to_numeric(df["Close"], errors="coerce")
    series = daily_rsi14(closes)
    value = _finite(series.iloc[-1]) if len(series) else None
    previous = _finite(series.iloc[-2]) if len(series) > 1 else None
    prior = series.iloc[-6:-1]
    complete_prior = len(prior) == 5 and prior.notna().all()
    low = _finite(prior.min()) if complete_prior else None
    low_date = str(prior.idxmin())[:10] if low is not None else None
    valid_closes = len(closes) > 0 and all(math.isfinite(v) and v > 0 for v in closes)
    close = _finite(closes.iloc[-1]) if valid_closes else None
    prev_close = _finite(closes.iloc[-2]) if valid_closes and len(closes) > 1 else None
    up_close = close is not None and prev_close is not None and close > prev_close
    volume = _finite(relative_volume)
    supported = up_close and volume is not None and volume >= 1
    sma50 = _finite(closes.iloc[-50:].mean()) if valid_closes and len(closes) >= 50 else None
    rebound = bool(value is not None and 30 < value <= 50 and low is not None and low <= 30 and supported)
    recovery = bool(value is not None and previous is not None and previous <= 50 < value and supported)
    pullback = bool(value is not None and previous is not None and 40 <= previous < value <= 50
                    and sma50 is not None and close > sma50 and supported)
    line = signal = histogram = prior_histogram = None
    if valid_closes and len(closes) >= 35:
        from scan.indicators import macd
        lines, signals, hist = macd(closes)
        line, signal = _finite(lines.iloc[-1]), _finite(signals.iloc[-1])
        histogram, prior_histogram = _finite(hist.iloc[-1]), _finite(hist.iloc[-2])
    bullish = histogram is not None and histogram > 0
    crossover = bullish and prior_histogram is not None and prior_histogram <= 0
    return {
        "rule_version": "daily-indicators-v1", "bar_date": str(df.index[-1])[:10] if len(df) else None,
        "rsi_14": value, "rsi_previous": previous, "prior_5_rsi_low": low, "prior_5_rsi_low_date": low_date,
        "close": close, "previous_close": prev_close,
        "change_pct": (close / prev_close - 1) * 100 if close is not None and prev_close else None,
        "relative_volume": volume, "sma50": sma50,
        "rebound_candidate": rebound, "momentum_recovery": recovery, "uptrend_pullback": pullback,
        "macd_line": line, "macd_signal": signal, "macd_histogram": histogram,
        "macd_previous_histogram": prior_histogram, "macd_bullish_cross": bool(crossover),
        "macd_above_signal": bool(bullish),
    }


def evaluate_screen(df, *, min_rsi=None, max_rsi=None, rebound=False, momentum_recovery=False,
                    uptrend_pullback=False, macd_confirmation=None, relative_volume=None):
    evidence = indicator_evidence(df, relative_volume=relative_volume)
    value = evidence["rsi_14"]
    matches = ((min_rsi is None and max_rsi is None) or
               (value is not None and (min_rsi is None or value >= min_rsi) and (max_rsi is None or value <= max_rsi)))
    for selected, key in ((rebound, "rebound_candidate"), (momentum_recovery, "momentum_recovery"),
                          (uptrend_pullback, "uptrend_pullback")):
        if selected:
            matches = matches and evidence[key]
    if macd_confirmation:
        key = {"bullish_cross": "macd_bullish_cross", "above_signal": "macd_above_signal"}.get(macd_confirmation)
        matches = matches and bool(key and evidence[key])
    return evidence, bool(matches)


def rsi_screen(df, *, min_rsi=None, max_rsi=None, rebound=False, relative_volume=None):
    """Backward-compatible RSI screen interface."""
    evidence, matches = evaluate_screen(df, min_rsi=min_rsi, max_rsi=max_rsi, rebound=rebound,
                                       relative_volume=relative_volume)
    return evidence["rsi_14"], evidence["rebound_candidate"], matches
