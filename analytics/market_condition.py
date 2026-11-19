"""Market condition from SPY's daily closes: Trending up, Choppy or Risk-off.

One transparent rule, used everywhere HSF shows a market condition (the Scanner,
Brief and stock-page badge, the public pages and the Track record split), so a
"choppy" day means the same thing in every view:

    sma20, sma50   simple moving averages of SPY's close
    slope20        sma20 now vs sma20 five sessions earlier
    vol20          annualised standard deviation of the last 20 daily returns

    Risk-off     close below both sma20 and sma50, OR vol20 >= HIGH_VOL
    Trending up  close above both, sma20 above sma50, sma20 rising, vol20 < HIGH_VOL
    Choppy       anything else (mixed signals: the trend isn't clear)

Point in time: the label for a day uses only closes up to and including that day.
Nothing here calls the network; callers pass the closes in.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

TRENDING_UP = "trending_up"
CHOPPY = "choppy"
RISK_OFF = "risk_off"
LABELS = {TRENDING_UP: "Trending up", CHOPPY: "Choppy", RISK_OFF: "Risk-off"}
HIGH_VOL = 25.0        # annualised %, roughly where SPY's daily swings stop being "normal"
MIN_CLOSES = 55        # 50 for sma50 plus 5 for the slope
RULE_VERSION = "spy-sma20-50-vol20-v1"

# What each condition means for someone reading a setup, shown beside the badge.
ADVICE = {
    TRENDING_UP: "The broad market is in a steady uptrend. Breakout and momentum setups have had the most room to work in this kind of market.",
    CHOPPY: "The market has no clear trend. Breakouts fail more often in choppy markets, so expect more false starts and consider smaller size or waiting for confirmation.",
    RISK_OFF: "The market is falling or swinging hard. Long setups fail more often here; stops get hit faster and gaps are bigger. Treat new long setups with extra caution.",
}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def classify(closes: Sequence[float]) -> Optional[Dict[str, Any]]:
    """The condition at the last close, or None with fewer than MIN_CLOSES usable closes."""
    c = [float(x) for x in closes if x is not None and math.isfinite(float(x)) and float(x) > 0]
    if len(c) < MIN_CLOSES:
        return None
    last = c[-1]
    sma20 = _mean(c[-20:])
    sma50 = _mean(c[-50:])
    sma20_prev = _mean(c[-25:-5])
    rets = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - 20, len(c))]
    m = _mean(rets)
    vol20 = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(252) * 100.0
    above20, above50 = last >= sma20, last >= sma50
    if (not above20 and not above50) or vol20 >= HIGH_VOL:
        key = RISK_OFF
    elif above20 and above50 and sma20 >= sma50 and sma20 >= sma20_prev:
        key = TRENDING_UP
    else:
        key = CHOPPY
    return {
        "condition": key,
        "label": LABELS[key],
        "advice": ADVICE[key],
        "spy_close": round(last, 2),
        "spy_vs_sma20_pct": round((last / sma20 - 1.0) * 100.0, 2),
        "spy_vs_sma50_pct": round((last / sma50 - 1.0) * 100.0, 2),
        "sma20_rising": sma20 >= sma20_prev,
        "volatility_20d_pct": round(vol20, 1),
        "rule": RULE_VERSION,
    }


def why(c: Dict[str, Any]) -> str:
    """One plain sentence with the numbers behind a classification."""
    side20 = "above" if c["spy_vs_sma20_pct"] >= 0 else "below"
    side50 = "above" if c["spy_vs_sma50_pct"] >= 0 else "below"
    return (f"S&P 500 (SPY) is {abs(c['spy_vs_sma20_pct']):.1f}% {side20} its 20-day average and "
            f"{abs(c['spy_vs_sma50_pct']):.1f}% {side50} its 50-day average, with 20-day volatility of "
            f"{c['volatility_20d_pct']:.0f}% a year.")


def by_day(series: Sequence[Tuple[str, float]]) -> Dict[str, str]:
    """{ISO day: condition key} for every day with enough history, from (day, close)
    pairs oldest first. Each day's label uses only that day's close and earlier ones."""
    out: Dict[str, str] = {}
    closes: List[float] = []
    for day, close in series:
        closes.append(float(close))
        if len(closes) >= MIN_CLOSES:
            c = classify(closes[-(MIN_CLOSES + 5):])
            if c:
                out[str(day)[:10]] = c["condition"]
    return out


def condition_on(day: str, labels: Dict[str, str], days: Optional[List[str]] = None) -> Optional[str]:
    """The condition a pick saved on `day` was made in: the label at the latest close
    BEFORE that day (an intraday scan can't know that day's close yet, so using it
    would look ahead). Pass `days` = sorted(labels) when calling in a loop."""
    import bisect

    days = days if days is not None else sorted(labels)
    i = bisect.bisect_left(days, str(day)[:10])
    return labels[days[i - 1]] if i > 0 else None
