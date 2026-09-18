"""Run 33 — Day Trader DT Score validation building blocks (pure, no I/O, no ML).

Deterministic, unit-testable helpers for a *direction-aware* backtest of the
Run 32 DT Score. They operate on bar arrays / observation dicts already provided
by the caller — they never fetch data and never run on the Streamlit path.

Lookahead discipline (Run 33 §22): the Run 32 features for a signal are computed
by the caller using ONLY bars at or before the signal timestamp T; the functions
here use future bars EXCLUSIVELY to measure outcomes (returns / MFE / MAE), never
to (re)compute a feature.

DT Score measures setup strength/coherence, NOT bullish probability, so success
is measured direction-aware: for a bearish signal a negative future return is
favorable follow-through (see `directional_return`).
"""
from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence

# Score buckets (Run 33 §7). Half-open [lo, hi).
SCORE_BUCKETS = [(0, 40, "0-39"), (40, 60, "40-59"), (60, 70, "60-69"),
                 (70, 80, "70-79"), (80, 90, "80-89"), (90, 101, "90-100")]


def directional_return(direction: str, future_return: Optional[float]) -> Optional[float]:
    """Sign the raw future return by the signal's direction so bullish and
    bearish setups are scored on one scale. Neutral is NOT a directional signal
    (returns None → excluded from hit rate / directional averages)."""
    if future_return is None:
        return None
    d = str(direction or "").lower()
    if d == "bullish":
        return future_return
    if d == "bearish":
        return -future_return
    return None  # neutral: no directional claim


def forward_return(prices: Sequence[float], t_index: int, horizon_bars: int) -> Optional[float]:
    """Percent return from price[t] to price[t+horizon]. None when the future bar
    is missing (end of session / insufficient history) — never fabricated."""
    if not prices or t_index < 0 or t_index >= len(prices):
        return None
    j = t_index + horizon_bars
    if j >= len(prices) or j < 0:
        return None
    p0, p1 = prices[t_index], prices[j]
    if p0 in (None, 0) or p1 is None:
        return None
    return (p1 - p0) / p0


def forward_returns(prices: Sequence[float], t_index: int,
                    horizons_bars: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Named forward returns (e.g. {'5m':5,'15m':15}) from t. Missing bars → None."""
    return {name: forward_return(prices, t_index, h) for name, h in horizons_bars.items()}


def mfe_mae(prices: Sequence[float], t_index: int, window_bars: int,
            direction: str) -> Dict[str, Optional[float]]:
    """Direction-aware Maximum Favorable / Adverse Excursion over (t, t+window].

    For a bullish signal MFE is the largest positive move and MAE the largest
    drawdown; for bearish the excursions are measured on the SHORT side (a fall is
    favorable). Neutral → both None. Uses only future bars, for OUTCOME only."""
    d = str(direction or "").lower()
    if d not in ("bullish", "bearish") or not prices or t_index >= len(prices):
        return {"mfe": None, "mae": None}
    p0 = prices[t_index]
    if not p0:
        return {"mfe": None, "mae": None}
    end = min(len(prices) - 1, t_index + window_bars)
    if end <= t_index:
        return {"mfe": None, "mae": None}
    rets = [(prices[j] - p0) / p0 for j in range(t_index + 1, end + 1) if prices[j] is not None]
    if not rets:
        return {"mfe": None, "mae": None}
    signed = rets if d == "bullish" else [-r for r in rets]
    return {"mfe": max(max(signed), 0.0), "mae": min(min(signed), 0.0)}


def assign_score_bucket(score: Optional[float]) -> Optional[str]:
    """Score → bucket label. None/insufficient → None (excluded from buckets)."""
    if score is None:
        return None
    for lo, hi, label in SCORE_BUCKETS:
        if lo <= score < hi:
            return label
    return None


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def bucket_report(observations: Sequence[Dict[str, Any]], *,
                  horizons: Sequence[str] = ("5m", "15m", "30m", "60m"),
                  min_n: int = 10) -> List[Dict[str, Any]]:
    """Aggregate observations by DT Score bucket. Each observation dict carries
    'direction', 'score', per-horizon 'directional_return_<h>', 'mfe', 'mae'.

    Hit rate / directional averages use ONLY directional (non-neutral) rows.
    Sample sizes are always reported; rates read INSUFFICIENT_SAMPLE below min_n."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for o in observations:
        b = assign_score_bucket(o.get("score"))
        if b is not None:
            grouped.setdefault(b, []).append(o)

    order = {label: i for i, (_, _, label) in enumerate(SCORE_BUCKETS)}
    out = []
    for b in sorted(grouped, key=lambda x: order.get(x, 99)):
        rows = grouped[b]
        dirs = [str(o.get("direction", "")).lower() for o in rows]
        directional = [o for o in rows if str(o.get("direction", "")).lower() in ("bullish", "bearish")]
        entry: Dict[str, Any] = {
            "bucket": b, "n": len(rows),
            "bullish": dirs.count("bullish"), "bearish": dirs.count("bearish"),
            "neutral": dirs.count("neutral"),
            "sufficient": len(directional) >= int(min_n),
        }
        for h in horizons:
            vals = [o.get(f"directional_return_{h}") for o in directional
                    if o.get(f"directional_return_{h}") is not None]
            entry[f"avg_dir_return_{h}"] = mean(vals) if vals else None
            entry[f"median_dir_return_{h}"] = median(vals) if vals else None
            entry[f"hit_rate_{h}"] = _rate(sum(1 for v in vals if v > 0), len(vals)) if vals else None
            entry[f"n_{h}"] = len(vals)
        mfes = [o.get("mfe") for o in directional if o.get("mfe") is not None]
        maes = [o.get("mae") for o in directional if o.get("mae") is not None]
        entry["avg_mfe"] = mean(mfes) if mfes else None
        entry["avg_mae"] = mean(maes) if maes else None
        out.append(entry)
    return out


def score_distribution(scores: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Distribution stats for DT Scores (Run 33 §13) to detect bunching / ceiling
    / floor effects. None scores are excluded and counted separately."""
    vals = sorted(s for s in scores if s is not None)
    n = len(vals)
    result: Dict[str, Any] = {"n": n, "n_insufficient": sum(1 for s in scores if s is None)}
    if not vals:
        return result

    def pct(p: float) -> float:
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return vals[int(k)]
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    result.update({
        "mean": mean(vals), "median": median(vals),
        "std": pstdev(vals) if n > 1 else 0.0,
        "p10": pct(0.10), "p25": pct(0.25), "p50": pct(0.50),
        "p75": pct(0.75), "p90": pct(0.90), "p95": pct(0.95),
        "min": vals[0], "max": vals[-1],
    })
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (for factor / ablation ranking-quality checks).
    None when fewer than 2 usable pairs or no variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)
