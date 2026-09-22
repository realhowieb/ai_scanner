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


# --- Feature coverage (parity/fidelity accounting) ---------------------------
# The seven Run 32 classifier inputs. Kept in sync with build_observation's
# diagnostic_inputs and analytics.day_trade_intel.
COVERAGE_FIELDS = ("chg_pct", "gap_pct", "rvol", "vs_vwap_pct",
                   "adx", "supertrend_direction", "ewo")
# Daily-derived inputs. When these are all missing the observation came from the
# intraday-only fallback (no daily-indicator reconstruction), NOT a full-feature
# row. Distinguishing the two is what the Run 33 harness bug obscured.
_DAILY_FIELDS = ("adx", "supertrend_direction", "ewo", "gap_pct", "rvol")


def _present(value: Any) -> bool:
    """A feature is 'present' when it is a real, usable value (not None/NaN)."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def feature_coverage(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-field presence + full-fidelity accounting for a set of observations.

    Reads each observation's ``diagnostic_inputs`` (the Run 32 inputs the score
    was actually computed from) and optional ``feature_source`` tag. Returns:
      * ``per_field``   — present count / pct for each of the 7 classifier inputs
      * ``full_feature`` — rows with ALL 7 inputs present (true full fidelity)
      * ``daily_missing`` — rows with EVERY daily-derived input missing
        (intraday-only fallback; the classifier renormalizes but cannot fire
        Strong, which needs ADX/RVOL confirmation)
      * ``by_source``   — counts grouped by the loader path that built the row
      * ``by_symbol``   — per-ticker full-feature / fallback share
    Pure: no I/O, no scoring, no future data. Read-only accounting.
    """
    n = len(observations)
    per_field = {f: 0 for f in COVERAGE_FIELDS}
    full = 0
    daily_missing = 0
    by_source: Dict[str, int] = {}
    by_symbol: Dict[str, Dict[str, int]] = {}

    for o in observations:
        inputs = o.get("diagnostic_inputs") or {}
        present = {f: _present(inputs.get(f)) for f in COVERAGE_FIELDS}
        for f, ok in present.items():
            if ok:
                per_field[f] += 1
        is_full = all(present.values())
        if is_full:
            full += 1
        if not any(present[f] for f in _DAILY_FIELDS):
            daily_missing += 1
        src = str(o.get("feature_source") or "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        sym = str(o.get("ticker") or "?")
        rec = by_symbol.setdefault(sym, {"n": 0, "full_feature": 0, "daily_missing": 0})
        rec["n"] += 1
        rec["full_feature"] += int(is_full)
        rec["daily_missing"] += int(not any(present[f] for f in _DAILY_FIELDS))

    def _pct(x: int) -> Optional[float]:
        return (x / n) if n else None

    return {
        "n": n,
        "per_field": {f: {"present": c, "pct": _pct(c)} for f, c in per_field.items()},
        "full_feature": {"count": full, "pct": _pct(full)},
        "daily_missing": {"count": daily_missing, "pct": _pct(daily_missing)},
        "by_source": by_source,
        "by_symbol": by_symbol,
    }


# --- Predictive validation (does DT Score rank forward outcomes?) -------------
def predictive_summary(observations: Sequence[Dict[str, Any]],
                       *, horizons: Sequence[str] = ("5m", "15m", "30m", "60m"),
                       ) -> Dict[str, Any]:
    """Does the DT Score predict *direction-aware* forward outcomes?

    For directional (non-neutral) observations only, per horizon:
      * ``spearman`` — rank correlation between DT Score and directional return
        (>0 means higher score → better follow-through; ~0 means no ranking edge).
      * ``top_minus_bottom`` — mean directional return of the top score tertile
        minus the bottom tertile (a simple long-the-strong edge in return units).
      * ``hit_top`` / ``hit_bottom`` — hit rate in each tertile.
      * ``n`` — usable pairs.
    Pure: uses future outcomes ONLY to measure, never to (re)compute a feature.
    """
    directional = [o for o in observations
                   if str(o.get("direction", "")).lower() in ("bullish", "bearish")
                   and o.get("score") is not None]
    out: Dict[str, Any] = {"directional_n": len(directional)}
    for h in horizons:
        pairs = [(float(o["score"]), float(o[f"directional_return_{h}"]))
                 for o in directional
                 if o.get(f"directional_return_{h}") is not None]
        n = len(pairs)
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if n >= 2 else None
        tb = ht = hb = None
        if n >= 6:
            pairs.sort(key=lambda p: p[0])
            k = n // 3
            bottom, top = pairs[:k], pairs[-k:]
            tb = mean([p[1] for p in top]) - mean([p[1] for p in bottom])
            ht = _rate(sum(1 for _, r in top if r > 0), len(top))
            hb = _rate(sum(1 for _, r in bottom if r > 0), len(bottom))
        out[h] = {"n": n, "spearman": rho, "top_minus_bottom": tb,
                  "hit_top": ht, "hit_bottom": hb}
    return out
