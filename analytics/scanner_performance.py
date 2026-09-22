"""Run 38 — reusable scanner-performance analytics (pure, no I/O, no ML).

Measures which scanners produce the strongest, most consistent SUBSEQUENT
outcomes. Consumes normalized outcome records and produces a machine-readable
scoreboard with sample-size protection, regime/session/liquidity segmentation,
and scanner-overlap analysis. Measurement only — nothing here changes any
scanner, threshold, or model.

A normalized outcome record:
    {
      "scanner": str,                 # which scanner fired
      "symbol": str,
      "timestamp": str,               # signal time
      "direction": "long"|"short",    # setups here are long unless stated
      "returns": {"1d": float, ...},  # horizon -> raw return (fraction)
      "mfe": float|None, "mae": float|None,
      "regime": str|None,             # bullish|bearish|neutral (optional)
      "session": str|None,            # premarket|open|morning|midday|afternoon|afterhours
      "liquidity": str|None,          # e.g. large|mid|small (optional)
    }

Adapters (below) build these from the real `signal_outcomes` table and from the
Run 36 canonical observation/outcome schema. No result is manufactured: a horizon
with no data reports None and a scanner under MIN_SAMPLE is INSUFFICIENT_DATA.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Below this many matured observations a scanner cannot be ranked (Task 4).
MIN_SAMPLE = 30
# A larger bar for the strongest classification.
STRONG_SAMPLE = 100


def wilson_interval(hits: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion (hit rate). None when n=0."""
    if n <= 0:
        return (None, None)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(center - margin, 4), round(center + margin, 4))


def directional_return(direction: Optional[str], raw: Optional[float]) -> Optional[float]:
    """Sign a raw return by the setup direction (short inverts). Long is the
    default for scanners that only ever go long."""
    if raw is None:
        return None
    return -raw if str(direction or "long").lower() == "short" else raw


def _horizons(records: Sequence[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for r in records:
        for h in (r.get("returns") or {}):
            if h not in seen:
                seen.append(h)
    return seen


def _horizon_stats(rows: Sequence[Dict[str, Any]], horizon: str) -> Dict[str, Any]:
    vals = []
    for r in rows:
        dr = directional_return(r.get("direction"), (r.get("returns") or {}).get(horizon))
        if dr is not None:
            vals.append(dr)
    n = len(vals)
    if not n:
        return {"n": 0, "hit_rate": None, "avg_return": None, "median_return": None,
                "hit_ci": (None, None)}
    hits = sum(1 for v in vals if v > 0)
    lo, hi = wilson_interval(hits, n)
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "avg_return": round(mean(vals), 6),
        "median_return": round(median(vals), 6),
        "hit_ci": (lo, hi),
    }


def classify_scanner(stats: Dict[str, Any], *, primary_horizon: str) -> str:
    """Evidence-based label. Uses the primary horizon's hit-rate CI vs a 0.5
    baseline and the average directional return sign. Documented, conservative."""
    n = stats.get("signals", 0)
    if n < MIN_SAMPLE:
        return "INSUFFICIENT_DATA"
    h = (stats.get("horizons") or {}).get(primary_horizon) or {}
    hit = h.get("hit_rate")
    lo, hi = h.get("hit_ci", (None, None))
    avg = h.get("avg_return")
    if hit is None or lo is None:
        return "INSUFFICIENT_DATA"
    # PROVEN: CI lower bound clears the coin-flip baseline AND positive edge AND
    # a substantial sample.
    if lo > 0.5 and (avg or 0) > 0 and n >= STRONG_SAMPLE:
        return "PROVEN"
    # PROMISING: point estimate has an edge and returns positive, but the CI still
    # straddles the baseline (needs more data to confirm).
    if hit > 0.5 and (avg or 0) > 0:
        return "PROMISING"
    # WEAK: the edge is negative with the CI upper bound below the baseline, or a
    # clearly negative average.
    if (hi is not None and hi < 0.5) or (avg is not None and avg < 0 and hit < 0.5):
        return "WEAK"
    return "NEUTRAL"


def scanner_scoreboard(
    records: Sequence[Dict[str, Any]],
    *,
    horizons: Optional[Sequence[str]] = None,
    min_sample: int = MIN_SAMPLE,
    primary_horizon: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-scanner scoreboard with sample-size protection and classification."""
    horizons = list(horizons or _horizons(records))
    primary_horizon = primary_horizon or (horizons[-1] if horizons else None)
    by_scanner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_scanner[str(r.get("scanner") or "unknown")].append(r)

    board: Dict[str, Any] = {}
    for scanner, rows in sorted(by_scanner.items()):
        n = len(rows)
        mfes = [r["mfe"] for r in rows if r.get("mfe") is not None]
        maes = [r["mae"] for r in rows if r.get("mae") is not None]
        hstats = {h: _horizon_stats(rows, h) for h in horizons}
        avg_mfe = round(mean(mfes), 6) if mfes else None
        avg_mae = round(mean(maes), 6) if maes else None
        rr = (round(abs(avg_mfe / avg_mae), 3)
              if avg_mfe is not None and avg_mae not in (None, 0) else None)
        stats = {
            "signals": n,
            "horizons": hstats,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "risk_reward": rr,
            "insufficient": n < int(min_sample),
            "min_sample": int(min_sample),
        }
        stats["classification"] = (
            classify_scanner(stats, primary_horizon=primary_horizon)
            if primary_horizon else "INSUFFICIENT_DATA")
        board[scanner] = stats
    return {"scanners": board, "horizons": horizons,
            "primary_horizon": primary_horizon, "min_sample": int(min_sample)}


def segment_scoreboard(
    records: Sequence[Dict[str, Any]], *, key: str,
    horizons: Optional[Sequence[str]] = None, min_sample: int = MIN_SAMPLE,
    primary_horizon: Optional[str] = None,
) -> Dict[str, Any]:
    """Scoreboard segmented by a record field (regime / session / liquidity).
    Segments with no value are grouped under 'unknown'. Only a descriptive split —
    thresholds are not invented to flatter results."""
    by_seg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_seg[str(r.get(key) or "unknown")].append(r)
    return {
        seg: scanner_scoreboard(rows, horizons=horizons, min_sample=min_sample,
                                primary_horizon=primary_horizon)
        for seg, rows in sorted(by_seg.items())
    }


def _day(ts: Any) -> str:
    return str(ts)[:10]


def overlap_analysis(
    records: Sequence[Dict[str, Any]], *, horizon: str, min_sample: int = MIN_SAMPLE,
) -> Dict[str, Any]:
    """Does scanner agreement improve outcomes? Groups records by (symbol, day),
    forms the set of scanners that fired, and compares co-fire outcomes to the
    single-scanner baseline at one horizon. Only combos with >= min_sample are
    reported (Task 8)."""
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in records:
        k = (str(r.get("symbol") or "?").upper(), _day(r.get("timestamp")))
        g = groups.setdefault(k, {"scanners": set(), "rows": []})
        g["scanners"].add(str(r.get("scanner") or "unknown"))
        g["rows"].append(r)

    single: List[float] = []
    multi: List[float] = []
    combo_rows: Dict[str, List[float]] = defaultdict(list)
    agreement_bucket: Dict[str, List[float]] = defaultdict(list)  # "1","2","3+"
    for g in groups.values():
        names = sorted(g["scanners"])
        # one representative directional return per (symbol, day) group
        drs = [directional_return(r.get("direction"), (r.get("returns") or {}).get(horizon))
               for r in g["rows"]]
        drs = [d for d in drs if d is not None]
        if not drs:
            continue
        dr = mean(drs)
        (multi if len(names) >= 2 else single).append(dr)
        bucket = "1" if len(names) == 1 else ("2" if len(names) == 2 else "3+")
        agreement_bucket[bucket].append(dr)
        if len(names) >= 2:
            combo_rows["+".join(names)].append(dr)

    def _summ(vals: List[float]) -> Dict[str, Any]:
        n = len(vals)
        if not n:
            return {"n": 0, "hit_rate": None, "avg_return": None, "sufficient": False}
        hits = sum(1 for v in vals if v > 0)
        return {"n": n, "hit_rate": round(hits / n, 4),
                "avg_return": round(mean(vals), 6),
                "hit_ci": wilson_interval(hits, n),
                "sufficient": n >= int(min_sample)}

    return {
        "horizon": horizon,
        "single_scanner": _summ(single),
        "multi_scanner": _summ(multi),
        "by_agreement_count": {k: _summ(v) for k, v in sorted(agreement_bucket.items())},
        "by_combination": {k: _summ(v) for k, v in sorted(combo_rows.items())
                           if len(v) >= int(min_sample)},
        "min_sample": int(min_sample),
    }


# --- Adapters ----------------------------------------------------------------
def from_signal_outcomes_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize `db.signal_outcomes` rows into scoreboard records.

    signal_outcomes stores DAILY horizons (return_1d/3d/5d, mfe_5d, mae_5d) for
    fired signals; the scanner id is source:signal_type. These signals are long.
    """
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        g = r.get if isinstance(r, dict) else (lambda k, d=None: None)
        source = g("source") or "unknown"
        stype = g("signal_type") or ""
        returns = {}
        for label, col in (("1d", "return_1d"), ("3d", "return_3d"), ("5d", "return_5d")):
            v = g(col)
            if v is not None:
                returns[label] = float(v)
        out.append({
            "scanner": f"{source}:{stype}" if stype else str(source),
            "symbol": g("ticker"), "timestamp": str(g("fired_at")),
            "direction": "long",
            "returns": returns,
            "mfe": (float(g("mfe_5d")) if g("mfe_5d") is not None else None),
            "mae": (float(g("mae_5d")) if g("mae_5d") is not None else None),
            "regime": g("regime"), "session": g("session"), "liquidity": g("liquidity"),
        })
    return out


def from_canonical_observations(observations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize Run 36 canonical observations-with-outcomes into scoreboard
    records — one record per (observation, scanner) so multi-scanner observations
    contribute to each scanner they triggered. Uses attached price outcomes."""
    out: List[Dict[str, Any]] = []
    for o in observations or []:
        outcomes = o.get("outcomes") or {}
        returns = {str(h).lstrip("+"): oc.get("raw_return")
                   for h, oc in outcomes.items() if oc.get("raw_return") is not None}
        mfe = next((oc.get("mfe") for oc in outcomes.values() if oc.get("mfe") is not None), None)
        mae = next((oc.get("mae") for oc in outcomes.values() if oc.get("mae") is not None), None)
        ctx = o.get("market_context") or {}
        dq = o.get("data_quality") or {}
        for s in (o.get("scanners") or []):
            if not s.get("triggered", True):
                continue
            out.append({
                "scanner": str(s.get("name") or "unknown"),
                "symbol": o.get("symbol"), "timestamp": o.get("timestamp"),
                "direction": str(s.get("direction") or "long"),
                "returns": dict(returns), "mfe": mfe, "mae": mae,
                "regime": ctx.get("market_regime"), "session": o.get("session"),
                "liquidity": ctx.get("liquidity"),
                # Research-quality metadata (Task 13) so the scoreboard can filter.
                "source": ctx.get("source") or str(o.get("context") or "").split(":")[0] or None,
                "feature_completeness": dq.get("feature_completeness"),
                "fallback": bool(dq.get("fallback_used")),
                "stale": bool(dq.get("stale")),
            })
    return out


# --- Research data-quality filtering (Task 13) -------------------------------
RESEARCH_MIN_COMPLETENESS = 0.5


def filter_research_records(
    records: Sequence[Dict[str, Any]], *,
    source: Optional[str] = "scheduled",
    allow_partial: bool = True,
    allow_fallback: bool = False,
    allow_stale: bool = False,
    min_completeness: float = RESEARCH_MIN_COMPLETENESS,
) -> List[Dict[str, Any]]:
    """Select clean research records. Defaults to scheduled production only, no
    fallback, no stale. Never silently mixes manual/test/reconstructed sources
    unless `source=None` is explicitly passed."""
    out = []
    for r in records:
        if source is not None and (r.get("source") or "scheduled") != source:
            continue
        if not allow_fallback and r.get("fallback"):
            continue
        if not allow_stale and r.get("stale"):
            continue
        fc = r.get("feature_completeness")
        if fc is not None and fc < min_completeness:
            continue
        if not allow_partial and fc is not None and fc < 0.999:
            continue
        out.append(r)
    return out


def sample_readiness(
    records: Sequence[Dict[str, Any]], *,
    horizons: Optional[Sequence[str]] = None, min_sample: int = MIN_SAMPLE,
    strong_sample: int = STRONG_SAMPLE,
) -> Dict[str, Any]:
    """Progress toward Run 38's sample gates, per scanner and horizon (Task 12).

    Readiness indicator ONLY — N >= min_sample does not imply statistical
    significance, only that the scoreboard will stop reporting INSUFFICIENT_DATA.
    """
    horizons = list(horizons or _horizons(records))
    by_scanner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_scanner[str(r.get("scanner") or "unknown")].append(r)
    out: Dict[str, Any] = {}
    for scanner, rows in sorted(by_scanner.items()):
        hz = {}
        for h in horizons:
            n = sum(1 for r in rows if (r.get("returns") or {}).get(h) is not None)
            hz[h] = {"n": n, "min_sample": min_sample,
                     "ready": n >= min_sample, "strong": n >= strong_sample}
        out[scanner] = {"signals": len(rows), "horizons": hz}
    return {"scanners": out, "min_sample": min_sample, "strong_sample": strong_sample,
            "note": "readiness indicator only; N>=min_sample is not significance"}


def build_scoreboard_report(
    records: Sequence[Dict[str, Any]], *,
    horizons: Optional[Sequence[str]] = None,
    primary_horizon: Optional[str] = None,
    min_sample: int = MIN_SAMPLE,
) -> Dict[str, Any]:
    """Full machine-readable report: overall scoreboard + regime/session/liquidity
    segmentation + overlap. Suitable for a future dashboard."""
    horizons = list(horizons or _horizons(records))
    primary = primary_horizon or (horizons[-1] if horizons else None)
    return {
        "schema": "hsf-scanner-scoreboard-1.0",
        "n_records": len(records),
        "horizons": horizons,
        "primary_horizon": primary,
        "overall": scanner_scoreboard(records, horizons=horizons,
                                      primary_horizon=primary, min_sample=min_sample),
        "by_regime": segment_scoreboard(records, key="regime", horizons=horizons,
                                        primary_horizon=primary, min_sample=min_sample),
        "by_session": segment_scoreboard(records, key="session", horizons=horizons,
                                         primary_horizon=primary, min_sample=min_sample),
        "by_liquidity": segment_scoreboard(records, key="liquidity", horizons=horizons,
                                           primary_horizon=primary, min_sample=min_sample),
        "overlap": (overlap_analysis(records, horizon=primary, min_sample=min_sample)
                    if primary else None),
    }
