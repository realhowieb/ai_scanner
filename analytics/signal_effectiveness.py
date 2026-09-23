"""Run 48 — signal effectiveness & selection-value analysis (pure, deterministic).

MEASUREMENT ONLY. Consumes the Run 47 research cohorts (candidate/near-miss/
control) joined to their separate matured outcomes and asks: does HSF select
better stocks than it rejects, and which signals carry evidence? It changes NO
scoring/scanner/ML/ranking — it only computes statistics.

Discipline built in: every breakdown reports N; conclusions below a minimum
sample are labeled INSUFFICIENT; uncertainty is quantified with deterministic
(seeded) bootstrap CIs; effect size is reported alongside CIs; and a manifest
separates PRE-SPECIFIED from EXPLORATORY analyses to resist p-hacking.
"""
from __future__ import annotations

import random
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Sequence

from analytics.scanner_performance import wilson_interval

HORIZONS = ("+5m", "+15m", "+30m", "+60m", "EOD")
# Sample-size → evidence level (documented, conservative).
MIN_SAMPLE = 30
_EVIDENCE_BANDS = ((30, "INSUFFICIENT"), (100, "PRELIMINARY"),
                   (500, "MODERATE"), (10**12, "STRONG"))
# A "meaningful move" threshold for meaningful-move %% (research only).
_MEANINGFUL = 0.005  # 0.5%


def evidence_level(n: int) -> str:
    for threshold, label in _EVIDENCE_BANDS:
        if n < threshold:
            return label
    return "STRONG"


def _cohort_of(o: Dict[str, Any]) -> str:
    c = o.get("research_cohort") or (o.get("market_context") or {}).get("research_cohort")
    return str(c) if c else "CANDIDATE"


def _direction_of(o: Dict[str, Any]) -> str:
    for s in (o.get("scanners") or []):
        if s.get("triggered", True) and s.get("direction"):
            return "short" if str(s["direction"]).lower() in ("short", "bearish") else "long"
    return "long"


def observation_to_record(o: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an observation-with-outcomes into an analysis record. Directional
    return = raw (long) or −raw (short). Outcomes come ONLY from the separate
    outcomes namespace; features are point-in-time."""
    ind = o.get("indicators") or {}
    models = o.get("models") or {}
    direction = _direction_of(o)
    outcomes = o.get("outcomes") or {}
    rec: Dict[str, Any] = {
        "cohort": _cohort_of(o),
        "direction": direction,
        "session": o.get("session"),
        "regime": (o.get("market_context") or {}).get("market_regime"),
        "coverage_health": (o.get("market_context") or {}).get("coverage_health"),
        "rank": o.get("rank"),
        "features": {
            "rvol": ind.get("rvol"), "adx": ind.get("adx"),
            "vs_vwap_pct": ind.get("vs_vwap_pct"), "gap_pct": ind.get("gap_pct"),
            "chg_pct": ind.get("chg_pct"), "ema9": ind.get("ema9"),
            "ema21": ind.get("ema21"), "rsi": ind.get("rsi"),
            "supertrend_direction": ind.get("supertrend_direction"), "ewo": ind.get("ewo"),
            "prebreakout": (models.get("prebreakout") or {}).get("probability"),
            "ai_confidence": (models.get("ai_confidence") or {}).get("confidence"),
            "price": (o.get("market") or {}).get("price"),
            "volume": (o.get("market") or {}).get("volume"),
        },
        "outcomes": {},
    }
    for h, oc in outcomes.items():
        if not isinstance(oc, dict) or str(oc.get("data_status")) != "MATURED":
            continue
        raw = oc.get("raw_return")
        if raw is None:
            continue
        rec["outcomes"][h] = {
            "directional_return": -raw if direction == "short" else raw,
            "raw_return": raw, "mfe": oc.get("mfe"), "mae": oc.get("mae"),
        }
    return rec


def _num(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def bootstrap_ci(values: Sequence[float], *, stat: Callable = median,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, Any]:
    """Deterministic (seeded) bootstrap CI for a statistic. None when too few."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < MIN_SAMPLE:
        return {"estimate": round(stat(vals), 6) if vals else None,
                "ci": (None, None), "n": len(vals), "sufficient": False}
    rng = random.Random(seed)
    boots = []
    k = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(k)] for _ in range(k)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"estimate": round(stat(vals), 6), "ci": (round(lo, 6), round(hi, 6)),
            "n": len(vals), "sufficient": True}


def _returns(records: Sequence[Dict[str, Any]], horizon: str) -> List[float]:
    out = []
    for r in records:
        oc = (r.get("outcomes") or {}).get(horizon)
        if oc and oc.get("directional_return") is not None:
            out.append(float(oc["directional_return"]))
    return out


def horizon_stats(records: Sequence[Dict[str, Any]], horizon: str, *, seed: int = 0) -> Dict[str, Any]:
    """Robust stats for one group at one horizon (Task 3)."""
    vals = _returns(records, horizon)
    n = len(vals)
    if n == 0:
        return {"n": 0, "evidence": "INSUFFICIENT"}
    hits = sum(1 for v in vals if v > 0)
    mfes = [_num((r["outcomes"].get(horizon) or {}).get("mfe")) for r in records
            if (r["outcomes"].get(horizon))]
    maes = [_num((r["outcomes"].get(horizon) or {}).get("mae")) for r in records
            if (r["outcomes"].get(horizon))]
    mfes = [x for x in mfes if x is not None]
    maes = [x for x in maes if x is not None]
    lo, hi = wilson_interval(hits, n)
    return {
        "n": n, "evidence": evidence_level(n),
        "mean_directional_return": round(mean(vals), 6),
        "median_directional_return": round(median(vals), 6),
        "win_rate": round(hits / n, 4), "win_rate_ci": (lo, hi),
        "positive_pct": round(hits / n, 4),
        "meaningful_move_pct": round(sum(1 for v in vals if abs(v) >= _MEANINGFUL) / n, 4),
        "median_mfe": round(median(mfes), 6) if mfes else None,
        "median_mae": round(median(maes), 6) if maes else None,
        "median_ci": bootstrap_ci(vals, seed=seed)["ci"],
    }


def cohort_performance(records: Sequence[Dict[str, Any]], *,
                       horizons: Sequence[str] = HORIZONS) -> Dict[str, Any]:
    by_cohort: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_cohort.setdefault(r["cohort"], []).append(r)
    return {c: {h: horizon_stats(rows, h) for h in horizons}
            for c, rows in sorted(by_cohort.items())}


def selection_lift(records: Sequence[Dict[str, Any]], horizon: str, *, seed: int = 0) -> Dict[str, Any]:
    """Candidate lift vs control and near-miss at one horizon (Task 4/5). Reports
    the median-difference estimate + a bootstrap CI of the difference."""
    groups = {c: _returns([r for r in records if r["cohort"] == c], horizon)
              for c in ("CANDIDATE", "NEAR_MISS", "CONTROL")}

    def _diff(a, b):
        if len(a) < MIN_SAMPLE or len(b) < MIN_SAMPLE:
            return {"estimate": None, "ci": (None, None), "n_a": len(a), "n_b": len(b),
                    "evidence": "INSUFFICIENT"}
        rng = random.Random(seed)
        boots = []
        for _ in range(1000):
            sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
            sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
            boots.append(median(sa) - median(sb))
        boots.sort()
        return {"estimate": round(median(a) - median(b), 6),
                "ci": (round(boots[25], 6), round(boots[974], 6)),
                "n_a": len(a), "n_b": len(b), "evidence": evidence_level(min(len(a), len(b)))}
    return {"horizon": horizon,
            "candidate_vs_control": _diff(groups["CANDIDATE"], groups["CONTROL"]),
            "candidate_vs_near_miss": _diff(groups["CANDIDATE"], groups["NEAR_MISS"])}


def _quantile_bins(records, feature, n_bins=4):
    vals = [( _num(r["features"].get(feature)), r) for r in records]
    vals = [(v, r) for v, r in vals if v is not None]
    vals.sort(key=lambda x: x[0])
    if len(vals) < n_bins:
        return []
    size = len(vals) // n_bins
    return [[r for _, r in vals[i * size:(i + 1) * size]] for i in range(n_bins)]


def signal_effectiveness(records, feature, horizon, *, n_bins=4) -> Dict[str, Any]:
    """Relationship between a feature's quantile bins and outcomes (Task 6/7/9).
    Non-linear-friendly (quantile bins), monotonicity flagged."""
    bins = _quantile_bins(records, feature, n_bins)
    if not bins:
        return {"feature": feature, "horizon": horizon, "n": 0, "evidence": "INSUFFICIENT"}
    band_stats = [horizon_stats(b, horizon) for b in bins]
    medians = [b.get("median_directional_return") for b in band_stats]
    valid = [m for m in medians if m is not None]
    monotonic = (len(valid) == len(medians) and
                 (all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1)) or
                  all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1))))
    total_n = sum(b.get("n", 0) for b in band_stats)
    return {"feature": feature, "horizon": horizon, "n": total_n,
            "evidence": evidence_level(total_n), "bands": band_stats,
            "monotonic": monotonic if total_n >= MIN_SAMPLE else None}


def group_analysis(records, key: str, horizon: str) -> Dict[str, Any]:
    """Generic grouping (direction/regime/session/tier) → per-group horizon stats."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(key) or "unknown"), []).append(r)
    return {g: horizon_stats(rows, horizon) for g, rows in sorted(groups.items())}


def rank_bucket_analysis(records, horizon: str, *, buckets=4) -> Dict[str, Any]:
    """Do higher-ranked candidates outperform lower-ranked (Task 18)?"""
    cands = [r for r in records if r["cohort"] == "CANDIDATE" and r.get("rank") is not None]
    cands.sort(key=lambda r: r["rank"])
    if len(cands) < buckets:
        return {"n": len(cands), "evidence": "INSUFFICIENT"}
    size = len(cands) // buckets
    return {"evidence": evidence_level(len(cands)),
            "buckets": [horizon_stats(cands[i * size:(i + 1) * size], horizon)
                        for i in range(buckets)]}


# --- Scorecard + recommendations ---------------------------------------------
SCORECARD_SIGNALS = ("prebreakout", "ai_confidence", "rvol", "adx", "vs_vwap_pct",
                     "ema9", "rsi", "supertrend_direction", "ewo", "chg_pct")


def _signal_status(eff: Dict[str, Any]) -> str:
    if eff.get("n", 0) < MIN_SAMPLE:
        return "INSUFFICIENT DATA"
    bands = eff.get("bands") or []
    top = bands[-1].get("median_directional_return") if bands else None
    bot = bands[0].get("median_directional_return") if bands else None
    if top is None or bot is None:
        return "INSUFFICIENT DATA"
    spread = top - bot
    if eff.get("monotonic") and spread > _MEANINGFUL:
        return "STRONG POSITIVE"
    if spread > _MEANINGFUL:
        return "POSITIVE"
    if spread < -_MEANINGFUL:
        return "NEGATIVE"
    if abs(spread) <= _MEANINGFUL / 2:
        return "NEUTRAL"
    return "MIXED"


def signal_scorecard(records, *, horizon: str = "+60m") -> Dict[str, Any]:
    out = {}
    for sig in SCORECARD_SIGNALS:
        eff = signal_effectiveness(records, sig, horizon)
        out[sig] = {"status": _signal_status(eff), "n": eff.get("n", 0),
                    "evidence": eff.get("evidence"), "horizon": horizon,
                    "monotonic": eff.get("monotonic")}
    return out


def build_recommendations(scorecard: Dict[str, Any],
                          cohort_perf: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run 49 recommendation manifest (Task 25/26) — evidence, not action."""
    recs: List[Dict[str, Any]] = []
    for sig, s in scorecard.items():
        if s["status"] == "INSUFFICIENT DATA":
            action, cat = "collect_more_data", "INSUFFICIENT DATA"
        elif s["status"] in ("STRONG POSITIVE", "POSITIVE"):
            action, cat = "keep", "KEEP"
        elif s["status"] == "NEGATIVE":
            action, cat = "investigate_removal", "POTENTIAL REMOVAL"
        elif s["status"] == "NEUTRAL":
            action, cat = "investigate_redundancy", "SIMPLIFY CANDIDATE"
        else:
            action, cat = "investigate", "INVESTIGATE"
        recs.append({"component": sig, "finding": s["status"], "category": cat,
                     "evidence_level": s["evidence"], "sample_size": s["n"],
                     "horizons": [s["horizon"]], "recommended_action": action})
    return recs


def dataset_evidence_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall dataset readiness (Task 1/30) — drives the effectiveness verdict."""
    by_cohort: Dict[str, int] = {}
    matured = {h: 0 for h in HORIZONS}
    for r in records:
        by_cohort[r["cohort"]] = by_cohort.get(r["cohort"], 0) + 1
        for h in HORIZONS:
            if (r.get("outcomes") or {}).get(h):
                matured[h] += 1
    # Effectiveness needs matured directional outcomes in candidate AND control.
    cand_out = sum(1 for r in records if r["cohort"] == "CANDIDATE" and r["outcomes"])
    ctrl_out = sum(1 for r in records if r["cohort"] == "CONTROL" and r["outcomes"])
    usable = min(cand_out, ctrl_out)
    return {"total_records": len(records), "by_cohort": by_cohort,
            "matured_by_horizon": matured,
            "candidate_with_outcomes": cand_out, "control_with_outcomes": ctrl_out,
            "usable_paired_n": usable, "evidence_level": evidence_level(usable),
            "sufficient_for_conclusions": usable >= MIN_SAMPLE}


ANALYSIS_MANIFEST = {
    "pre_specified": ["cohort_performance", "selection_lift(candidate_vs_control)",
                      "selection_lift(candidate_vs_near_miss)", "rank_bucket_analysis",
                      "prebreakout signal_effectiveness", "signal_scorecard"],
    "exploratory": ["per-feature signal_effectiveness (rvol/adx/vwap/ema/rsi/…)",
                    "regime/session/direction group_analysis", "combinations"],
    "multiple_testing_note": "Exploratory results are uncorrected and must not be "
                             "presented as confirmed; effect size + CI + N reported "
                             "for every finding.",
}


def build_analysis_report(observations: Sequence[Dict[str, Any]], *,
                          healthy_only: bool = True,
                          horizons: Sequence[str] = HORIZONS) -> Dict[str, Any]:
    """Full machine-readable Run 48 artifact (Task 27). Primary analysis is
    HEALTHY-scan-only (Task 2)."""
    records = [observation_to_record(o) for o in observations]
    if healthy_only:
        records = [r for r in records if r.get("coverage_health") == "HEALTHY"]
    summary = dataset_evidence_summary(records)
    perf = cohort_performance(records, horizons=horizons)
    scorecard = signal_scorecard(records)
    return {
        "schema": "hsf-signal-effectiveness-1.0",
        "healthy_only": healthy_only,
        "dataset": summary,
        "manifest": ANALYSIS_MANIFEST,
        "cohort_performance": perf,
        "selection_lift": {h: selection_lift(records, h) for h in horizons},
        "rank_buckets": {h: rank_bucket_analysis(records, h) for h in horizons},
        "signal_scorecard": scorecard,
        "by_direction": {h: group_analysis(records, "direction", h) for h in horizons},
        "by_regime": {h: group_analysis(records, "regime", h) for h in horizons},
        "by_session": {h: group_analysis(records, "session", h) for h in horizons},
        "recommendations": build_recommendations(scorecard, perf),
        "effectiveness_verdict": ("INSUFFICIENT LIVE DATA"
                                  if not summary["sufficient_for_conclusions"] else "SEE_REPORT"),
    }
