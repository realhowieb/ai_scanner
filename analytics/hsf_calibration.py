"""HSF Opportunity Score calibration & validation (pure analytics).

Answers, from REAL stored outcomes only: does a higher HSF score correspond to
better forward outcomes? Builds a leakage-safe calibration dataset (frozen
signal-time features joined to matured forward outcomes) and computes score-
bucket / status / signal-type / combination / baseline / monotonicity /
calibration diagnostics. Never fabricates statistics; every summary reports n
and a sample-size confidence label. All functions are safe on empty input.

Positive outcome = reached +4% within 5 trading days (mfe_5d >= 0.04), the
models' economic target — falls back to return_5d >= 0.04 when MFE is absent.
This is a positive-outcome rate, NOT a trading win rate.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

POSITIVE_MFE_THRESHOLD = 0.04
SCORE_BUCKETS = [(0, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 100)]


def confidence_label(n: int) -> str:
    n = int(n or 0)
    if n < 10:
        return "INSUFFICIENT DATA"
    if n < 30:
        return "LOW CONFIDENCE"
    if n < 100:
        return "MODERATE CONFIDENCE"
    return "STRONGER EVIDENCE"


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _positive(rec: Dict[str, Any]) -> Optional[bool]:
    """True/False when matured, None when pending. Uses MFE, falls back to 5D."""
    if not rec.get("matured"):
        return None
    mfe = rec.get("mfe_5d")
    if mfe is not None:
        return float(mfe) >= POSITIVE_MFE_THRESHOLD
    r5 = rec.get("return_5d")
    if r5 is not None:
        return float(r5) >= POSITIVE_MFE_THRESHOLD
    return None


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a raw signal_outcomes 'opportunity' row into a calibration record.

    Reads the frozen score/features from raw_signal + indicators (signal-time
    only) and the outcomes from the return_/mfe_/mae_ columns. No outcome field
    is ever pulled into the feature side.
    """
    raw = row.get("raw_signal") or {}
    ind = row.get("indicators") or {}
    comps = raw.get("score_components") or {}
    matured = row.get("outcome_computed_at") is not None
    rec = {
        "ticker": row.get("ticker"),
        "fired_at": row.get("fired_at"),
        "hsf_score": _num(raw.get("hsf_score")),
        "score_version": raw.get("score_version"),
        "status": raw.get("status") or ind.get("status"),
        "primary_setup": raw.get("primary_setup") or ind.get("primary_setup"),
        "signals": list(ind.get("signals") or []),
        "n_signals": ind.get("n_signals"),
        "breakout_score": _num(row.get("setup_score")),
        "prob": _num(row.get("prebreakout_prob")),
        "chg_pct": _num(ind.get("chg_pct")),
        "fading": bool(ind.get("fading")),
        "signals_component": _num(comps.get("signals_component")),
        "model_component": _num(comps.get("model_component")),
        "momentum_component": _num(comps.get("momentum_component")),
        "fading_penalty": _num(comps.get("fading_penalty")),
        "matured": matured,
        "return_5d": _num(row.get("return_5d")),
        "mfe_5d": _num(row.get("mfe_5d")),
        "mae_5d": _num(row.get("mae_5d")),
    }
    rec["positive"] = _positive(rec)
    return rec


def build_calibration_dataset(
    rows: Optional[List[Dict[str, Any]]] = None, *, days_back: int = 180
) -> Dict[str, Any]:
    """Join frozen opportunities to matured outcomes. Pass `rows` for tests;
    otherwise fetches from the DB. Distinguishes matured vs pending."""
    if rows is None:
        try:
            from db.signal_outcomes import fetch_opportunity_outcomes

            rows = fetch_opportunity_outcomes(days_back=days_back)
        except Exception:
            rows = []
    records = [normalize_row(r) for r in (rows or [])]
    matured = [r for r in records if r["matured"]]
    pending = [r for r in records if not r["matured"]]
    fired = [r["fired_at"] for r in records if r.get("fired_at") is not None]
    versions: Dict[str, int] = {}
    for r in records:
        v = str(r.get("score_version") or "unknown")
        versions[v] = versions.get(v, 0) + 1
    return {
        "records": records,
        "matured": matured,
        "n_total": len(records),
        "n_matured": len(matured),
        "n_pending": len(pending),
        "date_range": (min(fired), max(fired)) if fired else (None, None),
        "versions": versions,
        "quality": data_quality_checks(records),
    }


def _bucket_of(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    for lo, hi in SCORE_BUCKETS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return None


def _stat_block(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """positive-rate + MFE/MAE/return distribution for a group of records."""
    matured = [r for r in recs if r["matured"] and r["positive"] is not None]
    n_mat = len(matured)
    pos = sum(1 for r in matured if r["positive"])
    mfe = [r["mfe_5d"] for r in matured if r["mfe_5d"] is not None]
    mae = [r["mae_5d"] for r in matured if r["mae_5d"] is not None]
    ret = [r["return_5d"] for r in matured if r["return_5d"] is not None]

    def med(x):
        return round(statistics.median(x), 4) if x else None

    def avg(x):
        return round(statistics.fmean(x), 4) if x else None

    return {
        "n_total": len(recs),
        "n_matured": n_mat,
        "positive_count": pos,
        "positive_rate": round(pos / n_mat, 4) if n_mat else None,
        "median_mfe_5d": med(mfe), "avg_mfe_5d": avg(mfe),
        "median_mae_5d": med(mae), "avg_mae_5d": avg(mae),
        "median_return_5d": med(ret), "avg_return_5d": avg(ret),
        "confidence": confidence_label(n_mat),
    }


def summarize_score_buckets(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {f"{lo}-{hi}": [] for lo, hi in SCORE_BUCKETS}
    for r in records:
        b = _bucket_of(r.get("hsf_score"))
        if b is not None:
            groups[b].append(r)
    return [{"bucket": b, **_stat_block(groups[b])} for b, _ in [(f"{lo}-{hi}", None) for lo, hi in SCORE_BUCKETS]]


def summarize_status_performance(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for status in ("STRONG", "WATCH", "CAUTION"):
        recs = [r for r in records if str(r.get("status") or "").upper() == status]
        out.append({"status": status, **_stat_block(recs)})
    return out


def summarize_signal_types(records: List[Dict[str, Any]], *, min_n: int = 1) -> List[Dict[str, Any]]:
    types = ["breakout", "golden_cross", "prebreakout", "gapper", "gainer", "fading"]
    out = []
    for t in types:
        recs = [r for r in records if t in (r.get("signals") or []) or (t == "fading" and r.get("fading"))]
        block = _stat_block(recs)
        if block["n_total"] >= min_n:
            out.append({"signal_type": t, **block})
    return out


def analyze_signal_combinations(records: List[Dict[str, Any]], *, min_n: int = 10) -> List[Dict[str, Any]]:
    combos = [
        ("breakout+golden_cross", lambda s: "breakout" in s and "golden_cross" in s),
        ("breakout+prebreakout", lambda s: "breakout" in s and "prebreakout" in s),
        ("gapper+breakout", lambda s: "gapper" in s and "breakout" in s),
        ("golden_cross+prebreakout", lambda s: "golden_cross" in s and "prebreakout" in s),
        ("3+ signals", lambda s: len(s) >= 3),
    ]
    out = []
    for name, pred in combos:
        recs = [r for r in records if pred(set(r.get("signals") or []))]
        block = _stat_block(recs)
        if block["n_matured"] >= min_n:  # only report meaningful samples
            out.append({"combination": name, **block})
    return out


def _auc(scores: Sequence[Optional[float]], labels: Sequence[int]) -> Optional[float]:
    """Mann-Whitney AUC of a ranker vs binary outcome. None if not computable."""
    pairs = [(s, y) for s, y in zip(scores, labels) if s is not None]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    # Rank all values (average ranks for ties), sum ranks of positives.
    order = sorted(pairs, key=lambda x: x[0])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank for the tie block
        for k in range(i, j):
            ranks.setdefault(id(order[k]), avg_rank)
        i = j
    rank_sum = 0.0
    for idx, (s, y) in enumerate(order):
        if y == 1:
            rank_sum += ranks[id(order[idx])]
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return round(u / (n_pos * n_neg), 4)


def compare_baselines(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discrimination (AUC) of HSF score vs its simpler inputs, on matured rows
    with a defined outcome. Each row reports n so tiny samples are visible."""
    matured = [r for r in records if r["matured"] and r["positive"] is not None]
    labels = [1 if r["positive"] else 0 for r in matured]
    base_rate = round(sum(labels) / len(labels), 4) if labels else None
    rankers = {
        "hsf_score": [r.get("hsf_score") for r in matured],
        "n_signals": [r.get("n_signals") for r in matured],
        "breakout_score": [r.get("breakout_score") for r in matured],
        "prebreakout_prob": [r.get("prob") for r in matured],
        "chg_pct": [r.get("chg_pct") for r in matured],
        "hsf_no_momentum": [(_num(r.get("signals_component")) or 0) + (_num(r.get("model_component")) or 0)
                            - (_num(r.get("fading_penalty")) or 0) for r in matured],
        "hsf_no_signals": [(_num(r.get("model_component")) or 0) + (_num(r.get("momentum_component")) or 0)
                           - (_num(r.get("fading_penalty")) or 0) for r in matured],
        "hsf_no_model": [(_num(r.get("signals_component")) or 0) + (_num(r.get("momentum_component")) or 0)
                         - (_num(r.get("fading_penalty")) or 0) for r in matured],
    }
    out = []
    for name, scores in rankers.items():
        n_usable = sum(1 for s in scores if s is not None)
        out.append({"ranker": name, "auc": _auc(scores, labels),
                    "n": n_usable, "base_rate": base_rate,
                    "confidence": confidence_label(n_usable)})
    return out


def evaluate_monotonicity(buckets: List[Dict[str, Any]], *, min_n: int = 10) -> Dict[str, Any]:
    """Is positive_rate non-decreasing across score buckets (with enough n)?"""
    pts = [(b["bucket"], b["positive_rate"]) for b in buckets
           if b.get("positive_rate") is not None and b.get("n_matured", 0) >= min_n]
    if len(pts) < 2:
        return {"monotonic": None, "reason": "insufficient buckets with data",
                "points": pts, "confidence": "INSUFFICIENT DATA"}
    rates = [p[1] for p in pts]
    non_decreasing = all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    # Spearman-ish: rank correlation of bucket order vs rate order.
    ranks = list(range(len(rates)))
    mean_r = statistics.fmean(ranks)
    mean_v = statistics.fmean(rates)
    num = sum((ranks[i] - mean_r) * (rates[i] - mean_v) for i in range(len(rates)))
    den_r = sum((x - mean_r) ** 2 for x in ranks) ** 0.5
    den_v = sum((x - mean_v) ** 2 for x in rates) ** 0.5
    corr = round(num / (den_r * den_v), 4) if den_r and den_v else None
    return {"monotonic": bool(non_decreasing), "rank_correlation": corr,
            "points": pts, "confidence": confidence_label(sum(b.get("n_matured", 0) for b in buckets))}


def evaluate_calibration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Brier score of (hsf_score/100) as a probability vs positive outcome, plus
    the reliability table (empirical rate per bucket)."""
    matured = [r for r in records if r["matured"] and r["positive"] is not None and r.get("hsf_score") is not None]
    n = len(matured)
    if n == 0:
        return {"n": 0, "brier": None, "reliability": [], "confidence": "INSUFFICIENT DATA"}
    brier = statistics.fmean(
        ((r["hsf_score"] / 100.0) - (1.0 if r["positive"] else 0.0)) ** 2 for r in matured
    )
    reliability = [
        {"bucket": b["bucket"], "predicted": _bucket_mid(b["bucket"]),
         "observed": b["positive_rate"], "n": b["n_matured"]}
        for b in summarize_score_buckets(matured) if b["n_matured"] > 0
    ]
    return {"n": n, "brier": round(brier, 4), "reliability": reliability,
            "confidence": confidence_label(n)}


def _bucket_mid(bucket: str) -> Optional[float]:
    try:
        lo, hi = bucket.split("-")
        return round((int(lo) + int(hi)) / 200.0, 3)  # midpoint as a 0-1 prob
    except Exception:
        return None


def score_distribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["hsf_score"] for r in records if r.get("hsf_score") is not None]
    if not scores:
        return {"n": 0}
    ss = sorted(scores)

    def pct(p):
        idx = min(len(ss) - 1, int(round(p / 100.0 * (len(ss) - 1))))
        return ss[idx]

    counts = {f"{lo}-{hi}": sum(1 for s in scores if lo <= s <= hi) for lo, hi in SCORE_BUCKETS}
    return {
        "n": len(scores), "min": min(scores), "max": max(scores),
        "median": round(statistics.median(scores), 1), "mean": round(statistics.fmean(scores), 1),
        "p25": pct(25), "p75": pct(75), "p90": pct(90),
        "bucket_counts": counts,
    }


def data_quality_checks(records: List[Dict[str, Any]]) -> List[str]:
    """Fail-safe diagnostics — flags bad data instead of silently biasing stats."""
    warnings: List[str] = []
    seen = set()
    dups = 0
    invalid = 0
    missing_ticker = 0
    missing_version = 0
    missing_score = 0
    missing_components = 0
    early_outcome = 0
    comp_keys = ("signals_component", "model_component", "momentum_component", "fading_penalty")
    for r in records:
        key = (str(r.get("ticker") or "").upper(), r.get("fired_at"))
        if key in seen:
            dups += 1
        seen.add(key)
        s = r.get("hsf_score")
        if s is None:
            missing_score += 1
        elif s < 0 or s > 100:
            invalid += 1
        if not r.get("ticker"):
            missing_ticker += 1
        if not r.get("score_version"):
            missing_version += 1
        if all(r.get(k) is None for k in comp_keys):
            missing_components += 1
        if r.get("matured") and r.get("mfe_5d") is None and r.get("return_5d") is None:
            early_outcome += 1
    if dups:
        warnings.append(f"{dups} duplicate (ticker, fired_at) record(s)")
    if invalid:
        warnings.append(f"{invalid} score(s) outside 0-100")
    if missing_score:
        warnings.append(f"{missing_score} record(s) missing hsf_score")
    if missing_ticker:
        warnings.append(f"{missing_ticker} record(s) missing ticker")
    if missing_version:
        warnings.append(f"{missing_version} record(s) missing score_version (legacy/unknown)")
    if missing_components:
        warnings.append(f"{missing_components} record(s) missing score components")
    if early_outcome:
        warnings.append(f"{early_outcome} matured record(s) with no MFE/return (outcome may be incomplete)")
    return warnings


def historical_context(records: List[Dict[str, Any]], score: Optional[float]) -> Optional[Dict[str, Any]]:
    """Bucket context for a single live score — for the Market Brief line.

    Returns {bucket, positive_rate, n, confidence, sufficient} or None when no
    score. `sufficient` is False until the bucket has >= 10 matured samples, so
    the UI can show 'still building history' instead of misleading precision.
    """
    b = _bucket_of(score)
    if b is None:
        return None
    recs = [r for r in records if _bucket_of(r.get("hsf_score")) == b]
    block = _stat_block(recs)
    n = block["n_matured"]
    return {
        "bucket": b, "positive_rate": block["positive_rate"], "n": n,
        "median_mfe_5d": block["median_mfe_5d"], "median_mae_5d": block["median_mae_5d"],
        "confidence": confidence_label(n), "sufficient": n >= 10,
    }
