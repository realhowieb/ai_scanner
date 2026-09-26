"""Run 56 — forward-evidence readiness monitor (pure, read-only, anti-peeking).

Answers one question: do we have enough trustworthy FORWARD production evidence
(collected after the Run 55 evaluation) to rerun the Run 55 effectiveness
analysis? It measures evidence QUANTITY and QUALITY only.

Anti-peeking contract: this module never computes or exposes any effectiveness
statistic (win rate, mean or median return, cohort return differences, score or
feature correlations, best buckets or thresholds). It reads outcome records only
to check that fields are PRESENT, and to run integrity checks that are pure
equality tests (duplicate conflicts, direction-transform consistency). Its output
is therefore invariant to the sign and size of returns, which a test enforces,
and `assert_no_effectiveness_metrics` rejects any report that carries a
prohibited key.

All gates and state rules below are pre-registered (Run 56) and must not be
tuned against forward data. Nothing here writes to the store or changes the
scanner, scoring, cohorts, or maturation.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analytics.hsf_observation import OBSERVATION_SCHEMA_VERSION, OUTCOME_SCHEMA_VERSION
from analytics.research_cohorts import CANDIDATE, CONTROL, NEAR_MISS

SCHEMA = "hsf-forward-readiness-1.0"
HORIZONS = ("+5m", "+15m", "+30m", "+60m")
HORIZON_MIN = {"+5m": 5, "+15m": 15, "+30m": 30, "+60m": 60}
COHORTS = (CANDIDATE, NEAR_MISS, CONTROL)

# ---- Forward epoch (Part 1): everything anchored at/after this instant is new
# evidence; everything before it was inspected by Run 55 and is excluded. ------
FORWARD_EPOCH = {
    "forward_epoch_start_timestamp": "2026-09-26T07:23:11+00:00",  # Run 55 production analysis time
    "run55_evaluation_commit": "284e2ac8ec8640d73679c55555772eeda2505485",
    "run55_criteria_commit": "c5d34a751d00b9245a43d34e98694ce1e94de1dc",
    "run55_workflow_run": 36226568240,
    "research_schema_version": OBSERVATION_SCHEMA_VERSION,
    "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
}

# ---- Pre-registered gates (Part 4) ------------------------------------------
GATES = {
    "A_trading_days": {"min": 10, "preferred": 20},
    "B_scan_runs": {"min": 50, "preferred": 100},
    "C_cohort_clusters": {"min": 30},
    "D_horizon_maturation": {"short_horizons": 0.80, "long_horizon": 0.70,
                             "cohort_neutral_gap_pp": 5.0},
    "E_maturation_parity": {"max_gap_pp": 10.0, "preferred_gap_pp": 5.0},
    "F_directional_integrity": {"min_coverage": 0.90, "min_outcomes_to_judge": 100},
    "G_effective_clusters": {"min": 20, "preferred": 30},  # Run 55 STRONG = 20 clusters
    "H_research_integrity": {"max_minor_overlap_rate": 0.01},
}
SAMPLE_GATES = ("A_trading_days", "B_scan_runs", "C_cohort_clusters", "G_effective_clusters")
# A parity / coverage failure is attributed to the pipeline only once it is
# measurable: every cohort has this many settled observations at the horizon,
# drawn from at least this many scan runs. Below that it is attributed to time.
PARITY_MEASURABLE = {"min_settled_per_cohort": 100, "min_runs_per_cohort": 5}
MATURATION_SLACK = _dt.timedelta(minutes=15)   # worker default --slack-min
MATURATION_GRACE = _dt.timedelta(minutes=45)   # >= one 30-min maturation cycle
RETIRE_AFTER = _dt.timedelta(days=6)           # scripts.mature_observations.RETIRE_AFTER
APPROACHING_PROGRESS = 0.75
SHORT_MIN_CLUSTERS = 30

UNMATURED_REASONS = ("NOT_YET_ELIGIBLE", "PRICE_DATA_UNAVAILABLE", "INSUFFICIENT_FUTURE_BARS",
                     "RATE_LIMITED", "RETIRED", "FILTERED_BY_POLICY", "OTHER")

# Run 55 baseline (from artifacts/research/run55_signal_effectiveness.json + doc).
RUN55_BASELINE = {
    "trading_days": 3,
    "scan_runs_all": 16,
    "scan_runs_regular": 13,
    "candidate_60m_maturation_pct": 47.6,   # 571 / 1,200 regular-session candidates
    "control_60m_maturation_pct": 8.6,      # 112 / 1,300 regular-session controls
    "parity_gap_60m_pp": 39.0,
    "candidate_control_clusters_60m": 9,
    "short_clusters": 0,
    "directional_coverage_pct": 54.7,
}

# Anti-peeking: no key in the report may look like an effectiveness statistic.
_FORBIDDEN_KEY = re.compile(
    r"(win|hit_rate|mean|median|avg|average|spearman|pearson|corr|payoff|lift|edge|"
    r"best|optimal|sharpe|pnl|profit|return_diff|diff_|excess|alpha|expectancy|"
    r"raw_return|directional_return$|mfe$|mae$|drawdown)", re.I)


def parity_class(gap_pp: Optional[float]) -> str:
    """Run 58 completeness classes for a maturation parity gap (no performance info)."""
    if gap_pp is None:
        return "NOT_MEASURABLE"
    if gap_pp <= 5:
        return "HEALTHY"
    if gap_pp <= 10:
        return "ACCEPTABLE"
    if gap_pp <= 20:
        return "WARNING"
    return "CRITICAL"


# ---- Helpers -----------------------------------------------------------------
def _parse_dt(v: Any) -> Optional[_dt.datetime]:
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _et(d: _dt.datetime) -> _dt.datetime:
    try:
        from zoneinfo import ZoneInfo
        return d.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover
        return d.astimezone(_dt.timezone(_dt.timedelta(hours=-4)))


def _present(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def epoch_start() -> _dt.datetime:
    return _parse_dt(FORWARD_EPOCH["forward_epoch_start_timestamp"])


def cohort(rec: Dict[str, Any]) -> Optional[str]:
    c = rec.get("research_cohort") or (rec.get("market_context") or {}).get("research_cohort")
    return str(c).upper() if c else None


def anchor(rec: Dict[str, Any]) -> Optional[_dt.datetime]:
    return _parse_dt(rec.get("scan_timestamp") or rec.get("timestamp"))


def scan_run(rec: Dict[str, Any]) -> str:
    mc = rec.get("market_context") or {}
    return str(mc.get("scan_id") or rec.get("scan_timestamp") or rec.get("timestamp") or "")


def direction(rec: Dict[str, Any]) -> Optional[str]:
    for s in rec.get("scanners") or []:
        v = str((s or {}).get("direction") or "").strip().lower()
        if v:
            return "SHORT" if v in {"short", "bearish", "sell"} else "LONG"
    return None


def is_regular_session(d: Optional[_dt.datetime]) -> bool:
    if d is None:
        return False
    e = _et(d)
    m = e.hour * 60 + e.minute
    return e.weekday() < 5 and 570 <= m < 960


def trading_date(d: _dt.datetime) -> _dt.date:
    return _et(d).date()


def is_completed_trading_day(day: _dt.date, now: _dt.datetime) -> bool:
    """A trading day is complete once its last +60m horizon (15:35 slot) has had
    time to mature: 17:15 ET on that day."""
    n = _et(now)
    return day < n.date() or (day == n.date() and n.hour * 60 + n.minute >= 17 * 60 + 15)


def scoring_versions(rec: Dict[str, Any]) -> Tuple:
    v = rec.get("versions") or {}
    scanner = next((s.get("version") for s in rec.get("scanners") or []
                    if isinstance(s, dict) and s.get("name") == "breakout"), None)
    return (scanner, v.get("hsf_score"), v.get("prebreakout_model"))


def _exclusion(symbol: str) -> Optional[str]:
    try:
        from data.us_market_universe import symbol_exclusion_reason
        return symbol_exclusion_reason(symbol)
    except Exception:
        return None


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, 2) if den else None


# ---- Core ------------------------------------------------------------------------
def select_forward(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Explicitly tagged, deduplicated observations anchored at/after the epoch."""
    start = epoch_start()
    seen: Dict[str, str] = {}
    fwd: List[Dict[str, Any]] = []
    pre_epoch = legacy = dup_ids = conflicting_ids = missing_anchor = 0
    for rec in observations or []:
        if not rec:
            continue
        a = anchor(rec)
        c = cohort(rec)
        if c is None:
            legacy += 1
            continue
        if a is None:
            missing_anchor += 1
            continue
        if a < start:
            pre_epoch += 1
            continue
        oid = str(rec.get("observation_id") or "")
        body = repr(sorted((k, repr(v)) for k, v in rec.items()))
        if oid in seen:
            dup_ids += 1
            conflicting_ids += seen[oid] != body
            continue
        seen[oid] = body
        fwd.append(rec)
    return {"forward": fwd, "pre_epoch_excluded": pre_epoch, "legacy_excluded": legacy,
            "duplicate_observation_ids": dup_ids, "conflicting_observation_ids": conflicting_ids,
            "missing_anchor": missing_anchor}


def monitor(observations: Sequence[Dict[str, Any]],
            outcomes_by_id: Dict[str, List[Dict[str, Any]]], *,
            now: Optional[_dt.datetime] = None,
            maturation_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    outcomes_by_id = outcomes_by_id or {}
    sel = select_forward(observations)
    fwd = sel["forward"]
    regular = [o for o in fwd if is_regular_session(anchor(o))]

    # Outcome presence per (observation, horizon); integrity via equality only.
    matured: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dup_outcomes = conflicting_outcomes = pit = transform_mismatch = 0
    fwd_ids = {str(o.get("observation_id") or ""): o for o in fwd}
    for oid, rec in fwd_ids.items():
        per_h: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for oc in outcomes_by_id.get(oid) or []:
            if str(oc.get("data_status")) == "MATURED" and oc.get("horizon") in HORIZONS:
                per_h[oc["horizon"]].append(oc)
        a, d = anchor(rec), direction(rec)
        for h, ocs in per_h.items():
            if len(ocs) > 1:
                dup_outcomes += len(ocs) - 1
                conflicting_outcomes += len({repr(x.get("raw_return")) for x in ocs}) > 1
            oc = ocs[0]
            matured[(oid, h)] = oc
            ev = _parse_dt(oc.get("evaluation_time"))
            if a is not None and ev is not None and ev <= a:
                pit += 1
            raw, dr = oc.get("raw_return"), oc.get("directional_return")
            if _present(raw) and _present(dr) and d is not None:
                expected = -float(raw) if d == "SHORT" else float(raw)
                transform_mismatch += abs(float(dr) - expected) > 1e-6

    # ---- Time + scan coverage --------------------------------------------------
    anchors = [anchor(o) for o in fwd]
    days = sorted({trading_date(anchor(o)) for o in regular})
    completed_days = [d for d in days if is_completed_trading_day(d, now)]
    runs_all = {scan_run(o) for o in fwd}
    runs_regular = sorted({scan_run(o) for o in regular},
                          key=lambda r: min(anchor(o) for o in regular if scan_run(o) == r))
    run_cohorts: Dict[str, set] = defaultdict(set)
    for o in regular:
        run_cohorts[scan_run(o)].add(cohort(o))
    successful = [r for r in runs_regular if set(COHORTS) <= run_cohorts[r]]
    time_cov = {
        "forward_trading_days": len(days),
        "completed_forward_trading_days": len(completed_days),
        "calendar_days": ((max(anchors) - min(anchors)).days + 1) if anchors else 0,
        "first_forward_observation": min(anchors).isoformat() if anchors else None,
        "latest_forward_observation": max(anchors).isoformat() if anchors else None,
        "trading_dates": [d.isoformat() for d in days],
    }
    scan_cov = {
        "total_forward_scan_runs": len(runs_all),
        "regular_session_scan_runs": len(runs_regular),
        "successful_scan_runs": len(successful),
        "successful_definition": "regular-session run containing all three cohorts",
        "unique_symbols": len({str(o.get("symbol") or "").upper() for o in fwd}),
        "forward_observations_all_sessions": len(fwd),
        "forward_observations_regular_session": len(regular),
    }

    # ---- Cohorts / horizons (regular session = Run 55 primary population) -------
    cohorts_out: Dict[str, Any] = {}
    horizons_out: Dict[str, Any] = {h: {} for h in HORIZONS}
    reasons_out: Dict[str, Dict[str, Counter]] = {h: {c: Counter() for c in COHORTS} for h in HORIZONS}
    clusters: Dict[str, Dict[str, set]] = {h: {c: set() for c in COHORTS} for h in HORIZONS}
    settled_runs: Dict[str, Dict[str, set]] = {h: {c: set() for c in COHORTS} for h in HORIZONS}
    field_cov = {"outcomes": 0, "directional_return": 0, "mfe": 0, "mae": 0}
    for c in COHORTS:
        rows = [o for o in regular if cohort(o) == c]
        any_matured = sum(1 for o in rows if any((str(o.get("observation_id")), h) in matured for h in HORIZONS))
        cohorts_out[c] = {
            "observations": len(rows),
            "unique_symbols": len({str(o.get("symbol") or "").upper() for o in rows}),
            "unique_scan_runs": len({scan_run(o) for o in rows}),
            "matured_observations": any_matured,
            "unmatured_observations": len(rows) - any_matured,
        }
        for h in HORIZONS:
            settled = matured_n = dr_n = mfe_n = mae_n = 0
            for o in rows:
                oid = str(o.get("observation_id") or "")
                a = anchor(o)
                oc = matured.get((oid, h))
                is_settled = now >= a + _dt.timedelta(minutes=HORIZON_MIN[h]) + MATURATION_SLACK + MATURATION_GRACE
                if oc is not None:
                    matured_n += 1
                    settled += 1
                    settled_runs[h][c].add(scan_run(o))
                    clusters[h][c].add(scan_run(o))
                    dr_n += _present(oc.get("directional_return"))
                    mfe_n += _present(oc.get("mfe"))
                    mae_n += _present(oc.get("mae"))
                    continue
                if not is_settled:
                    reasons_out[h][c]["NOT_YET_ELIGIBLE"] += 1
                    continue
                settled += 1
                settled_runs[h][c].add(scan_run(o))
                reasons_out[h][c][_unmatured_reason(o, h, matured, now)] += 1
            field_cov["outcomes"] += matured_n
            field_cov["directional_return"] += dr_n
            field_cov["mfe"] += mfe_n
            field_cov["mae"] += mae_n
            horizons_out[h][c] = {
                "observations": len(rows),
                "eligible_observations": settled,
                "matured_observations": matured_n,
                "maturation_pct": _pct(matured_n, settled),
                "directional_return_coverage_pct": _pct(dr_n, matured_n),
                "mfe_coverage_pct": _pct(mfe_n, matured_n),
                "mae_coverage_pct": _pct(mae_n, matured_n),
                "matured_scan_run_clusters": len(clusters[h][c]),
                "unmatured_reasons": {r: reasons_out[h][c].get(r, 0) for r in UNMATURED_REASONS},
            }

    # ---- Parity (Part 3) --------------------------------------------------------
    parity: Dict[str, Any] = {}
    for h in HORIZONS:
        pcts = {c: horizons_out[h][c]["maturation_pct"] for c in COHORTS}
        known = [p for p in pcts.values() if p is not None]
        gap = round(max(known) - min(known), 2) if len(known) == len(COHORTS) else None
        measurable = all(horizons_out[h][c]["eligible_observations"] >= PARITY_MEASURABLE["min_settled_per_cohort"]
                         and len(settled_runs[h][c]) >= PARITY_MEASURABLE["min_runs_per_cohort"]
                         for c in COHORTS)
        parity[h] = {**{f"{c.lower()}_maturation_pct": pcts[c] for c in COHORTS},
                     "maturation_parity_gap": gap, "measurable": measurable,
                     "parity_classification": parity_class(gap)}

    # ---- Directions (Part 5) ------------------------------------------------------
    dirs: Dict[str, Any] = {}
    for label, pred in (("LONG", lambda o: direction(o) == "LONG"),
                        ("SHORT", lambda o: direction(o) == "SHORT"),
                        ("CONTROL_UNDIRECTED", lambda o: direction(o) is None)):
        rows = [o for o in regular if pred(o)]
        mat = [o for o in rows if (str(o.get("observation_id")), "+60m") in matured]
        dirs[label] = {"observations": len(rows), "matured_observations_60m": len(mat),
                       "scan_runs": len({scan_run(o) for o in rows}),
                       "matured_scan_run_clusters_60m": len({scan_run(o) for o in mat})}

    # ---- Data quality ---------------------------------------------------------------
    overlaps = Counter()
    by_run_sym: Dict[Tuple[str, str], set] = defaultdict(set)
    for o in fwd:
        by_run_sym[(scan_run(o), str(o.get("symbol") or "").upper())].add(cohort(o))
    for cs in by_run_sym.values():
        if CANDIDATE in cs and CONTROL in cs:
            overlaps["candidate_control"] += 1
        if CANDIDATE in cs and NEAR_MISS in cs:
            overlaps["candidate_near_miss"] += 1
        if NEAR_MISS in cs and CONTROL in cs:
            overlaps["near_miss_control"] += 1
    # Only scored rows carry a scanner version; CONTROL rows (no scanner) are not drift.
    versions = Counter(scoring_versions(o) for o in fwd if scoring_versions(o)[0] is not None)
    invalid = sum(1 for o in fwd if _invalid_market(o))
    start = epoch_start()
    orphans = 0
    for k, v in outcomes_by_id.items():
        if k in fwd_ids:
            continue
        for x in v:
            ts = _parse_dt(x.get("observation_timestamp"))
            orphans += ts is not None and ts >= start
    dq = {
        "pre_epoch_observations_excluded": sel["pre_epoch_excluded"],
        "legacy_untagged_excluded": sel["legacy_excluded"],
        "missing_anchor_excluded": sel["missing_anchor"],
        "duplicate_observation_ids": sel["duplicate_observation_ids"],
        "conflicting_observation_ids": sel["conflicting_observation_ids"],
        "duplicate_outcomes": dup_outcomes,
        "conflicting_outcomes": conflicting_outcomes,
        "orphan_forward_outcomes": orphans,
        "point_in_time_violations": pit,
        "direction_transform_mismatches": transform_mismatch,
        "invalid_market_values": invalid,
        "cohort_overlap": dict(overlaps),
        "scoring_versions": [{"breakout_scanner": k[0], "hsf_score": k[1], "prebreakout_model": k[2],
                              "observations": n} for k, n in sorted(versions.items(), key=lambda kv: repr(kv[0]))],
        "scoring_version_drift": len(versions) > 1,
        "maturation_run_report": _maturation_run_summary(maturation_report),
        # Run 58: the scheduler is cohort-neutral only while its symbol cap does not
        # bind (every ready symbol processed). Surfaced from the latest scheduled run.
        "maturation_capacity_binding": (
            None if not isinstance(maturation_report, dict) or "symbols_deferred" not in maturation_report
            else bool(maturation_report.get("symbols_deferred"))),
        "unmatured_reason_note": ("per-observation failure reasons are not persisted; PRICE_DATA_UNAVAILABLE "
                                  "and INSUFFICIENT_FUTURE_BARS are inferred, RATE_LIMITED is only visible "
                                  "run-level in maturation_run_report"),
    }

    directional_cov = {k: _pct(field_cov[k], field_cov["outcomes"])
                       for k in ("directional_return", "mfe", "mae")}
    gates = evaluate_gates(time_cov, scan_cov, horizons_out, parity, clusters, dq,
                           directional_cov, field_cov["outcomes"], len(regular))
    progress = progress_metrics(time_cov, scan_cov, horizons_out, parity, clusters)
    state = decide_state(gates, parity, len(regular), progress)
    estimate = estimate_days(gates, state, time_cov, scan_cov, clusters, progress)
    estimate.setdefault("sample_gates_value", estimate["value"])
    long_r, short_r = direction_readiness(state, dirs, time_cov)
    epoch = {**FORWARD_EPOCH,
             "forward_epoch_start_scan_run": runs_regular[0] if runs_regular else (
                 sorted(runs_all)[0] if runs_all else None),
             "scanner_scoring_version": dq["scoring_versions"][0] if dq["scoring_versions"] else None}
    report = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "epoch": epoch,
        "state": state["state"],
        "state_reason": state["reason"],
        "limiting_factor": state["limiting_factor"],
        "RUN55_RERUN_RECOMMENDED": state["state"] == "READY_FOR_RUN55_RERUN",
        "long_readiness": long_r,
        "short_readiness": short_r,
        "gates": gates,
        "time_coverage": time_cov,
        "scan_coverage": scan_cov,
        "cohorts": cohorts_out,
        "horizons": horizons_out,
        "directions": dirs,
        "maturation_parity": parity,
        "directional_field_coverage_pct": directional_cov,
        "data_quality": dq,
        "progress": progress,
        "estimated_trading_days_until_ready": estimate["value"],
        "estimate_basis": estimate["basis"],
        "estimated_trading_days_until_sample_gates": estimate["sample_gates_value"],
        "metadata_completeness": metadata_completeness(regular),
        "run55_comparison": run55_comparison(time_cov, scan_cov, horizons_out, parity, clusters,
                                             dirs, directional_cov),
        "pre_registered_gates": GATES,
        "anti_peeking": ("Evidence quantity/quality only. No win rate, return, cohort difference, "
                         "correlation, bucket or threshold statistics are computed or reported."),
    }
    assert_no_effectiveness_metrics(report)
    return report


def metadata_completeness(regular: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Run 57 point-in-time metadata COMPLETENESS per cohort (informational, not a
    gate). Presence counts only; metadata values are never related to outcomes."""
    out: Dict[str, Any] = {}
    for c in COHORTS:
        rows = [o for o in regular if cohort(o) == c]
        n = len(rows)

        def cov(field: str) -> Optional[float]:
            k = sum(1 for o in rows if (o.get("research_metadata") or {}).get(field) not in (None, ""))
            return _pct(k, n)
        out[c] = {"observations": n,
                  "metadata_block_coverage_pct": _pct(sum(1 for o in rows if o.get("research_metadata")), n),
                  "tier_metadata_coverage_pct": cov("tier_at_observation"),
                  "regime_metadata_coverage_pct": cov("market_regime_at_observation"),
                  "scoring_version_coverage_pct": cov("scoring_version"),
                  "commit_sha_coverage_pct": cov("scanner_commit_sha"),
                  "provider_coverage_pct": cov("price_provider")}
    return out


def _invalid_market(o: Dict[str, Any]) -> bool:
    mkt = o.get("market") or {}
    for k in ("price", "volume"):
        v = mkt.get(k)
        if v is None:
            continue
        if not _present(v) or float(v) < 0:
            return True
    return False


def _unmatured_reason(o: Dict[str, Any], h: str, matured, now: _dt.datetime) -> str:
    """Why a settled (time-eligible) horizon has no outcome. Inferred, see note."""
    if _exclusion(str(o.get("symbol") or "")):
        return "FILTERED_BY_POLICY"
    a = anchor(o)
    if a is not None and now >= a + RETIRE_AFTER:
        return "RETIRED"
    oid = str(o.get("observation_id") or "")
    if any((oid, other) in matured for other in HORIZONS):
        return "INSUFFICIENT_FUTURE_BARS"
    if a is not None and now >= a + _dt.timedelta(days=1):
        return "PRICE_DATA_UNAVAILABLE"
    return "OTHER"


def _maturation_run_summary(r: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(r, dict):
        return None
    keep = ("generated_at", "schema", "alpaca_requests", "alpaca_429_count", "rate_limited_symbols",
            "price_data_unavailable_symbols", "provider_error_symbols", "symbols_processed",
            "symbols_deferred", "outcomes_matured", "dry_run")
    out = {k: r.get(k) for k in keep if k in r}
    out["retired"] = r.get("retired")
    out["failures"] = r.get("failures")
    return out


# ---- Gates / state ----------------------------------------------------------------
def _gate(status: str, detail: str, **extra) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


def _tiered(value: float, minimum: float, preferred: Optional[float]) -> str:
    if preferred is not None and value >= preferred:
        return "PASS"
    if value >= minimum:
        return "PASS" if preferred is None else "WARN"
    return "FAIL"


def evaluate_gates(time_cov, scan_cov, horizons_out, parity, clusters, dq, directional_cov,
                   matured_outcomes: int, regular_n: int) -> Dict[str, Any]:
    g: Dict[str, Any] = {}
    a = GATES["A_trading_days"]
    days = time_cov["completed_forward_trading_days"]
    g["A_trading_days"] = _gate(_tiered(days, a["min"], a["preferred"]),
                                f"{days} completed forward trading days (min {a['min']}, preferred {a['preferred']})",
                                value=days)
    b = GATES["B_scan_runs"]
    runs = scan_cov["regular_session_scan_runs"]
    g["B_scan_runs"] = _gate(_tiered(runs, b["min"], b["preferred"]),
                             f"{runs} regular-session forward scan runs (min {b['min']}, preferred {b['preferred']})",
                             value=runs)
    cmin = GATES["C_cohort_clusters"]["min"]
    per = {h: {c: len(clusters[h][c]) for c in COHORTS} for h in HORIZONS}
    worst = min((per[h][c], h, c) for h in HORIZONS for c in COHORTS)
    g["C_cohort_clusters"] = _gate("PASS" if worst[0] >= cmin else "FAIL",
                                   f"min matured scan-run clusters = {worst[0]} ({worst[2]} at {worst[1]}); need >= {cmin} "
                                   "for every cohort at every horizon", value=worst[0], by_horizon=per)
    d = GATES["D_horizon_maturation"]
    d_detail, d_status = [], "PASS"
    for h in HORIZONS:
        pcts = [horizons_out[h][c]["maturation_pct"] for c in COHORTS]
        low = min((p for p in pcts if p is not None), default=None)
        need = 100 * (d["long_horizon"] if h == "+60m" else d["short_horizons"])
        if low is None:
            st = "FAIL"
        elif low >= need:
            st = "PASS"
        elif h == "+60m" and parity[h]["maturation_parity_gap"] is not None and \
                parity[h]["maturation_parity_gap"] <= d["cohort_neutral_gap_pp"]:
            st = "WARN"
        else:
            st = "FAIL"
        shown = "no eligible observations yet" if low is None else f"min cohort {low}%"
        d_detail.append(f"{h}: {shown} (need {need:.0f}%) → {st}")
        d_status = _worse(d_status, st)
    g["D_horizon_maturation"] = _gate(d_status, "; ".join(d_detail))
    e = GATES["E_maturation_parity"]
    e_detail, e_status = [], "PASS"
    for h in HORIZONS:
        gap = parity[h]["maturation_parity_gap"]
        st = "FAIL" if gap is None else _tiered(-gap, -e["max_gap_pp"], -e["preferred_gap_pp"])
        shown = "not measurable yet (a cohort has no eligible observations)" if gap is None else f"gap {gap}pp"
        e_detail.append(f"{h}: {shown} → {st}")
        e_status = _worse(e_status, st)
    g["E_maturation_parity"] = _gate(e_status, "; ".join(e_detail))
    f = GATES["F_directional_integrity"]
    cov = [directional_cov[k] for k in ("directional_return", "mfe", "mae")]
    low = min((c for c in cov if c is not None), default=None)
    if matured_outcomes < f["min_outcomes_to_judge"]:
        f_st = "FAIL" if matured_outcomes == 0 else ("PASS" if low is not None and low >= 100 * f["min_coverage"] else "FAIL")
        f_detail = f"{matured_outcomes} matured forward outcomes (need >= {f['min_outcomes_to_judge']} to judge); coverage {directional_cov}"
    else:
        f_st = "PASS" if low is not None and low >= 100 * f["min_coverage"] else "FAIL"
        f_detail = f"directional/MFE/MAE coverage {directional_cov} (need >= {100 * f['min_coverage']:.0f}%)"
    g["F_directional_integrity"] = _gate(f_st, f_detail, judged=matured_outcomes >= f["min_outcomes_to_judge"])
    gg = GATES["G_effective_clusters"]
    pairs = {}
    for label, other in (("candidate_vs_control", CONTROL), ("candidate_vs_near_miss", NEAR_MISS)):
        pairs[label] = {h: len(clusters[h][CANDIDATE] & clusters[h][other]) for h in HORIZONS}
    gmin = min(v for p in pairs.values() for v in p.values())
    g["G_effective_clusters"] = _gate(_tiered(gmin, gg["min"], gg["preferred"]),
                                      f"min paired scan-run clusters = {gmin} (min {gg['min']} = Run 55 STRONG, "
                                      f"preferred {gg['preferred']}); observations within one run are not independent",
                                      value=gmin, by_comparison=pairs)
    hard, soft = [], []
    for k in ("point_in_time_violations", "conflicting_outcomes", "conflicting_observation_ids",
              "direction_transform_mismatches", "invalid_market_values"):
        if dq[k]:
            hard.append(f"{k}={dq[k]}")
    if dq["cohort_overlap"].get("candidate_control"):
        hard.append(f"candidate/control overlap={dq['cohort_overlap']['candidate_control']}")
    minor = dq["cohort_overlap"].get("near_miss_control", 0) + dq["cohort_overlap"].get("candidate_near_miss", 0)
    if regular_n and minor / max(regular_n, 1) > GATES["H_research_integrity"]["max_minor_overlap_rate"]:
        hard.append(f"near-miss overlaps={minor} (> {GATES['H_research_integrity']['max_minor_overlap_rate']:.0%})")
    elif minor:
        soft.append(f"near-miss overlaps={minor} (deduped by Run 55)")
    if dq["scoring_version_drift"]:
        soft.append("scoring version changed during the forward epoch (scanner is meant to be frozen)")
    if dq["duplicate_outcomes"]:
        soft.append(f"duplicate identical outcomes={dq['duplicate_outcomes']}")
    h_st = "FAIL" if hard else ("WARN" if soft else "PASS")
    g["H_research_integrity"] = _gate(h_st, "; ".join(hard + soft) or "no integrity issues found")
    return g


_ORDER = {"PASS": 0, "NOT_APPLICABLE": 0, "WARN": 1, "FAIL": 2}


def _worse(a: str, b: str) -> str:
    return a if _ORDER[a] >= _ORDER[b] else b


def decide_state(gates: Dict[str, Any], parity: Dict[str, Any], regular_n: int,
                 progress: Dict[str, Any]) -> Dict[str, str]:
    """Pre-registered state rules.

    DATA_QUALITY_BLOCKED: H fails; or F fails once judged; or D/E fail on a horizon
      whose parity is measurable (enough settled observations and runs per cohort),
      i.e. more time will not fix it.
    READY_FOR_RUN55_RERUN: no gate FAILs.
    APPROACHING_READY: only sample-size gates (A/B/C/G), or D/E on not-yet-measurable
      horizons, fail, and the bottleneck progress is >= 75%.
    COLLECTING: otherwise (healthy but insufficient)."""
    if regular_n == 0:
        return {"state": "COLLECTING", "limiting_factor": "NO_FORWARD_DATA",
                "reason": "no regular-session forward observations yet"}
    failed = [k for k, v in gates.items() if v["status"] == "FAIL"]
    measurable_bias = [h for h in HORIZONS if parity[h]["measurable"] and (
        parity[h]["maturation_parity_gap"] is None
        or parity[h]["maturation_parity_gap"] > GATES["E_maturation_parity"]["max_gap_pp"])]
    blocked = []
    if "H_research_integrity" in failed:
        blocked.append("H_research_integrity")
    if "F_directional_integrity" in failed and gates["F_directional_integrity"].get("judged"):
        blocked.append("F_directional_integrity")
    if measurable_bias and ({"D_horizon_maturation", "E_maturation_parity"} & set(failed)):
        blocked.append(f"maturation parity at measurable horizons {measurable_bias}")
    if blocked:
        return {"state": "DATA_QUALITY_BLOCKED", "limiting_factor": "DATA_PIPELINE_BIAS"
                if measurable_bias else "DATA_INTEGRITY",
                "reason": "blocked by " + ", ".join(blocked)}
    if not failed:
        return {"state": "READY_FOR_RUN55_RERUN", "limiting_factor": "NONE",
                "reason": "all pre-registered gates pass (WARNs listed)"}
    time_related = set(SAMPLE_GATES) | {"D_horizon_maturation", "E_maturation_parity",
                                        "F_directional_integrity"}
    if set(failed) <= time_related and progress["sample_size_progress"] >= APPROACHING_PROGRESS:
        return {"state": "APPROACHING_READY", "limiting_factor": "NOT_ENOUGH_TIME",
                "reason": (f"only time/sample gates fail ({', '.join(failed)}); bottleneck "
                           f"{progress['bottleneck']} at {progress['sample_size_progress']:.0%}")}
    return {"state": "COLLECTING", "limiting_factor": "NOT_ENOUGH_TIME",
            "reason": (f"healthy but insufficient: {', '.join(failed)} fail; bottleneck "
                       f"{progress['bottleneck']} at {progress['sample_size_progress']:.0%}")}


def progress_metrics(time_cov, scan_cov, horizons_out, parity, clusters) -> Dict[str, Any]:
    cap = (lambda x: round(min(1.0, max(0.0, x)), 4))
    cmin = GATES["C_cohort_clusters"]["min"]
    p = {
        "trading_days_progress": cap(time_cov["completed_forward_trading_days"] / GATES["A_trading_days"]["min"]),
        "scan_runs_progress": cap(scan_cov["regular_session_scan_runs"] / GATES["B_scan_runs"]["min"]),
    }
    for c, key in ((CANDIDATE, "candidate"), (NEAR_MISS, "near_miss"), (CONTROL, "control")):
        p[f"{key}_cluster_progress"] = cap(min(len(clusters[h][c]) for h in HORIZONS) / cmin)
    cov = []
    for h in HORIZONS:
        need = 100 * (GATES["D_horizon_maturation"]["long_horizon"] if h == "+60m"
                      else GATES["D_horizon_maturation"]["short_horizons"])
        low = min((horizons_out[h][c]["maturation_pct"] for c in COHORTS
                   if horizons_out[h][c]["maturation_pct"] is not None), default=0.0)
        cov.append(low / need)
    p["horizon_coverage_progress"] = cap(min(cov) if cov else 0.0)
    gaps = [parity[h]["maturation_parity_gap"] for h in HORIZONS]
    lim = GATES["E_maturation_parity"]["max_gap_pp"]
    p["maturation_parity_progress"] = cap(min((1.0 if g is not None and g <= lim else
                                               (lim / g if g else 0.0)) for g in gaps))
    time_keys = ("trading_days_progress", "scan_runs_progress", "candidate_cluster_progress",
                 "near_miss_cluster_progress", "control_cluster_progress")
    p["bottleneck"] = min(time_keys, key=lambda k: p[k])
    p["sample_size_progress"] = p[p["bottleneck"]]
    return p


def estimate_days(gates, state, time_cov, scan_cov, clusters, progress) -> Dict[str, Any]:
    """Trading days until every sample-size gate reaches its minimum, from
    observed per-completed-day collection rates. UNKNOWN when not defensible."""
    if state["state"] == "READY_FOR_RUN55_RERUN":
        return {"value": 0, "basis": "all gates pass"}
    if state["state"] == "DATA_QUALITY_BLOCKED":
        return {"value": "UNKNOWN", "basis": "blocked by data quality; more time will not clear it"}
    sample = sample_gate_days(gates, time_cov, scan_cov)
    non_sample = [k for k, v in gates.items() if v["status"] == "FAIL" and k not in SAMPLE_GATES]
    if non_sample:
        return {"value": "UNKNOWN", "sample_gates_value": sample["value"],
                "basis": f"non-sample gates failing ({', '.join(non_sample)}); their rate of improvement "
                         f"is not a collection rate. Sample-size gates alone: {sample['value']} ({sample['basis']})"}
    return {**sample, "sample_gates_value": sample["value"]}


def sample_gate_days(gates, time_cov, scan_cov) -> Dict[str, Any]:
    """Trading days until gates A/B/C/G reach their minimums at observed rates."""
    days = time_cov["completed_forward_trading_days"]
    if days < 2:
        return {"value": "UNKNOWN", "basis": f"only {days} completed forward trading day(s); "
                                             "need >= 2 to measure collection rates"}
    need = []
    need.append(GATES["A_trading_days"]["min"] - days)
    runs = scan_cov["regular_session_scan_runs"]
    rate = runs / days
    need.append(math.ceil((GATES["B_scan_runs"]["min"] - runs) / rate) if rate > 0 else math.inf)
    for g_key, target in (("C_cohort_clusters", GATES["C_cohort_clusters"]["min"]),
                          ("G_effective_clusters", GATES["G_effective_clusters"]["min"])):
        v = gates[g_key]["value"]
        r = v / days
        need.append(math.ceil((target - v) / r) if r > 0 else math.inf)
    worst = max(0, max(need))
    if math.isinf(worst):
        return {"value": "UNKNOWN", "basis": "a required cluster count is not growing"}
    return {"value": int(worst), "basis": f"linear extrapolation of {days} completed trading days "
                                          "(runs and matured clusters per day); assumes rates hold"}


def direction_readiness(state: Dict[str, str], dirs: Dict[str, Any], time_cov) -> Tuple[str, str]:
    if state["state"] == "READY_FOR_RUN55_RERUN":
        long_r = "READY"
    elif state["state"] == "DATA_QUALITY_BLOCKED":
        long_r = "INSUFFICIENT"
    else:
        long_r = "COLLECTING"
    s = dirs["SHORT"]
    if s["observations"] == 0:
        short_r = "INSUFFICIENT"
    elif (s["matured_scan_run_clusters_60m"] >= SHORT_MIN_CLUSTERS
          and time_cov["completed_forward_trading_days"] >= GATES["A_trading_days"]["min"]
          and state["state"] != "DATA_QUALITY_BLOCKED"):
        short_r = "READY"
    else:
        short_r = "COLLECTING"
    return long_r, short_r


def run55_comparison(time_cov, scan_cov, horizons_out, parity, clusters, dirs, directional_cov):
    b = RUN55_BASELINE
    return [
        {"metric": "Trading days", "run55": b["trading_days"],
         "forward": time_cov["completed_forward_trading_days"], "gate": ">= 10 (pref 20)"},
        {"metric": "Scan runs (regular session)", "run55": b["scan_runs_regular"],
         "forward": scan_cov["regular_session_scan_runs"], "gate": ">= 50 (pref 100)"},
        {"metric": "+60m CANDIDATE maturation %", "run55": b["candidate_60m_maturation_pct"],
         "forward": horizons_out["+60m"][CANDIDATE]["maturation_pct"], "gate": ">= 70%"},
        {"metric": "+60m CONTROL maturation %", "run55": b["control_60m_maturation_pct"],
         "forward": horizons_out["+60m"][CONTROL]["maturation_pct"], "gate": ">= 70%"},
        {"metric": "+60m parity gap (pp)", "run55": b["parity_gap_60m_pp"],
         "forward": parity["+60m"]["maturation_parity_gap"], "gate": "<= 10 (pref 5)"},
        {"metric": "+60m CANDIDATE∩CONTROL clusters", "run55": b["candidate_control_clusters_60m"],
         "forward": len(clusters["+60m"][CANDIDATE] & clusters["+60m"][CONTROL]), "gate": ">= 20 (pref 30)"},
        {"metric": "LONG matured clusters (+60m)", "run55": None,
         "forward": dirs["LONG"]["matured_scan_run_clusters_60m"], "gate": ">= 30"},
        {"metric": "SHORT matured clusters (+60m)", "run55": b["short_clusters"],
         "forward": dirs["SHORT"]["matured_scan_run_clusters_60m"], "gate": "separate"},
        {"metric": "Directional/MFE/MAE coverage %", "run55": b["directional_coverage_pct"],
         "forward": min((v for v in directional_cov.values() if v is not None), default=None),
         "gate": ">= 90%"},
    ]


# ---- Anti-peeking guard ---------------------------------------------------------
_ALLOWED_KEYS = {"directional_return_coverage_pct", "directional_return", "mfe", "mae",
                 "mfe_coverage_pct", "mae_coverage_pct"}


def forbidden_keys(obj: Any, path: str = "") -> List[str]:
    """Every dict key anywhere in `obj` that looks like an effectiveness metric."""
    bad: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            coverage_ctx = path.endswith("directional_field_coverage_pct")
            if ks not in _ALLOWED_KEYS and _FORBIDDEN_KEY.search(ks):
                bad.append(f"{path}.{ks}")
            elif ks in {"directional_return", "mfe", "mae"} and not coverage_ctx:
                bad.append(f"{path}.{ks}")
            bad += forbidden_keys(v, f"{path}.{ks}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += forbidden_keys(v, f"{path}[{i}]")
    return bad


def assert_no_effectiveness_metrics(report: Dict[str, Any]) -> None:
    bad = forbidden_keys(report)
    if bad:
        raise ValueError(f"anti-peeking contract violated: {bad[:5]}")
