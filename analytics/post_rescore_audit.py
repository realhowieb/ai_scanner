"""Read-only post-repair diagnostics; no fitting, score changes or label synthesis.

Primary descriptive outcome is the existing ML v3 return_h > 0 definition.
That is NOT either production model's training target. Any probability metrics
below are explicitly proxy diagnostics, not certified model calibration.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from analytics import ml_readiness as mr
from analytics import ml_v3_audit as audit
from analytics import research_dataset as rd

REPAIR_START = dt.datetime(2026, 10, 9, 5, 12, tzinfo=dt.timezone.utc)
REPAIR_END = dt.datetime(2026, 10, 9, 5, 16, tzinfo=dt.timezone.utc)
EXPECTED = 613


def number(value: Any) -> float | None:
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def recovered(row: dict) -> bool:
    stamp = rd.to_dt(row.get("outcome_computed_at"))
    return stamp is not None and REPAIR_START <= stamp < REPAIR_END


def before_repair(rows: list[dict]) -> list[dict]:
    """Counterfactual pre-write coverage only; old computed_at is irretrievable.

    The script's guarded UPDATE fills only rows with all three returns NULL.
    The exact old empty-label timestamp cannot be reconstructed.
    """
    return [{**r, **({k: None for k in (
        "return_1d", "return_3d", "return_5d", "mfe_5d", "mae_5d",
        "benchmark_return_1d", "benchmark_return_3d", "benchmark_return_5d",
    )} if recovered(r) else {})} for r in rows]


def usable(row: dict, h: int, now: dt.datetime) -> bool:
    end = rd.label_window_end(rd.entry_day(row.get("fired_at")), h)
    computed = rd.to_dt(row.get("outcome_computed_at"))
    if end is None or computed is None:
        return False
    from analytics.market_calendar import session_bounds_utc

    complete = session_bounds_utc(end)[1] + dt.timedelta(minutes=20)
    return computed >= complete and now >= complete and number(row.get(f"return_{h}d")) is not None


def signal_days(rows: list[dict]) -> list[dict]:
    # Freeze the first observation of each ticker/entry day BEFORE label filtering:
    # do not select a later duplicate because its outcome or score looks better.
    out = {}
    for r in sorted(rows, key=lambda r: (str(r["fired_at"]), r["id"])):
        out.setdefault((r["ticker"], str(rd.entry_day(r["fired_at"]))), r)
    return list(out.values())


def distribution(rows: list[dict], h: int, now: dt.datetime) -> dict:
    vals = [number(r.get(f"return_{h}d")) for r in rows if usable(r, h, now)]
    positives = sum(v > 0 for v in vals)
    return {"total_observations": len(rows), "labeled_observations": len(vals),
            "directional_observations": None, "direction_note": "Direction is not frozen on opportunity rows",
            "positive_labels": positives, "negative_labels": len(vals) - positives,
            "positive_rate": positives / len(vals) if vals else None,
            "usable_evaluation_rows": len(vals),
            "distinct_entry_days": len({str(rd.entry_day(r["fired_at"])) for r in rows if usable(r, h, now)})}


def metrics(rows: list[dict], h: int, now: dt.datetime, field: str) -> dict:
    chosen = [r for r in rows if usable(r, h, now) and number(r.get(field)) is not None
              and 0 <= float(r[field]) <= 100]
    if not chosen:
        return {"n": 0, "roc_auc": None}
    y = np.array([int(float(r[f"return_{h}d"]) > 0) for r in chosen])
    p = np.array([float(r[field]) / 100 for r in chosen])
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    if field == "hsf_score":
        # A heuristic is not a probability: do not manufacture Brier/calibration.
        result = {"n": len(chosen), "positive_rate": float(y.mean()),
                  "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                  "probability_metrics": "NOT_APPLICABLE: HSF Score is a heuristic"}
    else:
        result = audit.classification_metrics(y, p, 0.5)
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, p > 0.5)) if len(set(y)) == 2 else None
        result["ece"] = audit.ece(y, p)
        result["mean_prediction_minus_positive_rate"] = float(p.mean() - y.mean())
        result["probability_target_warning"] = "return_h > 0 proxy differs from FutureQualitySetupHit; not true target calibration"
        result["buckets"] = audit.reliability_bins(y, p)
        for bucket in result["buckets"]:
            bucket["calibration_error"] = (abs(bucket["mean_predicted"] - bucket["observed_rate"])
                                            if bucket["n"] else None)
    days = [str(rd.entry_day(r["fired_at"])) for r in chosen]
    result["auc_ci95"] = audit.day_block_bootstrap_auc(y, p, days, n_boot=300)
    result["distinct_days"] = len(set(days))
    result["support"] = "UNDER_SAMPLED" if len(chosen) < 30 or len(set(days)) < 10 or min(int(y.sum()), len(y) - int(y.sum())) < 10 else "DESCRIPTIVE_ONLY"
    return result


def analyze(rows: list[dict], models: dict, now: dt.datetime) -> dict:
    for r in rows:
        raw = r.get("raw_signal") or {}
        r["hsf_score"] = number(raw.get("hsf_score"))
        r["entry_day"] = str(rd.entry_day(r["fired_at"]))
        r["tier"] = raw.get("status") or "UNKNOWN"
        r["recommendation"] = raw.get("recommendation") or "NOT_RECORDED"
        r["setup"] = raw.get("primary_setup") or "UNKNOWN"
        r["direction"] = raw.get("direction") or "NOT_RECORDED"
    cohort = [r for r in rows if recovered(r)]
    keys = Counter((r["source"], r.get("source_event_id"), r["ticker"], r.get("signal_type")) for r in rows)
    stages = Counter(mr.maturation_stage(r, now)["category"] for r in rows)
    orphan = sum(r.get("id") is None or not r.get("ticker") or not r.get("fired_at") for r in rows)
    mismatched = sum(r.get("source_event_id") is not None and int(r["source_event_id"]) != int(rd.to_dt(r["fired_at"]).timestamp()) for r in rows)
    bench_found = sum(all(number(r.get(f"benchmark_return_{h}d")) is not None for h in (1, 3, 5)) for r in cohort)
    full = sum(all(usable(r, h, now) for h in (1, 3, 5)) for r in cohort)
    duplicate = sum(n - 1 for n in keys.values() if n > 1)
    integrity = {"recovered_rows_expected": EXPECTED, "recovered_rows_found": len(cohort),
                 "recovered_complete_labels": full, "recovered_complete_benchmarks": bench_found,
                 "duplicate_labels": duplicate, "duplicate_benchmarks": duplicate,
                 "orphan_labels": orphan, "orphan_benchmarks": orphan,
                 "source_event_timestamp_mismatches": mismatched,
                 "relationships": "Labels and benchmarks are columns on signal_outcomes.id, not child tables. source_event_id is snapshot epoch seconds, not a foreign key.",
                 "remaining_premature": stages[mr.PREMATURE_LABEL_WRITE],
                 "remaining_unscorable": stages[mr.MISSING_PRICE_DATA] + stages[mr.PREMATURE_LABEL_WRITE],
                 "unscorable_reason_breakdown": dict(stages),
                 "recovered_by_entry_day": dict(Counter(r["entry_day"] for r in cohort)),
                 "recovered_still_premature": sum(mr.maturation_stage(r, now)["category"] == mr.PREMATURE_LABEL_WRITE for r in cohort),
                 "idempotency_status": "PASS_GUARDED_ROWS" if full == len(cohort) else "FAIL",
                 "old_outcome_timestamp": "Not retained; cohort inferred from isolated workflow transaction window"}
    integrity["status"] = "PASS" if len(cohort) == full == bench_found == EXPECTED and not (duplicate or orphan or mismatched) else "FAIL"
    before = before_repair(rows)
    days = signal_days(rows)
    day_before = before_repair(days)
    before_counts = {str(h): distribution(before, h, now) for h in (1, 3, 5)}
    after_counts = {str(h): distribution(rows, h, now) for h in (1, 3, 5)}
    # Existing trained model, no fitting. Exclude overlap with its training-label
    # horizon conservatively; do not certify unrecorded served-model provenance.
    active = models.get("prebreakout_models") or []
    trained = rd.to_dt(active[0].get("trained_at")) if len(active) == 1 else None
    safe_day = rd.label_window_end(trained.date(), 8) if trained else None
    safe = [r for r in days if safe_day and rd.entry_day(r["fired_at"]) > safe_day]
    performance = {}
    for h in (1, 3, 5):
        performance[str(h)] = {
            "hsf_score_before": metrics(day_before, h, now, "hsf_score"),
            "hsf_score_after": metrics(days, h, now, "hsf_score"),
            "served_prebreakout_proxy_after_training_gap": metrics(safe, h, now, "prebreakout_prob"),
            "served_prebreakout_proxy_before_repair": metrics(before_repair(safe), h, now, "prebreakout_prob"),
        }
    segments = {}
    for field in ("entry_day", "tier", "recommendation", "setup", "direction", "source"):
        groups = defaultdict(list)
        for row in days:
            groups[str(row.get(field, "UNKNOWN"))].append(row)
        segments[field] = {k: {str(h): {"counts": distribution(group, h, now),
                                      "hsf": metrics(group, h, now, "hsf_score"),
                                      "prebreakout_proxy": metrics([r for r in group if r in safe], h, now, "prebreakout_prob")}
                              for h in (1, 3, 5)} for k, group in sorted(groups.items())}
    cohorts = {}
    for name, group in (("recovered", cohort), ("previously_valid", [r for r in rows if not recovered(r) and usable(r, 5, now)])):
        cohorts[name] = {"observations": len(group), "signal_days": len(signal_days(group)),
                         "entry_days": dict(Counter(r["entry_day"] for r in group)),
                         "tiers": dict(Counter(r["tier"] for r in group)),
                         "recommendations": dict(Counter(r["recommendation"] for r in group)),
                         "prebreakout_distribution": audit.distribution([number(r.get("prebreakout_prob")) for r in group]),
                         "horizons": {}}
        for h in (1, 3, 5):
            vals = [r for r in group if usable(r, h, now) and number(r.get(f"benchmark_return_{h}d")) is not None]
            excess = [float(r[f"return_{h}d"]) - float(r[f"benchmark_return_{h}d"]) for r in vals]
            cohorts[name]["horizons"][str(h)] = {**distribution(group, h, now), "benchmark_n": len(vals),
                "benchmark_beat_rate": float(np.mean(np.array(excess) > 0)) if excess else None,
                "median_excess_return": float(np.median(excess)) if excess else None}
    return {"audit_as_of": now.isoformat(), "POST_RESCORE_INTEGRITY": integrity,
            "scope": "Frozen opportunities before repair start, since 2026-09-01; no live fitting",
            "models": models, "dataset_before": before_counts, "dataset_after": after_counts,
            "delta": {str(h): {"labels_added": after_counts[str(h)]["labeled_observations"] - before_counts[str(h)]["labeled_observations"],
                               "positive_rate_change": (after_counts[str(h)]["positive_rate"] - before_counts[str(h)]["positive_rate"])
                               if before_counts[str(h)]["positive_rate"] is not None and after_counts[str(h)]["positive_rate"] is not None else None}
                      for h in (1, 3, 5)},
            "performance": performance, "segments": segments, "cohorts": cohorts,
            "evaluation_signal_days": len(days), "after_training_gap_signal_days": len(safe),
            "training_gap_end": str(safe_day),
            "provenance": {"stored_ai_confidence": sum(number(r.get("ai_confidence")) is not None for r in rows),
                           "stored_served_model_version": sum(bool((r.get("raw_signal") or {}).get("prebreakout_model_version")) for r in rows)},
            "strict_production_model_performance": "UNAVAILABLE: original path/setup target and per-row served artifact version are not persisted; return>0 is a proxy only",
            "CALIBRATION": "UNVERIFIABLE_FOR_MODEL_TARGET; proxy bucket diagnostics are separate",
            "ML_READINESS": "NOT_READY",
            "NEXT_RECOMMENDED_RUN": "HSF ML v3 - Frozen Prediction and Target Provenance Audit"}
