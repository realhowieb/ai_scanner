"""Read-only provenance inventory. Never loads models or regenerates predictions."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import research_dataset as rd  # noqa: E402

FIELDS = ("model_version", "artifact_sha256", "feature_schema_version", "feature_values",
          "calibration_version", "training_data_end", "target_name", "target_version",
          "target_value", "prediction_raw", "prediction_timestamp")


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def inventory(rows):
    """Availability is not verification of provenance or reproducibility."""
    n = len(rows)
    checks = {
        "identity": lambda r: bool(r.get("id") and r.get("ticker") and r.get("fired_at")),
        "prebreakout_prediction": lambda r: finite(r.get("prebreakout_prob")),
        "ai_prediction": lambda r: finite(r.get("ai_confidence")),
        "benchmark_5d": lambda r: finite(r.get("benchmark_return_5d")),
    }
    checks.update({key: lambda r, k=key: (r.get("provenance") or {}).get(k) is not None for key in FIELDS})
    return {key: {"count": sum(bool(check(r)) for r in rows),
                  "percent": round(100 * sum(bool(check(r)) for r in rows) / n, 2) if n else None}
            for key, check in checks.items()}


def blockers(row, cutoff=None):
    p = row.get("provenance") or {}
    reasons = []
    if not finite(row.get("prebreakout_prob")):
        reasons.append("MISSING_PREDICTION")
    for key in FIELDS:
        if p.get(key) is None:
            reasons.append("MISSING_" + key.upper())
    if p.get("target_name") != "FutureQualitySetupHit":
        reasons.append("TARGET_IDENTITY_UNVERIFIED")
    observed = rd.to_dt(row.get("fired_at"))
    boundary = rd.to_dt(cutoff or p.get("training_data_end"))
    if observed is None or boundary is None or observed <= boundary:
        reasons.append("TRAINING_BOUNDARY_UNVERIFIED_OR_OVERLAPPING")
    # Presence alone never proves the artifact/input/target chain was verified.
    reasons.append("PROVENANCE_CHAIN_REQUIRES_VERIFICATION")
    return reasons


def summarize(rows):
    by_day = defaultdict(list)
    first = {}
    for r in sorted(rows, key=lambda r: (str(r["fired_at"]), r["id"])):
        day = str(rd.entry_day(r["fired_at"]))
        by_day[day].append(r)
        first.setdefault((r["ticker"], day), r)
    unique = list(first.values())
    later = [r for r in unique if str(rd.entry_day(r["fired_at"])) > "2026-09-22"]
    mature = [r for r in later if finite(r.get("return_5d"))]
    probability_counts = Counter(str(r["prebreakout_prob"]) for r in rows if finite(r.get("prebreakout_prob")))
    return {
        "rows": len(rows), "coverage": inventory(rows),
        "by_entry_date": {d: {"rows": len(g), "coverage": inventory(g),
                              "probability_counts": dict(Counter(str(r["prebreakout_prob"]) for r in g if finite(r.get("prebreakout_prob"))))}
                          for d, g in sorted(by_day.items())},
        "by_source": {s: inventory([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
        "prior_audit_funnel": {"rows": len(rows), "first_ticker_entry_day": len(unique),
                              "after_conservative_gap": len(later), "with_5d_return": len(mature),
                              "with_prediction": sum(finite(r.get("prebreakout_prob")) for r in mature)},
        "probability_counts": dict(probability_counts),
        "blocker_counts": dict(Counter(reason for r in rows for reason in blockers(r))),
        "classification": {"VERIFIED_EVALUABLE": 0, "RECONSTRUCTABLE_WITH_LIMITATIONS": 0,
                           "NOT_EVALUABLE": len(rows)},
        "classification_note": "No provenance chain verified; availability does not establish historical reconstruction.",
    }


def read_rows(conn):
    conn.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    conn.execute("SET LOCAL statement_timeout = '30s'")
    rows = conn.execute("""
        SELECT id, ticker, source, fired_at, prebreakout_prob, ai_confidence,
               return_5d, benchmark_return_5d,
               jsonb_build_object(
                 'model_version', raw_signal->'prebreakout_model_version',
                 'artifact_sha256',raw_signal->'artifact_sha256',
                 'feature_schema_version',raw_signal->'feature_schema_version',
                 'feature_values',raw_signal->'feature_values',
                 'calibration_version',raw_signal->'calibration_version',
                 'training_data_end',raw_signal->'training_data_end',
                 'target_name',raw_signal->'target_name',
                 'target_version',raw_signal->'target_version',
                 'target_value',raw_signal->'target_value',
                 'prediction_raw',raw_signal->'prediction_raw',
                 'prediction_timestamp',raw_signal->'prediction_timestamp') AS provenance
        FROM signal_outcomes
        WHERE source='opportunity' AND fired_at >= '2026-09-01'
          AND fired_at < '2026-10-09 05:12:00+00'
        ORDER BY fired_at,id LIMIT 10001
    """).fetchall()
    if len(rows) > 10000:
        raise ValueError("Audit row limit exceeded")
    return rows


def scan_inventory(runs, features, calibration_map, observations):
    """Inspect canonical scheduled payloads without exporting individual rows.

    Source-field signatures are not final inference vectors. Candidate joins
    are counted only; ticker/day or run timestamps do not certify linkage.
    """
    import numpy as np

    required = ("IsBreakout", "BreakoutScore", "BreakoutPos20D", "Last")
    counts = Counter()
    date_counts = defaultdict(Counter)
    signatures = Counter()
    missing_fields = Counter()
    floor_raw = Counter()
    ticker_dates = Counter()
    run_times = Counter()
    for run in runs:
        run_times[str(rd.to_dt(run["created_at"]))] += 1
        day = str(rd.entry_day(run["created_at"]))
        if (run.get("payload_bytes") or 0) > 2_000_000:
            counts["oversized_payloads_skipped"] += 1
            continue
        payload = run.get("results_json") or "[]"
        if len(payload.encode()) > 2_000_000:
            counts["oversized_payloads_skipped"] += 1
            continue
        try:
            records = json.loads(payload)
        except (TypeError, ValueError):
            counts["invalid_payloads"] += 1
            continue
        if not isinstance(records, list):
            counts["non_list_payloads"] += 1
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            counts["scan_rows"] += 1
            date_counts[day]["scan_rows"] += 1
            symbol = record.get("Symbol") or record.get("Ticker")
            if isinstance(symbol, str):
                ticker_dates[(symbol.upper(), day)] += 1
            for field in ("PreBreakoutProbRaw", "PreBreakoutProb%", "AI Confidence",
                          "FutureQualitySetupHit", "ForwardReturnHit", "Return_5D"):
                if finite(record.get(field)):
                    counts[field] += 1
                    date_counts[day][field] += 1
            if all(field in record and record[field] is not None for field in required):
                counts["candidate_and_setup_source_fields"] += 1
            available = [field for field in features if finite(record.get(field))]
            missing_fields.update(field for field in features if field not in available)
            counts["complete_current_schema_source_rows"] += int(len(available) == len(features) and bool(features))
            # Only numeric source values enter a signature; missingness preserved.
            vector = [float(record[field]) if field in available else None for field in features]
            signature = hashlib.sha256(json.dumps(vector, allow_nan=False).encode()).hexdigest()
            signatures[signature] += 1
            raw = record.get("PreBreakoutProbRaw")
            displayed = record.get("PreBreakoutProb%")
            if finite(raw) and finite(displayed):
                counts["raw_display_pairs"] += 1
                if abs(float(displayed) - 13.1) < 1e-6:
                    floor_raw[str(float(raw))] += 1
                    counts["display_13_1_with_raw"] += 1
                    if calibration_map and calibration_map.get("x"):
                        counts["display_13_1_raw_below_current_endpoint"] += int(float(raw) <= calibration_map["x"][0])
                if calibration_map and calibration_map.get("x") and calibration_map.get("y"):
                    mapped = round(float(np.interp(float(raw), calibration_map["x"], calibration_map["y"])) * 100, 1)
                    counts["pairs_match_current_calibration"] += int(abs(mapped - float(displayed)) < 1e-6)
    return {
        "runs": len(runs), "counts": dict(counts),
        "by_entry_date": {k: dict(v) for k, v in sorted(date_counts.items())},
        "distinct_current_schema_source_signatures": len(signatures),
        "largest_source_signature_cluster": max(signatures.values(), default=0),
        "missing_current_schema_source_fields": dict(missing_fields),
        "floor_raw_summary": {"n": sum(floor_raw.values()), "unique_values": len(floor_raw),
                             "min": min((float(v) for v in floor_raw), default=None),
                             "max": max((float(v) for v in floor_raw), default=None)},
        "observations_with_ticker_entry_day_candidate": sum(ticker_dates[(r["ticker"], str(rd.entry_day(r["fired_at"])))] > 0 for r in observations),
        "observations_with_exact_run_timestamp_candidate": sum(run_times[str(rd.to_dt(r["fired_at"]))] > 0 for r in observations),
        "warning": "Candidate joins and current-map agreement do not establish historical artifact or calibration identity; source signatures are not served feature vectors.",
    }


def main():
    import psycopg
    from psycopg.rows import dict_row

    url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Database configuration unavailable; no fallback")
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=15) as conn:
        rows = read_rows(conn)
        # Only safe aggregate key names/metadata are exported, never raw payloads.
        keys = conn.execute("""SELECT DISTINCT jsonb_object_keys(raw_signal) AS key
            FROM signal_outcomes WHERE source='opportunity'
            AND fired_at >= '2026-09-01' AND fired_at < '2026-10-09 05:12:00+00'
            LIMIT 100""").fetchall()
        indicator_keys = conn.execute("""SELECT DISTINCT jsonb_object_keys(indicators) AS key
            FROM signal_outcomes WHERE source='opportunity'
            AND fired_at >= '2026-09-01' AND fired_at < '2026-10-09 05:12:00+00'
            LIMIT 100""").fetchall()
        registry = conn.execute("""SELECT id, model_version, trained_at,
            metadata->'calibration_map' AS calibration_map,
            metadata->'training_data_end' AS training_data_end,
            metadata->'train_end' AS train_end,
            metadata->'training_end' AS training_end, feature_names
            FROM prebreakout_models WHERE is_active=true LIMIT 5""").fetchall()
        runs = conn.execute("""SELECT id, created_at,
            CASE WHEN octet_length(results_json) <= 2000000 THEN results_json ELSE NULL END AS results_json,
            octet_length(results_json) AS payload_bytes
            FROM runs WHERE username='cron' AND label='US_MARKET'
            AND created_at >= '2026-09-01' AND created_at < '2026-10-09 05:12:00+00'
            ORDER BY created_at,id LIMIT 2001""").fetchall()
        if len(runs) > 2000 or sum(r["payload_bytes"] or 0 for r in runs) > 64_000_000:
            raise SystemExit("Scheduled history audit cap exceeded")
        canonical = {"status": "TABLE_UNAVAILABLE"}
        if conn.execute("SELECT to_regclass('hsf_observations') AS name").fetchone()["name"]:
            canonical = {"status": "READ", "groups": conn.execute("""
                SELECT timestamp::date AS utc_date, context, COUNT(*) AS rows,
                  COUNT(*) FILTER (WHERE record#>>'{models,prebreakout,probability}' IS NOT NULL) AS prebreakout_predictions,
                  COUNT(*) FILTER (WHERE record#>>'{models,ai_confidence,confidence}' IS NOT NULL) AS ai_predictions,
                  COUNT(*) FILTER (WHERE record#>>'{versions,prebreakout_model}' IS NOT NULL) AS code_model_version_tags,
                  COUNT(*) FILTER (WHERE record#>>'{research_metadata,feature_schema_version}' IS NOT NULL) AS observation_schema_tags,
                  COUNT(*) FILTER (WHERE record#>>'{research_metadata,scanner_commit_sha}' IS NOT NULL) AS scanner_commit_tags,
                  COUNT(*) FILTER (WHERE record#>>'{models,prebreakout,artifact_sha256}' IS NOT NULL) AS artifact_hashes,
                  COUNT(*) FILTER (WHERE record#>>'{models,prebreakout,calibration_version}' IS NOT NULL) AS calibration_identity,
                  COUNT(*) FILTER (WHERE record#>>'{models,prebreakout,target_name}' IS NOT NULL) AS target_identity,
                  COUNT(*) FILTER (WHERE record#>>'{research_metadata,row_features,breakout_pos_20d}' IS NOT NULL) AS resistance_source,
                  COUNT(*) FILTER (WHERE record#>>'{market,high}' IS NOT NULL AND record#>>'{market,low}' IS NOT NULL) AS high_low_source
                FROM hsf_observations
                WHERE timestamp >= '2026-09-01' AND timestamp < '2026-10-09 05:12:00+00'
                  AND context LIKE 'scheduled:%%'
                GROUP BY timestamp::date,context ORDER BY timestamp::date,context LIMIT 501
            """).fetchall()}
            if len(canonical["groups"]) > 500:
                raise SystemExit("Canonical inventory group cap exceeded")
    result = summarize(rows)
    result.update(raw_signal_keys=[r["key"] for r in keys], active_registry=registry,
                  indicator_keys=[r["key"] for r in indicator_keys],
                  workflow_run_id=os.getenv("GITHUB_RUN_ID"))
    active = registry[0] if len(registry) == 1 else {}
    result["scheduled_history"] = scan_inventory(runs, active.get("feature_names") or [], active.get("calibration_map"), rows)
    result["canonical_research_store"] = canonical
    payload = json.dumps(result, default=str, allow_nan=False).encode()
    print("=== PROVENANCE_BUNDLE ===")
    print(base64.b64encode(gzip.compress(payload)).decode())
    print("FUNNEL", result["prior_audit_funnel"])


if __name__ == "__main__":
    main()
