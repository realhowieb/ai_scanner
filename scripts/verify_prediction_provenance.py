"""Bounded, enforced read-only production inventory; outputs aggregates only."""
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.prediction_provenance import digest, models_from_row  # noqa: E402

SINCE = "2026-10-09T16:48:29+00:00"
FIELDS = ("inferred_at", "build_sha", "source_scan_id", "input_timestamp", "feature_names",
          "feature_values", "feature_schema_hash", "default_mask", "preprocessing_identity",
          "calibration_snapshot", "calibration_hash", "target", "target_version",
          "training_data_end", "calibration_data_end", "loaded_artifact")


def inventory(records):
    result = {}
    for role in ("prebreakout", "ai_confidence"):
        counts, missing, invalid, reasons = Counter(), Counter(), Counter(), Counter()
        for models in records:
            p = models.get(role)
            if not isinstance(p, dict):
                counts["absent"] += 1
                continue
            counts[p.get("status", "unknown")] += 1
            if p.get("status") == "unavailable":
                reasons[p.get("reason", "unspecified")] += 1
                continue
            missing.update(f for f in FIELDS if p.get(f) is None)
            identity = p.get("loaded_artifact") or {}
            missing.update("artifact_" + f for f in ("registry_id", "version", "sha256") if identity.get(f) is None)
            names, values, mask = (p.get(f) for f in ("feature_names", "feature_values", "default_mask"))
            if not all(isinstance(v, list) for v in (names, values, mask)) or not len(names) == len(values) == len(mask):
                invalid["matrix_lengths"] += 1
            else:
                invalid["non_finite_inputs"] += int(any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values))
                invalid["mask_types"] += int(any(not isinstance(v, bool) for v in mask))
            for field in ("raw_probability", "calibrated_probability"):
                value = p.get(field)
                invalid[field] += int(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1)
            invalid["feature_schema_hash"] += int(p.get("feature_schema_hash") != digest(names))
            if p.get("calibration_snapshot") is not None:
                invalid["calibration_hash"] += int(p.get("calibration_hash") != digest(p["calibration_snapshot"]))
        result[role] = {"counts": dict(counts), "missing": dict(missing),
                        "invalid": dict(invalid), "unavailable_reasons": dict(reasons)}
    return result


def main():
    import psycopg
    from psycopg.rows import dict_row

    url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Database configuration unavailable; no local fallback")
    with psycopg.connect(url, connect_timeout=15, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        if conn.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"] != "on":
            raise RuntimeError("Read-only enforcement failed")
        conn.execute("SET LOCAL statement_timeout='30s'")
        runs = conn.execute("""SELECT id,created_at,octet_length(results_json) AS bytes,
            CASE WHEN octet_length(results_json)<=4000000 THEN results_json ELSE NULL END AS payload
            FROM runs WHERE username='cron' AND created_at >= %s
            ORDER BY created_at,id LIMIT 51""", (SINCE,)).fetchall()
        if len(runs) > 50 or sum(r["bytes"] or 0 for r in runs) > 32000000:
            raise RuntimeError("Read cap exceeded")
        observations = conn.execute("""SELECT symbol,record->'models' AS models,
            record#>>'{market_context,scan_id}' AS scan_id
            FROM hsf_observations WHERE timestamp >= %s::timestamptz - interval '1 hour'
            AND (record->>'scan_timestamp')::timestamptz >= %s AND context LIKE 'scheduled:%%'
            ORDER BY timestamp LIMIT 5001""", (SINCE, SINCE)).fetchall()
        freezes = conn.execute("""SELECT raw_signal->'models' AS models
            FROM signal_outcomes WHERE source='opportunity' AND fired_at >= %s
            ORDER BY fired_at,id LIMIT 1001""", (SINCE,)).fetchall()
        if len(observations) > 5000 or len(freezes) > 1000:
            raise RuntimeError("Read cap exceeded")
        conn.execute("ROLLBACK")
    saved, exact = [], {}
    oversized = 0
    for run in runs:
        if run["payload"] is None:
            oversized += 1
            continue
        for row in json.loads(run["payload"] or "[]"):
            models = models_from_row(row)
            saved.append(models)
            for role, p in models.items():
                if isinstance(p, dict) and p.get("source_scan_id") and p.get("inferred_at"):
                    key = (p["source_scan_id"], row.get("Ticker") or row.get("Symbol"), role, p["inferred_at"])
                    exact[key] = digest(p)
    links = Counter()
    for row in observations:
        for role, p in (row["models"] or {}).items():
            if not isinstance(p, dict):
                continue
            key = (p.get("source_scan_id"), row["symbol"], role, p.get("inferred_at"))
            if all(key) and key in exact:
                links["matched"] += 1
                links["identical"] += int(exact[key] == digest(p))
            else:
                links["unlinked"] += 1
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "since": SINCE,
              "audit_workflow_run_id": os.getenv("GITHUB_RUN_ID"),
              "read_only_enforced": True, "status": "PARTIAL" if saved else "WAITING_FOR_PROSPECTIVE_RUN",
              "saved_run_count": len(runs), "saved_candidate_count": len(saved),
              "saved_payload_bytes": sum(r["bytes"] or 0 for r in runs), "oversized_skipped": oversized,
              "canonical_count": len(observations), "frozen_count": len(freezes),
              "saved": inventory(saved), "canonical": inventory([r["models"] or {} for r in observations]),
              "frozen": inventory([r["models"] or {} for r in freezes]), "exact_links": dict(links),
              "evaluation_ready": False}
    print("PROVENANCE_VERIFICATION_JSON=" + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
