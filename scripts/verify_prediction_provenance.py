"""Bounded, enforced read-only production inventory; outputs aggregates only."""
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.prediction_provenance import digest, models_from_row  # noqa: E402

SINCE = "2026-10-09T17:53:47+00:00"
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
            if identity.get("sha256") is not None:
                invalid["artifact_hash_format"] += int(re.fullmatch(r"[0-9a-f]{64}", str(identity["sha256"])) is None)
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
        runs = conn.execute("""SELECT id,created_at,label,is_snapshot,octet_length(results_json) AS bytes,
            CASE WHEN octet_length(results_json)<=4000000 THEN results_json ELSE NULL END AS payload
            FROM runs WHERE username='cron' AND created_at >= %s
            ORDER BY created_at,id LIMIT 51""", (SINCE,)).fetchall()
        if len(runs) > 50 or sum(r["bytes"] or 0 for r in runs) > 32000000:
            raise RuntimeError("Read cap exceeded")
        observations = conn.execute("""SELECT symbol,context,created_at,record->'models' AS models,
            record->>'research_cohort' AS cohort,
            record->>'selection_reason' AS selection_reason,
            record#>>'{market_context,scan_id}' AS scan_id
            FROM hsf_observations WHERE timestamp >= %s::timestamptz - interval '1 hour'
            AND (record->>'scan_timestamp')::timestamptz >= %s AND context LIKE 'scheduled:%%'
            ORDER BY timestamp LIMIT 5001""", (SINCE, SINCE)).fetchall()
        freezes = conn.execute("""SELECT id,ticker AS symbol,fired_at,created_at,
            prebreakout_prob,raw_signal->'models' AS models
            FROM signal_outcomes WHERE source='opportunity'
            AND fired_at >= %s::timestamptz - interval '1 day'
            AND (fired_at >= %s OR (raw_signal#>>'{models,prebreakout,inferred_at}')::timestamptz >= %s)
            ORDER BY fired_at,id LIMIT 1001""", (SINCE, SINCE, SINCE)).fetchall()
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
    links = {}
    for stage, records in (("canonical", observations), ("frozen", freezes)):
        counts = Counter()
        for row in records:
            for role, p in (row["models"] or {}).items():
                if not isinstance(p, dict):
                    continue
                key = (p.get("source_scan_id"), row["symbol"], role, p.get("inferred_at"))
                if all(key) and key in exact:
                    counts["matched"] += 1
                    counts["identical"] += int(exact[key] == digest(p))
                else:
                    counts["unlinked"] += 1
        links[stage] = dict(counts)
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "since": SINCE,
              "audit_workflow_run_id": os.getenv("GITHUB_RUN_ID"),
              "read_only_enforced": True, "status": "PARTIAL" if runs or observations or freezes else "WAITING_FOR_PROSPECTIVE_RUN",
              "saved_run_count": len(runs), "saved_candidate_count": len(saved),
              "saved_payload_bytes": sum(r["bytes"] or 0 for r in runs), "oversized_skipped": oversized,
              "canonical_count": len(observations), "frozen_count": len(freezes),
              "saved": inventory(saved), "canonical": inventory([r["models"] or {} for r in observations]),
              "frozen": inventory([r["models"] or {} for r in freezes]), "exact_links": links,
              "coverage_reconciliation": reconcile(runs, observations, freezes),
              "evaluation_ready": False}
    print("PROVENANCE_VERIFICATION_JSON=" + json.dumps(result, sort_keys=True, allow_nan=False))


def reconcile(runs, observations, freezes):
    """Separate storage representations, inference events and applicability."""
    events, groups, headers, source_tickers = {}, {}, [], {}
    for run in runs:
        rows = json.loads(run.get("payload") or "[]")
        headers.append({"id": run["id"], "universe": run.get("label"),
                        "snapshot": run.get("is_snapshot"), "rows": len(rows),
                        "created_at": str(run["created_at"])})
        for row in rows:
            symbol = row.get("Ticker") or row.get("Symbol")
            for role, p in models_from_row(row).items():
                if p.get("status") != "captured":
                    continue
                key = (p.get("source_scan_id"), symbol, role, p.get("inferred_at"))
                if all(key):
                    events.setdefault(key, set()).add(digest(p))
                    source_tickers.setdefault(symbol, set()).add(p["source_scan_id"])
    for row in observations:
        cohort = row.get("cohort") or "LEGACY_UNKNOWN"
        key = str(row.get("context")) + ":" + cohort
        group = groups.setdefault(key, Counter())
        group["rows"] += 1
        models = row.get("models") or {}
        for role in ("prebreakout", "ai_confidence"):
            if role in models:
                group[role + "_present"] += 1
            elif role == "ai_confidence" or cohort in ("NEAR_MISS", "CONTROL"):
                group[role + "_expected_not_invoked"] += 1
            else:
                group[role + "_unexpected_or_unverified_absence"] += 1
    return {"saved_runs": headers, "unique_linkable_inferences": len(events),
            "conflicting_duplicate_identities": sum(len(v) > 1 for v in events.values()),
            "cohort_coverage": {k: dict(v) for k, v in groups.items()},
            "frozen_records": [{"id": r["id"], "ticker": r["symbol"],
                                "fired_at": str(r["fired_at"]), "created_at": str(r["created_at"]),
                                "models_present": sorted((r.get("models") or {}).keys()),
                                "displayed_prebreakout_present": r.get("prebreakout_prob") is not None,
                                "same_ticker_saved_source_candidates": len(source_tickers.get(r["symbol"], set())),
                                "link_note": "ticker presence is NOT exact inference linkage"} for r in freezes]}


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never emit connection details or a provider/record payload on failure.
        raise SystemExit("Read-only verification failed: " + type(exc).__name__) from None
