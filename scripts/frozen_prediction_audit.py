"""Read-only provenance inventory. Never loads models or regenerates predictions."""
from __future__ import annotations

import base64
import gzip
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
    checks.update({key: lambda r, k=key: r.get("provenance", {}).get(k) is not None for key in FIELDS})
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
        registry = conn.execute("""SELECT id, model_version, trained_at,
            metadata->'calibration_map' AS calibration_map,
            metadata->'training_data_end' AS training_data_end,
            metadata->'train_end' AS train_end,
            metadata->'training_end' AS training_end
            FROM prebreakout_models WHERE is_active=true LIMIT 5""").fetchall()
    result = summarize(rows)
    result.update(raw_signal_keys=[r["key"] for r in keys], active_registry=registry,
                  workflow_run_id=os.getenv("GITHUB_RUN_ID"))
    payload = json.dumps(result, default=str, allow_nan=False).encode()
    print("=== PROVENANCE_BUNDLE ===")
    print(base64.b64encode(gzip.compress(payload)).decode())
    print("FUNNEL", result["prior_audit_funnel"])


if __name__ == "__main__":
    main()
