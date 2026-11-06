"""Read-only aggregate diagnosis of the oldest finalized dataset. Never fits models."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def diagnose(entry, current, asof):
    """Only describe differences for which retained evidence exists."""
    meta = entry.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    expected = set(map(int, entry.get("observation_ids") or []))
    result = {"dataset_version": entry["dataset_version"], "frozen_at": str(entry["created_at"]),
              "original_code_revision": meta.get("code_revision"), "stored_fingerprint": entry["fingerprint"],
              "stored_members": len(expected), "original_component_hashes_available": bool(meta.get("component_fingerprints")),
              "rebuilds": {}}
    for name, built in (("current", current), ("as_of_freeze", asof)):
        have = set(map(int, built["observation_ids"]))
        quality = built["metadata"].get("data_quality") or {}
        old = meta.get("data_quality") or {}
        result["rebuilds"][name] = {
            "fingerprint": built["metadata"]["fingerprint"], "matches_original": entry["fingerprint"] == built["metadata"]["fingerprint"],
            "members": len(have), "missing_members": len(expected-have), "unexpected_members": len(have-expected),
            "quality_changes": {k: {"original": old.get(k), "rebuilt": quality.get(k)}
                                for k in ("scan_join_status", "maturity_5d", "missing_outcomes", "benchmark_gaps_among_matured_5d")
                                if old.get(k) != quality.get(k)},
            "changed_components": [k for k, v in (meta.get("component_fingerprints") or {}).items()
                                   if (built["metadata"].get("component_fingerprints") or {}).get(k) != v],
        }
    result["limits"] = ("Original feature/label matrices are not retained in the version registry. "
                        "Absent original component hashes, aggregate differences do not identify every changed cell or a unique cause.")
    return result


def main():
    # Set transaction mode after connecting: some poolers reject startup PGOPTIONS.
    from db import engine

    original_connect = engine.get_neon_conn

    def readonly_connect():
        conn = original_connect()
        if conn is not None:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL statement_timeout = 60000")
        return conn

    engine.get_neon_conn = readonly_connect
    from analytics import ml_v3_pipeline as pipeline
    from analytics import research_dataset as rd
    from api.research import load_window
    from db.research_datasets import get_dataset_version, list_dataset_versions
    from scripts.ml_v4_train import _as_list, _benchmark_computed_at

    entries = list_dataset_versions()
    if not entries:
        raise SystemExit("No finalized dataset exists")
    chosen = min(entries, key=lambda x: (str(x["created_at"]), x["dataset_version"]))
    entry = get_dataset_version(chosen["dataset_version"])
    entry["observation_ids"] = _as_list(entry.get("observation_ids"))
    meta = rd._json(entry.get("metadata"))
    frozen = rd.to_dt(entry["created_at"])
    window = load_window(dt.date(2025, 1, 1), dt.datetime.now(dt.timezone.utc).date())
    benchmark_at = _benchmark_computed_at([r["id"] for r in window["rows"]])
    rows = [{**r, "benchmark_computed_at": benchmark_at.get(int(r["id"]))} for r in window["rows"]]
    kwargs = {"filters": meta.get("filters") or {}, "members": entry["observation_ids"],
              "feature_schema_version": entry["feature_schema_version"], "label_schema_version": entry["label_schema_version"]}
    current = rd.build_dataset(rows, window["scans"], **kwargs)
    asof = rd.build_dataset(pipeline.as_of(rows, frozen), window["scans"], **kwargs)
    report = diagnose(entry, current, asof)
    from analytics import ml_readiness as readiness
    from analytics import ml_v3_audit as audit
    from analytics import ml_v4_providers as providers

    now = dt.datetime.now(dt.timezone.utc)
    fresh_rows = [r for r in pipeline.as_of(rows, now) if rd.to_dt(r["fired_at"]) >=
                  dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc)]
    fresh = rd.build_dataset(fresh_rows, window["scans"], filters={"start_date": "2026-09-14", "end_date": now.date().isoformat()})
    audit_rows = audit.build_audit_rows(fresh["records"], fresh_rows, unit="signal_day")
    target = providers.PRIMARY_TARGET
    data = audit.eligible(audit_rows, target.horizon, target.label)
    split = providers.training_split(audit_rows)
    folds = audit.walk_forward_folds(split["train"], target.horizon, **readiness.FOLD_SPEC) if split["holdout"]["valid"] else []
    positives = sum(y for _, y in data)
    live = readiness.build_report(fresh_rows, window["scans"], now=now)
    report["fresh_unfinalized_preview"] = {
        "as_of": now.isoformat(), "start_date": "2026-09-14", "observations": len(fresh["observation_ids"]),
        "eligible_signal_days": len(data), "target": target.description,
        "positive": positives, "negative": len(data)-positives,
        "minority_share": min(positives, len(data)-positives)/len(data) if data else None,
        "training_rows": split["train_rows"], "training_positive": split["train_positive"],
        "holdout": {k: v for k, v in split["holdout"].items() if k != "ids"},
        "pre_holdout_folds": len(folds), "blocking_reasons": live["blocking_reasons"],
        "finalized": False, "models_fitted": 0,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
