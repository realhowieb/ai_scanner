"""Run the predeclared December feature comparison (development data only).

    python scripts/ml_feature_comparison.py                  # dry: coverage + fold layout, no outcomes used
    python scripts/ml_feature_comparison.py --mode compare --dataset-version hsf-ml-YYYY-MM-DD-vN

dry     Builds research-schema-v2 features from the live database for the
        protocol window up to today. Outcome columns are blanked before any
        record is built, so nothing about results can be seen. Prints weekly
        input coverage, where the core inputs came from, and weekly row counts.
compare Loads a frozen schema-v2 dataset through the trainer's own verified
        loader (fingerprint + reserved-window guard), refuses rows outside the
        protocol window or not yet matured, runs the four arms, applies the
        locked keep rule and writes artifacts/ml_feature_comparison/result.json.

Read-only in both modes: nothing is written to the database.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "docs/ml_feature_comparison_protocol_2026-12.json"
OUTCOME_COLUMNS = ("return_1d", "return_3d", "return_5d", "mfe_5d", "mae_5d", "outcome_computed_at",
                   "benchmark_return_1d", "benchmark_return_3d", "benchmark_return_5d", "benchmark_computed_at")
INPUTS = ("trend_10d_pct", "trend_20d_pct", "rvol_20", "dollar_vol_20", "scanner_breakout_score", "scan_gap_pct",
          "rsi_14", "spy_vs_sma20_pct", "spy_above_sma20")


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def dry(protocol: dict) -> dict:
    from analytics import ml_feature_comparison as C
    from analytics import ml_v3_audit as A
    from analytics import research_dataset as rd
    from db.ml_forward_cohorts import development_ids
    from db.research_datasets import fetch_opportunity_rows, fetch_scan_records

    reserved = dt.date.fromisoformat(protocol["reserved_start"])
    end = min(dt.datetime.now(dt.timezone.utc).date(), dt.date.fromisoformat(protocol["development_end_inclusive"]))
    lo = dt.datetime.fromisoformat(protocol["development_start"]).replace(tzinfo=dt.timezone.utc)
    hi = dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    if hi.date() > reserved:
        raise SystemExit("window reaches the reserved test; refusing")
    allowed = development_ids(lo, hi)
    raw = fetch_opportunity_rows(lo, hi, ids=allowed)
    raw = [{**r, **{c: None for c in OUTCOME_COLUMNS if c in r}} for r in raw]  # never look at results
    scans = fetch_scan_records({r["ticker"] for r in raw}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
    rows = A.build_audit_rows(rd.build_records(raw, scans, feature_schema_version=2), raw, unit="signal_day")
    C.check_window(rows, protocol)
    sources = Counter((C.iso_week(r.entry_day), r.features.get("core_inputs_source")) for r in rows)
    weeks = sorted({w for w, _ in sources})
    return {"mode": "dry", "outcomes_used": False, "window": [lo.date().isoformat(), end.isoformat()],
            "signal_days": len(rows), "coverage_pct_by_week": C.coverage(rows, INPUTS),
            "core_inputs_source_by_week": {w: {s: n for (wk, s), n in sources.items() if wk == w} for w in weeks}}


def compare(protocol: dict, dataset_version: str) -> dict:
    from analytics import ml_feature_comparison as C
    from analytics import ml_v3_audit as A
    from analytics import ml_v4_providers as P
    from scripts.ml_v4_train import load_audit_rows

    if protocol.get("dataset_version") not in (None, dataset_version):
        raise SystemExit(f"protocol is pinned to {protocol['dataset_version']}; refusing {dataset_version}")
    loaded = load_audit_rows(dataset_version)  # fingerprint + reserved-window guard
    if int(loaded["entry"].get("feature_schema_version") or 1) != protocol["feature_schema_version"]:
        raise SystemExit("dataset is not research feature schema v2")
    rows = loaded["audit_rows"]
    C.check_window(rows, protocol)
    h = int(protocol["horizon"])
    immature = sum(1 for r in rows if not r.matured(h))
    if immature:
        raise SystemExit(f"{immature} signal-days not matured at {h}d; freeze after the last window closes")
    data = A.eligible(rows, h, protocol["target"])
    factory = lambda arm: P.ProviderModel(protocol["model"]["provider"], arm)  # noqa: E731
    main = C.run_arms(C.weekly_folds(data, protocol), factory)
    check = C.run_arms(C.weekly_folds(data, protocol,
                                      train_from=dt.date.fromisoformat(protocol["check_run"]["train_from"])), factory)
    decision = C.decide(main, check, protocol["keep_rule"])
    return {"mode": "compare", "protocol_id": protocol["protocol_id"], "dataset_version": dataset_version,
            "fingerprint": loaded["entry"].get("fingerprint"), "signal_days": len(rows), "eligible": len(data),
            "mean_weekly_auc": C.summary_auc(main), "check_mean_weekly_auc": C.summary_auc(check),
            "main": main, "check": check, "decision": decision,
            "library_versions": P.library_versions()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("dry", "compare"), default="dry")
    ap.add_argument("--dataset-version", default=None)
    a = ap.parse_args(argv)
    protocol = load_protocol()
    if a.mode == "compare":
        if not a.dataset_version:
            raise SystemExit("--dataset-version is required for compare")
        result = compare(protocol, a.dataset_version)
    else:
        result = dry(protocol)
    out = ROOT / "artifacts/ml_feature_comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.mode}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
