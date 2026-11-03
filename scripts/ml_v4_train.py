"""HSF ML v4: train XGBoost, LightGBM and CatBoost on a finalized research dataset.

    workflow_dispatch "ML v4 Train" (manual)       or    python scripts/ml_v4_train.py

Milestone 2 is a DRY RUN by design:
  * reads (SELECT only) the research window and an EXISTING finalized dataset
    version; it never finalizes a new version (that writes to production and
    needs Howard's go), never writes to any table, never touches AI Confidence
    or PreBreakout;
  * applies the dataset's freeze instant, so outcomes that matured after the
    freeze are treated as pending (the same view as the ML v3 audit);
  * trains every provider x feature set on identical rows, stopping before the
    reserved final holdout (purged + embargoed), for the pre-declared target
    "+4% before -2% within 5 trading days";
  * writes versioned artifacts + a manifest to --out (default a fresh folder
    under ml/artifacts/ml_v4/, git-ignored); the workflow uploads that folder
    as a run artifact.

No evaluation numbers are produced here; walk-forward evaluation and
calibration are Milestone 3. Training-set AUCs would be meaningless and are
deliberately not printed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EARLIEST = dt.date(2025, 1, 1)


def _log(msg: str) -> None:
    print(f"[ml_v4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _benchmark_computed_at(ids) -> dict:
    """SELECT only: when each row's SPY benchmark was filled (for the as-of view)."""
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, benchmark_computed_at FROM signal_outcomes WHERE id = ANY(%s)", ([int(i) for i in ids],))
        cols = [d[0] for d in cur.description]
        out = {}
        for r in cur.fetchall():
            r = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
            out[int(r["id"])] = r["benchmark_computed_at"]
        return out
    except Exception:
        conn.rollback()
        return {}
    finally:
        cur.close()
        conn.close()


def _pick_version(requested: str | None) -> dict:
    from db.research_datasets import get_dataset_version, list_dataset_versions

    versions = list_dataset_versions()
    if not versions:
        raise SystemExit("No finalized research dataset version exists. Finalizing one writes to production, "
                         "so it needs Howard's go; this script never does it.")
    if requested:
        entry = get_dataset_version(requested)
        if entry is None:
            raise SystemExit(f"No dataset version {requested!r}. Available: "
                             + ", ".join(v["dataset_version"] for v in versions))
        return entry
    newest = sorted(versions, key=lambda v: (str(v["created_at"]), v["dataset_version"]))[-1]
    return get_dataset_version(newest["dataset_version"])


def _as_list(value):
    return json.loads(value) if isinstance(value, str) else (value or [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-version", default=os.getenv("ML_V4_DATASET_VERSION") or None,
                    help="finalized research dataset version (default: newest)")
    ap.add_argument("--providers", default=os.getenv("ML_V4_PROVIDERS") or "xgboost,lightgbm,catboost")
    ap.add_argument("--feature-sets", default=os.getenv("ML_V4_FEATURE_SETS") or "production,leakage_safe")
    ap.add_argument("--out", default=None, help="artifact folder (default: ml/artifacts/ml_v4/<UTC stamp>)")
    args = ap.parse_args(argv)

    from analytics import ml_v3_audit as A
    from analytics import ml_v3_pipeline as V3
    from analytics import ml_v4_providers as P
    from analytics import research_dataset as rd
    from api.research import load_window

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    for p in providers:
        if p not in P.PROVIDERS:
            raise SystemExit(f"unknown provider {p!r}")
    for f in feature_sets:
        if f not in P.FEATURE_SETS:
            raise SystemExit(f"unknown feature set {f!r}")
    missing = [p for p in providers if p not in P.available_providers()]
    if missing:
        raise SystemExit(f"not installed: {missing}; pip install -r requirements-ml-v4.txt -c requirements.lock")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else ROOT / "ml" / "artifacts" / "ml_v4" / stamp
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; ML v4 never overwrites artifacts")

    t0 = time.time()
    entry = _pick_version(args.dataset_version)
    meta = entry.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    members = _as_list(entry.get("observation_ids"))
    frozen_at = rd.to_dt(entry.get("created_at"))
    _log(f"dataset {entry['dataset_version']} frozen_at={frozen_at} members={len(members)}")

    window = load_window(EARLIEST, dt.datetime.now(dt.timezone.utc).date())
    bca = _benchmark_computed_at([r["id"] for r in window["rows"]])
    rows = [{**r, "benchmark_computed_at": bca.get(int(r["id"]))} for r in window["rows"]]
    rows_asof = V3.as_of(rows, frozen_at)
    built = rd.build_dataset(rows_asof, window["scans"], filters=meta.get("filters") or {}, members=members)
    audit_rows = A.build_audit_rows(built["records"], rows_asof, unit="signal_day")
    _log(f"signal-day rows={len(audit_rows)}")

    split = P.training_split(audit_rows, P.PRIMARY_TARGET)
    hold = split["holdout"]
    _log(f"target={P.PRIMARY_TARGET.label}@{P.PRIMARY_TARGET.horizon}d eligible={split['eligible_rows']} "
         f"train={split['train_rows']} (positive {split['train_positive']}) purged={split['purged']} "
         f"embargoed={split['embargoed']} holdout={'from ' + hold['start'] + ', ' + str(hold['rows']) + ' rows' if hold['valid'] else 'NOT reserved: ' + str(hold['reason'])}")
    if split["train_rows"] == 0 or split["train_positive"] in (0, split["train_rows"]):
        raise SystemExit("training rows need both classes; nothing trained")

    dataset = {"dataset_version": entry["dataset_version"], "dataset_fingerprint": entry.get("fingerprint"),
               "unit": "signal_day"}
    results = P.train_all(split["train"], providers=providers, feature_sets=feature_sets,
                          target=P.PRIMARY_TARGET, dataset=dataset, git_sha=os.getenv("GITHUB_SHA"))
    manifests, failures = [], []
    for res in results:
        rec = res["record"]
        if res["model"] is None:
            failures.append(rec)
            _log(f"FAILED {rec['provider']}/{rec['feature_set']}: {rec['error']}")
            continue
        man = P.save_artifact(res["model"], rec, out)
        manifests.append(man)
        _log(f"trained {man['model_version']} in {man['training_seconds']}s, {len(man['columns'])} columns, "
             f"{man['artifact_bytes']} bytes, sha256 {man['artifact_sha256'][:12]}")

    summary = {
        "engine_version": P.ENGINE_VERSION, "mode": "dry_run (no database writes)",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "git_sha": os.getenv("GITHUB_SHA"),
        "dataset": {**dataset, "frozen_at": str(frozen_at), "members": len(members),
                    "fingerprint_rebuilt_as_of": built["metadata"].get("fingerprint")},
        "target": {"label": P.PRIMARY_TARGET.label, "horizon_days": P.PRIMARY_TARGET.horizon,
                   "description": P.PRIMARY_TARGET.description},
        "split": {k: v for k, v in split.items() if k not in ("train", "target")} | {
            "holdout": {k: v for k, v in hold.items() if k != "ids"}},
        "models": [{k: m[k] for k in ("model_version", "provider", "feature_set", "artifact_file",
                                      "artifact_sha256", "artifact_bytes", "training_seconds")} for m in manifests],
        "failures": failures,
        "library_versions": P.library_versions(),
        "seconds": round(time.time() - t0, 1),
        "note": "No evaluation metrics: walk-forward evaluation and calibration are Milestone 3.",
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ml_v4_training_manifest.json", "x", encoding="utf-8") as fh:
        fh.write(A.dumps(summary) + "\n")
    _log(f"wrote {len(manifests)} artifacts + manifest to {out} in {summary['seconds']}s")
    print(A.dumps(summary))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
