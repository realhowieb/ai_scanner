"""HSF ML v3 leakage-safe walk-forward audit, run against production.

    workflow_dispatch Diagnostics -> script=ml_v3_audit.py

What it does (in order):
  1. Loads the canonical research window (signal_outcomes opportunities +
     scheduled hsf_observations), exactly as /v1/research does.
  2. Freezes the audit dataset: if no research dataset version exists yet it
     finalizes ONE with the canonical builder (one INSERT into
     research_dataset_versions, ON CONFLICT DO NOTHING; the only write this
     script can make). Later runs reuse that version's members and its
     created_at as the freeze instant, so every rerun sees the same labels.
  3. Reads (SELECT only) the model registry, the active AI Confidence model,
     SPY daily closes (regime labels) and the runs-table history (legacy
     metric reproduction).
  4. Runs analytics.ml_v3_pipeline and writes ml/reports/ml_v3_*.
  5. Prints a summary and, last, a base64 tar.gz of the reports between
     markers, because workflow artifacts can't be downloaded from the
     sandbox and only the log tail is readable.

It never retrains or saves a model, never changes a score, and never writes
anything except the one dataset-version row in step 2.
"""
from __future__ import annotations

import base64
import datetime as dt
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EARLIEST = dt.date(2025, 1, 1)
OUT = ROOT / "ml" / "reports"
BUNDLE_BEGIN = "=== ML_V3_BUNDLE_BEGIN ==="
BUNDLE_END = "=== ML_V3_BUNDLE_END ==="


def _ensure_ml_libs() -> None:
    try:
        importlib.import_module("xgboost")
        importlib.import_module("sklearn")
        importlib.import_module("joblib")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--prefer-binary",
                        "-r", str(ROOT / "requirements-ml.txt"), "-c", str(ROOT / "requirements.lock")], check=True)
        importlib.invalidate_caches()


def _log(msg: str) -> None:
    print(f"[ml_v3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _registry_summary() -> dict:
    """Model registry rows without model bytes (SELECT only)."""
    from db.engine import get_neon_conn

    keys = ("auc", "mean_auc", "std_auc", "rows", "positive_rows", "training_rows", "validation_rows",
            "validation_method", "target", "target_rule", "purge_days", "selected_feature_set", "promotion_result",
            "calibration_map")
    out = {}
    conn = get_neon_conn()
    cur = conn.cursor()
    for table, extra in (("ai_confidence_models", ""), ("prebreakout_models", ", auc")):
        cur.execute(f"SELECT id, model_version, is_active, trained_at, created_at, feature_names, metadata{extra} "
                    f"FROM {table} ORDER BY id")  # nosec B608 - fixed table names
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            r = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            fn = r.get("feature_names") or []
            if isinstance(fn, str):
                fn = json.loads(fn)
            m = {k: meta.get(k) for k in keys if k in meta}
            if isinstance(m.get("calibration_map"), dict):
                m["calibration_map"] = {"method": m["calibration_map"].get("method"),
                                        "points": len(m["calibration_map"].get("x") or [])}
            rows.append({"id": r["id"], "model_version": r["model_version"], "is_active": r["is_active"],
                         "trained_at": str(r["trained_at"]), "created_at": str(r["created_at"]),
                         "auc_column": r.get("auc"), "feature_count": len(fn), "features": fn, "metadata": m})
        out[table] = rows
    cur.close()
    conn.close()
    return out


def _benchmark_computed_at(ids) -> dict:
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


def _spy_closes() -> dict:
    from data.price_alpaca import download_multi_alpaca

    bars = download_multi_alpaca(["SPY"], period="220d", interval="1d", prepost=False, timeout_s=25.0)
    df = bars.get("SPY")
    if df is None or df.empty:
        return {}
    return {(ts.date() if hasattr(ts, "date") else ts): float(c) for ts, c in df["Close"].dropna().items()}


def _freeze(window: dict) -> dict:
    """Finalize the audit dataset version once; afterwards reuse it."""
    from analytics import research_dataset as rd
    from db.research_datasets import get_dataset_version, list_dataset_versions, save_dataset_version

    versions = list_dataset_versions()
    created_now = False
    if not versions:
        built = rd.build_dataset(window["rows"], window["scans"],
                                 filters={"start_date": window["start"], "end_date": window["end"]},
                                 code_revision=os.getenv("GITHUB_SHA"))
        name = rd.version_name(dt.datetime.now(dt.timezone.utc).date(), [])
        meta = {**built["metadata"], "dataset_version": name, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purpose": "ML v3 leakage-safe walk-forward audit"}
        created_now = save_dataset_version({
            "dataset_version": name, "feature_schema_version": meta["feature_schema_version"],
            "label_schema_version": meta["label_schema_version"], "fingerprint": meta["fingerprint"],
            "observation_count": meta["observation_count"], "observation_ids": built["observation_ids"],
            "metadata": meta})
        _log(f"finalized {name}: written={created_now} fingerprint={meta['fingerprint']}")
        versions = list_dataset_versions()
    # the audit uses the oldest version (the one frozen for this audit)
    oldest = sorted(versions, key=lambda v: (str(v["created_at"]), v["dataset_version"]))[0]
    entry = get_dataset_version(oldest["dataset_version"])
    entry["created_now"] = created_now
    return entry


def main() -> int:
    t0 = time.time()
    _ensure_ml_libs()
    from analytics import ml_v3_audit as A
    from analytics import ml_v3_pipeline as P
    from analytics import ml_v3_reports as R
    from analytics import research_dataset as rd
    from analytics import research_schema as rs
    from api.research import load_window

    today = dt.datetime.now(dt.timezone.utc).date()
    _log("loading research window")
    window = load_window(EARLIEST, today)
    _log(f"rows={len(window['rows'])} scans={len(window['scans'])}")

    entry = _freeze(window)
    meta = entry.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    members = entry.get("observation_ids") or []
    if isinstance(members, str):
        members = json.loads(members)
    frozen_at = rd.to_dt(entry.get("created_at"))
    _log(f"dataset {entry['dataset_version']} frozen_at={frozen_at} members={len(members)}")

    rebuilt = rd.build_dataset(window["rows"], window["scans"], filters=meta.get("filters") or {},
                               members=members, code_revision=os.getenv("GITHUB_SHA"))
    bca = _benchmark_computed_at([r["id"] for r in window["rows"]])
    rows = [{**r, "benchmark_computed_at": bca.get(int(r["id"]))} for r in window["rows"]]
    rows_asof = P.as_of(rows, frozen_at)
    asof_built = rd.build_dataset(rows_asof, window["scans"], filters=meta.get("filters") or {}, members=members)
    asof_records = asof_built["records"]
    cov = rd.coverage(asof_records)

    # model versions represented / scoring versions represented
    manifest = {
        "audit_version": A.AUDIT_VERSION,
        "dataset_version": entry["dataset_version"],
        "dataset_fingerprint": entry["fingerprint"],
        "dataset_created_at": str(entry.get("created_at")),
        "dataset_created_in_this_run": bool(entry.get("created_now")),
        "fingerprint_rebuilt_now": rebuilt["metadata"]["fingerprint"],
        "fingerprint_reproducible_now": rebuilt["metadata"]["fingerprint"] == entry["fingerprint"],
        "fingerprint_as_of_freeze": asof_built["metadata"]["fingerprint"],
        "as_of_rule": "outcomes computed after dataset_created_at are treated as pending and benchmarks computed "
                      "after it as absent, so reruns see the labels exactly as frozen",
        "git_revision": os.getenv("GITHUB_SHA"),
        "feature_schema_version": entry["feature_schema_version"],
        "label_schema_version": entry["label_schema_version"],
        "observation_count": len(asof_records),
        "matured_count": cov["matured_observations"],
        "certified_count": cov["certified_observations"],
        "pending_count": cov["pending_observations"],
        "unavailable_count": cov["unavailable_observations"],
        "earliest_observation": cov["earliest_observation"],
        "latest_observation": cov["latest_observation"],
        "model_versions_represented": cov["model_versions"],
        "scoring_versions_represented": cov["scoring_versions"],
        "feature_coverage": cov["features"],
        "unavailable_features": cov["unavailable_features"],
        "outcome_coverage": cov["horizons"],
        "benchmark_coverage": cov["benchmark"],
        "excess_coverage": cov["excess"],
        "mfe_coverage": cov["mfe_5d"],
        "mae_coverage": cov["mae_5d"],
        "unsupported_horizons": cov["unsupported_horizons"],
        "scan_feature_join_rate": cov["scan_feature_join_rate"],
        "filters": meta.get("filters"),
        "registry_metadata_counts": {k: meta.get(k) for k in ("observation_count", "matured_count", "certified_count")},
    }

    registry = {}
    try:
        registry = _registry_summary()
    except Exception as e:
        registry = {"error": f"{type(e).__name__}: {e}"}

    ai_model, ai_meta = None, {}
    try:
        from scan.ai_confidence import load_ai_confidence_bundle

        ai_model, ai_meta, warn = load_ai_confidence_bundle()
        _log(f"AI Confidence model loaded={ai_model is not None} trained_at={ai_meta.get('trained_at')} warn={warn}")
    except Exception as e:
        _log(f"AI Confidence model unavailable: {type(e).__name__}: {e}")

    spy = {}
    try:
        spy = _spy_closes()
        _log(f"SPY closes: {len(spy)}")
    except Exception as e:
        _log(f"SPY closes unavailable: {type(e).__name__}")

    legacy: dict = {}
    pb = [r for r in (registry.get("prebreakout_models") or []) if str(r.get("model_version")) == "prebreakout-xgb-v1"]
    if pb:
        best = max(pb, key=lambda r: r.get("auc_column") or 0)
        legacy["registry_legacy_auc"] = {"model": best["model_version"], "registry_id": best["id"],
                                         "trained_at": best["trained_at"], "auc": best["auc_column"],
                                         "all_v1_aucs": [r.get("auc_column") for r in pb]}
    try:
        _log("loading runs history for legacy reproduction")
        from ml_prebreakout import add_forward_return_labels, load_run_history

        hist = load_run_history(days_back=90, max_runs=2000)
        _log(f"runs history rows={len(hist)}")
        cols = ["Symbol", "Timestamp", "run_label", "IsBreakout", *A.AI_CONFIDENCE_FEATURE_MAP]
        base = hist[[c for c in cols if c in hist.columns]].copy()
        legacy["v1_recipes"] = A.legacy_v1_recipes(base)
        _log("v1 recipes done")
        labeled = add_forward_return_labels(hist, lookback_days=90)
        frame = labeled[[c for c in cols + ["ForwardReturnHit"] if c in labeled.columns]].copy()
        legacy["ai_confidence_current_recipe"] = {
            "split_diagnostics": A.legacy_split_diagnostics(frame),
            "variants": A.legacy_variants(frame)}
        _log("current-recipe variants done")
    except Exception as e:
        legacy["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    _log("running audit pipeline")
    res = P.run(raw_rows=rows_asof, scan_rows=window["scans"], manifest=manifest, members=members,
                ai_model=ai_model, ai_meta=ai_meta, spy_closes=spy, legacy=legacy)
    res["models"] = registry
    written = R.write_all(res, OUT)
    (OUT / "ml_v3_model_registry.json").write_text(A.dumps(registry), encoding="utf-8")
    written.append(OUT / "ml_v3_model_registry.json")
    _log(f"wrote {len(written)} files in {time.time() - t0:.0f}s")

    q = res["quality"]
    print(json.dumps({"dataset_version": manifest["dataset_version"], "quality": q["verdict"],
                      "blockers": q["blockers"], "observations": manifest["observation_count"],
                      "matured": manifest["matured_count"], "benchmark_5d": manifest["benchmark_coverage"].get("5d"),
                      "horizons": {h: {"n": v["n"], "folds": len(v["folds"]),
                                       "auc": {n: (r.get("pooled") or {}).get("roc_auc")
                                               for n, r in v["results"].items()}}
                                   for h, v in res["horizons"].items()}}, indent=1, default=str))

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in written:
            tar.add(p, arcname=p.name)
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(BUNDLE_BEGIN)
    for i in range(0, len(b64), 2000):
        print(b64[i:i + 2000])
    print(BUNDLE_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
