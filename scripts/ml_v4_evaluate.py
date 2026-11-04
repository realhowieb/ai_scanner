"""HSF ML v4: walk-forward evaluation + calibration of every model on identical rows.

    workflow_dispatch "ML v4 Train & Evaluate" (mode=evaluate)   or   python scripts/ml_v4_evaluate.py

Read-only by default: SELECTs a finalized research dataset
version (as it stood at its freeze instant), the frozen AI Confidence value per
observation, and SPY daily closes for regime labels. Writes report files to
--out only; never saves or swaps a model. Explicit --final-holdout writes a
durable claim and completed report in ml_holdout_evaluations BEFORE exposing
final-test performance. An exact completed report is reused on subsequent runs.

Three evaluations, all leakage-safe (purged, embargoed walk-forward folds, a
reserved final holdout withheld by default, calibrators fitted only on earlier folds):
  1. primary target, every provider x feature set + baselines, all eligible rows;
  2. head-to-head with production AI Confidence AS SERVED, on the rows where
     that frozen value exists (identical rows for every model);
  3. secondary targets by horizon (beat SPY, return > 0 at 1/3/5 days), compact.
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
sys.path.insert(0, str(ROOT / "scripts"))

import ml_v4_train as T  # noqa: E402  (shared read-only loader)


def _ensure_ml_libs() -> None:
    """Install requirements-ml-v4.txt when missing (e.g. run through the Diagnostics workflow)."""
    import importlib
    import subprocess

    try:
        for mod in ("sklearn", "xgboost", "lightgbm", "catboost"):
            importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--prefer-binary",
                        "-r", str(ROOT / "requirements-ml-v4.txt"), "-c", str(ROOT / "requirements.lock")], check=True)
        importlib.invalidate_caches()


def _served_ai_confidence(rows) -> dict:
    """Use the same frozen rows already verified by preflight; no second DB read."""
    return {int(r["id"]): float(r["ai_confidence"]) for r in rows if r.get("ai_confidence") is not None}


def _spy_closes() -> dict:
    try:
        from data.price_alpaca import download_multi_alpaca

        bars = download_multi_alpaca(["SPY"], period="400d", interval="1d", prepost=False, timeout_s=25.0)
        df = bars.get("SPY")
        if df is None or df.empty:
            return {}
        return {(ts.date() if hasattr(ts, "date") else ts): float(c) for ts, c in df["Close"].dropna().items()}
    except Exception as e:  # regime is a breakdown, not a requirement
        T._log(f"SPY closes unavailable ({type(e).__name__}); regime breakdown skipped")
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-version", default=os.getenv("ML_V4_DATASET_VERSION") or None)
    ap.add_argument("--providers", default=os.getenv("ML_V4_PROVIDERS") or "xgboost,lightgbm,catboost")
    ap.add_argument("--feature-sets", default=os.getenv("ML_V4_FEATURE_SETS") or "production,leakage_safe")
    ap.add_argument("--cost-bps", type=float, default=float(os.getenv("ML_V4_COST_BPS") or 20.0))
    ap.add_argument("--out", default=None)
    ap.add_argument("--git-sha", default=os.getenv("GITHUB_SHA"))
    ap.add_argument("--final-holdout", action="store_true", default=os.getenv("ML_V4_FINAL_HOLDOUT") == "true",
                    help="Explicit database write: reserve the final test in the durable ledger before scoring it")
    args = ap.parse_args(argv)
    _ensure_ml_libs()

    from analytics import ml_v3_audit as A
    from analytics import ml_v4_evaluation as E
    from analytics import ml_v4_providers as P

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    bad = [p for p in providers if p not in P.PROVIDERS] + [f for f in feature_sets if f not in P.FEATURE_SETS]
    if bad:
        raise SystemExit(f"unknown provider/feature set: {bad}")
    missing = [p for p in providers if p not in P.available_providers()]
    if missing:
        raise SystemExit(f"not installed: {missing}; pip install -r requirements-ml-v4.txt -c requirements.lock")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else ROOT / "ml" / "artifacts" / "ml_v4_eval" / stamp
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; reports are never overwritten")

    t0 = time.time()
    from analytics import ml_v4_preflight as PF

    try:
        loaded = T.load_audit_rows(args.dataset_version)
    except PF.PreparationError as e:
        PF.write_load_failure(out, e, args.dataset_version)
        raise
    preparation, split, evaluation_binding = PF.prepare(loaded, providers, feature_sets, git_sha=args.git_sha)
    # Costs affect report selection too; include them in the immutable binding.
    evaluation_binding["cost_bps"] = args.cost_bps
    out.mkdir(parents=True, exist_ok=True)
    (out / "ml_v4_preflight.json").write_text(A.dumps(preparation) + "\n")
    (out / "ml_v4_preflight.md").write_text(PF.markdown(preparation))
    print(PF.markdown(preparation))
    PF.require_ready(preparation)
    rows, entry = loaded["audit_rows"], loaded["entry"]
    served = _served_ai_confidence(loaded["raw_rows"])
    matched = E.attach_served(rows, served)
    spy = _spy_closes()
    regime = A.spy_regime_by_day(spy, sorted({r.entry_day for r in rows})) if spy else None
    T._log(f"rows={len(rows)} ai_confidence_frozen={matched} regime_days={len(regime or {})}")

    target = P.PRIMARY_TARGET
    data = A.eligible(rows, target.horizon, target.label)
    common = dict(cost_bps=args.cost_bps, regime_by_day=regime)
    T._log(f"evaluating {target.label}@{target.horizon}d on {len(data)} eligible signal-days")
    ensembles = E.default_ensembles(providers=providers, feature_sets=feature_sets)
    primary_models = E.default_models(providers=providers, feature_sets=feature_sets)
    head_models = E.default_models(providers=providers, feature_sets=feature_sets, include_champion=True)
    # Freeze a single final-test boundary for every comparison. Exclude it even
    # when the served-probability subset is too small to reserve its own test.
    holdout_ids = split["holdout"]["ids"]
    claim_key, cached = None, None
    if args.final_holdout:
        from db import ml_holdout_evaluations as ledger

        claim_key, cached = ledger.reserve(evaluation_binding, holdout_ids,
                                           run_id=os.getenv("GITHUB_RUN_ID") or args.git_sha)
    if cached:
        report = cached
    else:
        report = _evaluate_report(E, P, A, data, rows, primary_models, head_models, ensembles, common,
                                  loaded, entry, args, evaluation_binding, t0, split)
        if claim_key:
            report = json.loads(A.dumps(report))
            report["holdout_claim"] = claim_key
            ledger.complete(claim_key, evaluation_binding, report)
    _write_report(out, report, E, A)
    return 0


def _evaluate_report(E, P, A, data, rows, primary_models, head_models, ensembles, common,
                     loaded, entry, args, evaluation_binding, t0, split):
    from analytics.ml_v4_preflight import PreparationError

    primary = E.evaluate(data, primary_models, ensembles=ensembles,
                         evaluate_holdout=args.final_holdout, reserved_holdout=split["holdout"], **common)
    if (primary["sample"]["common_rows"] != len(data)
            or primary["training_slice_fingerprint"] != evaluation_binding["training_slice_fingerprint"]):
        raise PreparationError("Evaluation drops rows used by artifact training; refuse to attach mismatched evidence")
    T._log("head-to-head with AI Confidence as served")
    h2h = E.evaluate(data, head_models, reference="ai_confidence_as_served", ensembles=ensembles,
                     evaluate_holdout=args.final_holdout, reserved_holdout=split["holdout"], **common)
    T._log("secondary targets by horizon")
    # Secondary questions use development folds only, never the primary final test.
    secondary_rows = [r for r in rows if r.window_end[P.PRIMARY_TARGET.horizon].isoformat() < split["holdout"]["start"]]
    secondary = E.evaluate_targets(secondary_rows, P.SECONDARY_TARGETS, primary_models, evaluate_holdout=False, **common)

    report = {
        "evaluation_version": E.EVALUATION_VERSION, "engine_version": P.ENGINE_VERSION,
        "mode": "final holdout (durable ledger)" if args.final_holdout else "development (final test withheld)",
        "binding": evaluation_binding,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "git_sha": args.git_sha,
        "dataset": {"dataset_version": entry["dataset_version"], "dataset_fingerprint": entry.get("fingerprint"),
                    "frozen_at": str(loaded["frozen_at"]), "members": len(loaded["members"])},
        "library_versions": P.library_versions(),
        "primary": primary, "head_to_head_ai_confidence": h2h, "secondary_targets": secondary,
        "seconds": round(time.time() - t0, 1),
    }
    return report


def _write_report(out, report, E, A):
    primary, h2h, secondary = report["primary"], report["head_to_head_ai_confidence"], report["secondary_targets"]
    md = "\n".join([
        E.render_markdown(primary, title="ML v4 evaluation: all models, all eligible rows"),
        E.render_markdown(h2h, title="ML v4 evaluation: head-to-head with AI Confidence as served"),
        "# Secondary targets (walk-forward AUC, raw)", "",
        "| Target | Horizon | Rows | Folds | " + " | ".join(next((s["models"] for s in secondary if s.get("models")), {})) + " |",
    ])
    names = list(next((s["models"] for s in secondary if s.get("models")), {}))
    md += "\n|" + "---|" * (4 + len(names)) + "\n"
    for s in secondary:
        cells = [E._f((s.get("models") or {}).get(n, {}).get("roc_auc")) for n in names]
        md += f"| {s['target']} | {s['horizon_days']}d | {s['rows']} | {s.get('folds', 0)} | " + " | ".join(cells) + " |\n"
    with open(out / "ml_v4_evaluation.json", "x", encoding="utf-8") as fh:
        fh.write(A.dumps(report) + "\n")
    with open(out / "ml_v4_evaluation.md", "x", encoding="utf-8") as fh:
        fh.write(md)
    T._log(f"wrote reports to {out}")
    print(md)


if __name__ == "__main__":
    raise SystemExit(main())
