"""HSF ML v4: walk-forward evaluation + calibration of every model on identical rows.

    workflow_dispatch "ML v4 Train & Evaluate" (mode=evaluate)   or   python scripts/ml_v4_evaluate.py

Read-only, like the training dry run: SELECTs a finalized research dataset
version (as it stood at its freeze instant), the frozen AI Confidence value per
observation, and SPY daily closes for regime labels. Writes report files to
--out only; never writes to the database, never saves or swaps a model.

Three evaluations, all leakage-safe (purged, embargoed walk-forward folds, a
final holdout evaluated once, calibrators fitted only on earlier folds):
  1. primary target, every provider x feature set + baselines, all eligible rows;
  2. head-to-head with production AI Confidence AS SERVED, on the rows where
     that frozen value exists (identical rows for every model);
  3. secondary targets by horizon (beat SPY, return > 0 at 1/3/5 days), compact.
"""
from __future__ import annotations

import argparse
import datetime as dt
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


def _served_ai_confidence(ids) -> dict:
    """SELECT only: the AI Confidence % frozen on each observation at fire time."""
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, ai_confidence FROM signal_outcomes WHERE id = ANY(%s) AND ai_confidence IS NOT NULL",
                    ([int(i) for i in ids],))
        cols = [d[0] for d in cur.description]
        out = {}
        for r in cur.fetchall():
            r = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
            out[int(r["id"])] = float(r["ai_confidence"])
        return out
    except Exception as e:
        conn.rollback()
        T._log(f"AI Confidence read failed ({type(e).__name__}); head-to-head skipped")
        return {}
    finally:
        cur.close()
        conn.close()


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
    loaded = T.load_audit_rows(args.dataset_version)
    rows, entry = loaded["audit_rows"], loaded["entry"]
    served = _served_ai_confidence([r.observation_id for r in rows])
    matched = E.attach_served(rows, served)
    spy = _spy_closes()
    regime = A.spy_regime_by_day(spy, sorted({r.entry_day for r in rows})) if spy else None
    T._log(f"rows={len(rows)} ai_confidence_frozen={matched} regime_days={len(regime or {})}")

    target = P.PRIMARY_TARGET
    data = A.eligible(rows, target.horizon, target.label)
    common = dict(cost_bps=args.cost_bps, regime_by_day=regime)
    T._log(f"evaluating {target.label}@{target.horizon}d on {len(data)} eligible signal-days")
    ensembles = E.default_ensembles(providers=providers, feature_sets=feature_sets)
    primary = E.evaluate(data, E.default_models(providers=providers, feature_sets=feature_sets),
                         ensembles=ensembles, **common)
    T._log("head-to-head with AI Confidence as served")
    h2h = E.evaluate(data, E.default_models(providers=providers, feature_sets=feature_sets, include_champion=True),
                     reference="ai_confidence_as_served", ensembles=ensembles, **common)
    T._log("secondary targets by horizon")
    secondary = E.evaluate_targets(rows, P.SECONDARY_TARGETS,
                                   E.default_models(providers=providers, feature_sets=feature_sets), **common)

    report = {
        "evaluation_version": E.EVALUATION_VERSION, "engine_version": P.ENGINE_VERSION,
        "mode": "read-only (no database writes, no model saved)",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "git_sha": os.getenv("GITHUB_SHA"),
        "dataset": {"dataset_version": entry["dataset_version"], "dataset_fingerprint": entry.get("fingerprint"),
                    "frozen_at": str(loaded["frozen_at"]), "members": len(loaded["members"])},
        "library_versions": P.library_versions(),
        "primary": primary, "head_to_head_ai_confidence": h2h, "secondary_targets": secondary,
        "seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
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
    T._log(f"wrote reports to {out} in {report['seconds']}s")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
