"""Ablation: does enriching AI Confidence features and/or sweeping the target
raise its walk-forward AUC?

Read-only measurement — trains many (feature set x target) combinations with the
same walk-forward validation the production models use and prints a ranked
table. Ships nothing; we decide what (if anything) to promote from the numbers.

Run in an environment with Neon + Alpaca secrets set. Usage:
    python scripts/ablate_ai_confidence.py --days-back 90 --max-runs 1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml_prebreakout  # noqa: E402
from ml_prebreakout import (  # noqa: E402
    ATR_COMPRESSION_FEATURE_COLS,
    BASE_FEATURE_COLS,
    BOLLINGER_COMPRESSION_FEATURE_COLS,
    BREAKOUT_POSITION_EVOLUTION_FEATURE_COLS,
    REGIME_FEATURE_COLS,
    RELATIVE_STRENGTH_FEATURE_COLS,
    SPY_RELATIVE_STRENGTH_FEATURE_COLS,
    TARGET_COLUMN,
    VOLUME_VOLATILITY_EVOLUTION_FEATURE_COLS,
    _download_label_bars,
    _new_prebreakout_classifier,
    add_forward_return_labels_variant,
    add_prebreakout_features,
    classification_diagnostics,
    load_benchmark_regime_context,
    load_run_history,
    walk_forward_split,
)

try:
    import pandas as pd
    from sklearn.metrics import roc_auc_score
except Exception as exc:  # pragma: no cover
    print(f"ablation needs pandas + scikit-learn: {exc}")
    raise SystemExit(1)


# Candidate feature sets: baseline (current production 6) plus enrichments that
# made PreBreakout work. Each is baseline + one family, then everything.
def _dedup(cols: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return list(seen.keys())


FEATURE_SETS = {
    "baseline (6)": list(BASE_FEATURE_COLS),
    "+regime": _dedup(BASE_FEATURE_COLS + REGIME_FEATURE_COLS),
    "+rel_strength": _dedup(BASE_FEATURE_COLS + RELATIVE_STRENGTH_FEATURE_COLS + SPY_RELATIVE_STRENGTH_FEATURE_COLS),
    "+compression": _dedup(BASE_FEATURE_COLS + ATR_COMPRESSION_FEATURE_COLS + BOLLINGER_COMPRESSION_FEATURE_COLS),
    "+position": _dedup(BASE_FEATURE_COLS + BREAKOUT_POSITION_EVOLUTION_FEATURE_COLS),
    "all_enriched": _dedup(
        BASE_FEATURE_COLS
        + REGIME_FEATURE_COLS
        + RELATIVE_STRENGTH_FEATURE_COLS
        + SPY_RELATIVE_STRENGTH_FEATURE_COLS
        + ATR_COMPRESSION_FEATURE_COLS
        + BOLLINGER_COMPRESSION_FEATURE_COLS
        + BREAKOUT_POSITION_EVOLUTION_FEATURE_COLS
        + VOLUME_VOLATILITY_EVOLUTION_FEATURE_COLS
    ),
}

# Target sweep: thresholds/horizon for "+X% before -Y% within H trading days".
TARGET_SPECS = [
    {"name": "A +4/-2 5d (current)", "upside": 0.04, "downside": -0.02, "horizon": 5},
    {"name": "B +3/-2 5d", "upside": 0.03, "downside": -0.02, "horizon": 5},
    {"name": "C +4/-2 10d", "upside": 0.04, "downside": -0.02, "horizon": 10},
    {"name": "D +5/-2 5d", "upside": 0.05, "downside": -0.02, "horizon": 5},
    {"name": "E +4/-1.5 5d", "upside": 0.04, "downside": -0.015, "horizon": 5},
]


def _evaluate(labeled: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    cols = [c for c in feature_cols if c in labeled.columns]
    if len(cols) < 2 or TARGET_COLUMN not in labeled.columns:
        return None
    X = labeled[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = labeled[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        return None
    x_train, x_val, y_train, y_val = walk_forward_split(X, y, labeled)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return None
    model = _new_prebreakout_classifier()
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_val)[:, 1]
    diag = classification_diagnostics(y_val, proba)
    return {
        "auc": float(roc_auc_score(y_val, proba)),
        "lift10": diag.get("lift_over_baseline"),
        "top10_hit": diag.get("top_10pct_hit_rate"),
        "base_rate": diag.get("baseline_hit_rate"),
        "features": len(cols),
        "val_rows": int(len(y_val)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--max-runs", type=int, default=1500)
    args = parser.parse_args()

    # ml_prebreakout lazy-loads xgboost/sklearn; the classifier factory assumes
    # they're already loaded (the training entrypoints call this first).
    ml_prebreakout._load_ml_libs()

    history = load_run_history(days_back=args.days_back, max_runs=args.max_runs)
    if history is None or history.empty:
        print("No scan history available.")
        return 1
    print(f"[ablate] history: {len(history)} rows")

    benchmark_context = load_benchmark_regime_context(args.days_back)
    featured = add_prebreakout_features(history, benchmark_context=benchmark_context, include_market_features=True)

    symbols = sorted({str(s).upper() for s in featured.get("Symbol", pd.Series(dtype=str)).dropna() if str(s).strip()})
    bars = _download_label_bars(symbols, lookback_days=args.days_back) if symbols else {}
    if not bars:
        print("[ablate] WARNING: no label bars; targets fall back to close-return labels.")

    results = []
    for spec in TARGET_SPECS:
        labeled = add_forward_return_labels_variant(
            featured,
            horizon_days=int(spec["horizon"]),
            hit_threshold=float(spec["upside"]),
            stop_threshold=float(spec["downside"]),
            lookback_days=args.days_back,
            label_column=TARGET_COLUMN,
            bars_by_symbol=bars or None,
            force_path=bool(bars),
        )
        if labeled.empty:
            print(f"[ablate] target {spec['name']}: no labeled rows, skipping")
            continue
        pos_rate = float(labeled[TARGET_COLUMN].mean()) if TARGET_COLUMN in labeled.columns else 0.0
        for set_name, cols in FEATURE_SETS.items():
            res = _evaluate(labeled, cols)
            if not res:
                continue
            res.update({"target": spec["name"], "feature_set": set_name, "pos_rate": pos_rate})
            results.append(res)
            print(
                f"[ablate] {spec['name']:<20} | {set_name:<16} | "
                f"AUC={res['auc']:.4f} feats={res['features']:>2} "
                f"top10={(res['top10_hit'] or 0):.3f} val={res['val_rows']}"
            )

    if not results:
        print("[ablate] no results")
        return 1

    results.sort(key=lambda r: r["auc"], reverse=True)
    baseline = next(
        (r for r in results if r["feature_set"] == "baseline (6)" and r["target"].startswith("A ")),
        None,
    )
    base_auc = baseline["auc"] if baseline else None
    print("\n===== RANKED (best AUC first) =====")
    print(f"{'AUC':>7}  {'d_vs_base':>9}  {'feats':>5}  target / feature_set")
    for r in results[:15]:
        delta = f"{r['auc'] - base_auc:+.4f}" if base_auc is not None else "  n/a"
        print(f"{r['auc']:.4f}  {delta:>9}  {r['features']:>5}  {r['target']} / {r['feature_set']}")
    if base_auc is not None:
        print(f"\nProduction baseline (A / baseline 6 feats): AUC={base_auc:.4f}")
    best = results[0]
    print(f"Best: AUC={best['auc']:.4f}  ->  {best['target']} / {best['feature_set']} ({best['features']} feats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
