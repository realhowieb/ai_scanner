"""Read-only HSF Opportunity Score calibration report.

Prints the dataset range, matured/pending counts, score-version distribution,
score-bucket table, status table, signal-type table, baseline comparison,
monotonicity result, calibration diagnostics, and data-quality warnings — all
from real stored outcomes. Writes nothing. Says so plainly when there is not
enough matured data yet.

Run with the Neon secret set (NEON_DATABASE_URL / DATABASE_URL).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt_rate(r):
    return f"{r*100:.0f}%" if r is not None else "—"


def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "—"


def main() -> int:
    from analytics import hsf_calibration as hc

    ds = hc.build_calibration_dataset(days_back=365)
    lo, hi = ds["date_range"]
    print("=== HSF SCORE CALIBRATION REPORT ===")
    print(f"frozen opportunities : {ds['n_total']}")
    print(f"matured outcomes     : {ds['n_matured']}")
    print(f"pending (maturing)   : {ds['n_pending']}")
    print(f"date range           : {lo} → {hi}")
    print(f"score versions       : {ds['versions']}")
    if ds["quality"]:
        print("data-quality warnings:")
        for w in ds["quality"]:
            print(f"  ⚠ {w}")

    matured = ds["matured"]
    if not matured:
        print("\nINSUFFICIENT DATA: no matured HSF opportunities yet — infrastructure "
              "is in place; the report populates as forward windows complete.")
        return 0

    print("\n--- Score buckets (positive outcome = reached +4% in 5D) ---")
    buckets = hc.summarize_score_buckets(matured)
    print(f"{'bucket':>8} {'n_mat':>6} {'pos%':>6} {'medMFE':>8} {'medMAE':>8}  confidence")
    for b in buckets:
        print(f"{b['bucket']:>8} {b['n_matured']:>6} {_fmt_rate(b['positive_rate']):>6} "
              f"{_fmt(b['median_mfe_5d']):>8} {_fmt(b['median_mae_5d']):>8}  {b['confidence']}")

    mono = hc.evaluate_monotonicity(buckets)
    print(f"\nMonotonic (higher score -> better outcome)? {mono['monotonic']} "
          f"(rank corr {mono.get('rank_correlation')})  [{mono['confidence']}]")

    print("\n--- Status ---")
    for s in hc.summarize_status_performance(matured):
        print(f"  {s['status']:>8}: n={s['n_matured']:>4} pos={_fmt_rate(s['positive_rate'])} "
              f"medMFE={_fmt(s['median_mfe_5d'])}  [{s['confidence']}]")

    print("\n--- Signal types ---")
    for s in hc.summarize_signal_types(matured):
        print(f"  {s['signal_type']:>14}: n={s['n_matured']:>4} pos={_fmt_rate(s['positive_rate'])}  [{s['confidence']}]")

    combos = hc.analyze_signal_combinations(matured)
    if combos:
        print("\n--- Signal combinations (n>=10) ---")
        for c in combos:
            print(f"  {c['combination']:>22}: n={c['n_matured']:>4} pos={_fmt_rate(c['positive_rate'])}")

    print("\n--- Baseline discrimination (AUC vs positive outcome) ---")
    for b in hc.compare_baselines(matured):
        print(f"  {b['ranker']:>16}: AUC={b['auc']}  n={b['n']}  [{b['confidence']}]")

    cal = hc.evaluate_calibration(matured)
    print(f"\nCalibration: Brier={cal['brier']}  n={cal['n']}  [{cal['confidence']}]")

    dist = hc.score_distribution(ds["records"])
    print(f"\nScore distribution: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
