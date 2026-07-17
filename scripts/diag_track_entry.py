"""Diagnostic: is the negative track record entry-timing, regime, or selection?

Recomputes the signal track record under two entry conventions and prints them
side by side:

  close — enter at the signal-day CLOSE (production default). For premarket
          snapshots this discards the entire signal-day move.
  open  — enter at the signal-day OPEN (the realistic premarket-signal fill),
          which captures the move the breakout signal is designed to catch.

If `open` flips the excess materially toward/above zero, the negative number is
a methodology artifact (entering after the move). If both stay negative, it's
regime/selection and a public track-record page should wait.

Also dumps the 5D per-day excess series so a few dominant days (a regime blip)
are distinguishable from a persistent drag. Read-only; run headless via the
diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZONS = (1, 5, 20)
MODES = ("close", "open")


def _fmt(summary) -> str:
    return (
        f"avg={summary['avg_return']:+.2%}  med={summary['median_return']:+.2%}  "
        f"beat={summary['win_rate']:.0%}  n={summary['sample_size']}  "
        f"runs={summary['runs_used']}"
    )


def main() -> None:
    from analytics.track_record import compute_track_record

    print("=" * 78)
    print("TRACK RECORD — entry-timing A/B (close vs open)")
    print("=" * 78)

    for horizon in HORIZONS:
        print(f"\n### Horizon {horizon}D")
        by_mode = {}
        for mode in MODES:
            by_mode[mode] = compute_track_record(horizon_days=horizon, entry_mode=mode)
        for ranking in ("breakout", "prebreakout"):
            print(f"  {ranking}:")
            for mode in MODES:
                res = by_mode.get(mode) or {}
                summary = res.get(ranking)
                if summary:
                    print(f"    entry={mode:<5} {_fmt(summary)}")
                else:
                    print(f"    entry={mode:<5} (no data)")

    # Per-day 5D excess distribution: regime blip vs persistent drag.
    print("\n" + "=" * 78)
    print("5D per-day excess (prebreakout, close entry) — oldest→newest")
    print("=" * 78)
    res = compute_track_record(horizon_days=5, entry_mode="close") or {}
    daily = (res.get("prebreakout") or {}).get("daily") or []
    if not daily:
        print("  (no daily series)")
    else:
        pos = sum(1 for _d, mx, _n in daily if mx > 0)
        for run_date, mean_excess, n_picks in daily:
            bar = "+" if mean_excess > 0 else "-"
            print(f"  {run_date}  {mean_excess:+.2%}  (n={n_picks}) {bar}")
        worst = min(daily, key=lambda t: t[1])
        best = max(daily, key=lambda t: t[1])
        print(
            f"\n  days_positive={pos}/{len(daily)}  "
            f"worst={worst[0]} {worst[1]:+.2%}  best={best[0]} {best[1]:+.2%}"
        )


if __name__ == "__main__":
    main()
