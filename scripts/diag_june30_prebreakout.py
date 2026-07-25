"""Diagnostic: why was the PreBreakout 5D heatmap cell for ~Jun 30 so negative?

Finds the snapshot on/nearest a target scan date, takes that day's top-5
PreBreakout picks, and prints each pick's 5-trading-day forward return, SPY's
forward return over the same window, and the per-name excess — so we can see
whether the deep-red day was one blowup or a broad-based week.

Read-only; run headless via the diagnostics workflow. Reuses the exact machinery
(_eligible_snapshots / _ranked_symbols / _forward_return) the heatmap is built on.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = dt.date(2026, 6, 30)
HORIZON = 5
RANKING = "prebreakout"


def main() -> None:
    from analytics import track_record as tr

    print("=" * 72)
    print(f"PreBreakout {HORIZON}D — diagnosing the {TARGET} heatmap cell")
    print("=" * 72)

    # Widen the lookback so a late-June snapshot is in range from 'now'.
    days_back = (dt.date.today() - TARGET).days + HORIZON + 10
    snaps = tr._eligible_snapshots(HORIZON, max(days_back, 45), 120)
    print(f"eligible {HORIZON}D snapshots in range: {len(snaps)}")
    if not snaps:
        print("no eligible snapshots — nothing to diagnose")
        return

    # Nearest snapshot to the target date.
    run_date, df = min(snaps, key=lambda rd_df: abs((rd_df[0] - TARGET).days))
    print(f"nearest snapshot: {run_date} (target {TARGET}), rows={len(df)}")

    picks = tr._ranked_symbols(df, RANKING, tr.TOP_N)
    print(f"PreBreakout top-{tr.TOP_N} picks: {picks}")
    if not picks:
        print("no PreBreakout picks that day (model returned no usable ranking).")
        return

    from data.price_alpaca import download_multi_alpaca

    syms = sorted(set(picks) | {tr.BENCHMARK})
    bars = download_multi_alpaca(syms, period="90d", interval="1d",
                                 prepost=False, timeout_s=25.0) or {}
    spy_bars = bars.get(tr.BENCHMARK)
    if spy_bars is None:
        print("no SPY bars — cannot compute excess")
        return
    spy_ret = tr._forward_return(spy_bars, run_date, HORIZON, "close")
    print(f"SPY {HORIZON}D forward return from {run_date}: "
          f"{('%+.2f%%' % (spy_ret * 100)) if spy_ret is not None else 'n/a'}")
    print("-" * 72)
    print(f"{'ticker':<8}{'fwd 5D':>12}{'excess vs SPY':>16}")
    print("-" * 72)

    excesses = []
    for sym in picks:
        b = tr._bars_for(bars, sym)
        r = tr._forward_return(b, run_date, HORIZON, "close") if b is not None else None
        if r is None or spy_ret is None:
            print(f"{sym:<8}{'no data':>12}{'—':>16}")
            continue
        exc = r - spy_ret
        excesses.append((sym, exc))
        print(f"{sym:<8}{('%+.2f%%' % (r * 100)):>12}{('%+.2f%%' % (exc * 100)):>16}")

    print("-" * 72)
    if excesses:
        mean_exc = sum(e for _s, e in excesses) / len(excesses)
        worst = min(excesses, key=lambda x: x[1])
        print(f"day mean excess: {mean_exc * 100:+.2f}%   "
              f"(this is the heatmap cell value for {run_date})")
        print(f"worst pick: {worst[0]} {worst[1] * 100:+.2f}% excess")
        # How much of the day is one name?
        rest = [e for s, e in excesses if s != worst[0]]
        if rest:
            rest_mean = sum(rest) / len(rest)
            print(f"mean excess excluding {worst[0]}: {rest_mean * 100:+.2f}%  "
                  f"→ {'one-name blowup' if rest_mean > mean_exc + 0.005 else 'broad-based'}")


if __name__ == "__main__":
    main()
