"""Freeze Scanner-scored HSF opportunities for past runs (dry run by default).

Opportunity rows used to be frozen from the Market Brief's top-5 lists, whose
HSF scores came out lower than the Scanner's for the same name (CCC 50 vs 77),
so the Track record "By HSF score" buckets were built on scores nobody saw.
The cron now freezes each run's Scanner ranking; this re-freezes past saved
full-market runs the same way (analytics.opportunity_freeze.freeze_run), at the
run's own created_at. Readers then drop that snapshot's Brief rows.

    python scripts/backfill_scanner_opportunities.py              # dry run: counts only
    python scripts/backfill_scanner_opportunities.py --apply      # writes rows

--apply writes production data, so run it only on the owner's go. It only
INSERTs new rows (ON CONFLICT DO NOTHING); Brief rows stay in the table for
audit. Features come from the run's saved results (signal-time data), never
from prices after the run. Outcomes are filled by the normal outcome cron,
which downloads 60 days of bars, so --days stays under that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from analytics.opportunity_freeze import FREEZE_TOP_N, freeze_run, market_runs, scanner_opportunities

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the rows (production write)")
    ap.add_argument("--days", type=int, default=45, help="runs from the last N days (max 50)")
    ap.add_argument("--top-n", type=int, default=FREEZE_TOP_N)
    args = ap.parse_args(argv)
    days = max(1, min(50, int(args.days)))
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    runs = [r for r in market_runs(limit=1000) if r["created_at"] >= since]
    print(f"runs in the last {days} days: {len(runs)}")
    total = empty = 0
    for run in sorted(runs, key=lambda r: r["created_at"]):
        if args.apply:
            n = freeze_run(run, args.top_n, score_confidence=False)  # never today's model on past rows
        else:
            n = len(scanner_opportunities(int(run["id"]), args.top_n))
        empty += n == 0
        total += n
    verb = "froze" if args.apply else "would freeze"
    print(f"{verb} {total} Scanner opportunity row(s); {empty} run(s) had no saved results or no setups")
    if not args.apply:
        print("dry run: nothing written (pass --apply on the owner's go)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
