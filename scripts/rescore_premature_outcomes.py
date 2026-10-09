"""Re-score opportunity outcomes that were written empty by mistake (dry run by default).

Before the open-window guard in analytics.signal_outcomes, the outcome cron
wrote an empty (unavailable) label whenever a row was 8 calendar days old but
its 5-day window had not closed yet (weekend snapshots), and that empty label
was final. analytics.ml_readiness classifies those rows PREMATURE_LABEL_WRITE
(and DATA_PROVIDER_FAILURE when today's bars can label them).

    python scripts/rescore_premature_outcomes.py            # dry run: counts only
    python scripts/rescore_premature_outcomes.py --apply    # writes labels

--apply writes production data, so run it only on the owner's go. It fills a
row only while its 1/3/5-day returns are all still NULL (never overwrites a
real label), uses the same score_signal on finished sessions only, and fills
the SPY benchmark the same way the cron does. Features are never touched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECOVER = ("PREMATURE_LABEL_WRITE", "DATA_PROVIDER_FAILURE")
_UPDATE = ("UPDATE signal_outcomes SET return_1d = %s, return_3d = %s, return_5d = %s, mfe_5d = %s, mae_5d = %s, "
           "outcome_computed_at = NOW() WHERE id = %s AND source = 'opportunity' "
           "AND return_1d IS NULL AND return_3d IS NULL AND return_5d IS NULL")


def main(argv=None) -> int:
    from analytics import ml_readiness as mr
    from analytics.signal_outcomes import BENCHMARK, _save_benchmark_for, complete_session_bars, score_signal
    from data.price_alpaca import download_multi_alpaca
    from db.engine import get_neon_conn
    from db.research_datasets import fetch_readiness_rows

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the labels (production write)")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    s = mr.DATASET_START
    rows = fetch_readiness_rows(dt.datetime(s.year, s.month, s.day, tzinfo=dt.timezone.utc), now)
    targets = [r for r in rows if mr.maturation_stage(r, now)["category"] in
               (mr.PREMATURE_LABEL_WRITE, mr.MISSING_PRICE_DATA)]
    print(f"candidates: {len(targets)} of {len(rows)} rows")
    if not targets:
        return 0
    days = min(1500, (now - min(r["fired_at"] for r in targets)).days + 30)
    tickers = sorted({str(r["ticker"]).upper() for r in targets} | {BENCHMARK})
    bars = download_multi_alpaca(tickers, period=f"{days}d", interval="1d", prepost=False, timeout_s=60.0) or {}
    bars = {k: complete_session_bars(v, now) for k, v in bars.items()}
    plan, skipped = [], Counter()
    for r in targets:
        b = bars.get(str(r["ticker"]).upper())
        out = score_signal(b, r["fired_at"]) if b is not None else None
        if out is None:
            skipped[str(r["ticker"]).upper()] += 1
            continue
        plan.append((r, out))
    entry_days = Counter(mr.maturation_timing(r["fired_at"])["entry_day"].isoformat() for r, _ in plan)
    print(f"scorable now: {len(plan)}; still unscorable: {sum(skipped.values())} {dict(skipped)}")
    print(f"recovered rows by entry day: {dict(sorted(entry_days.items()))}")
    if not args.apply:
        print("dry run: nothing written (pass --apply on the owner's go)")
        return 0
    conn = get_neon_conn()
    if conn is None:
        raise SystemExit("database unavailable")
    written = 0
    try:
        cur = conn.cursor()
        for r, o in plan:
            cur.execute(_UPDATE, (o["return_1d"], o["return_3d"], o["return_5d"], o["mfe_5d"], o["mae_5d"],
                                  int(r["id"])))
            written += cur.rowcount
        conn.commit()
        cur.close()
    finally:
        conn.close()
    bench = sum(1 for r, _ in plan if _save_benchmark_for(r, bars.get(BENCHMARK)))
    print(f"written: {written} labels, {bench} benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
