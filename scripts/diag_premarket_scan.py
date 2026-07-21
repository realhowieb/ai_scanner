"""Diagnose why cron premarket/postmarket scans return 0 results.

Runs the same small universe through the headless scan under each session label
and reports the universe size, fetched/filtered price counts, and final row
count — so we can see whether results vanish at universe load, price fetch,
filtering, or the session-specific (premarket/postmarket) breakout path.
Read-only. Run via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "AVGO",
            "GOOGL", "NFLX", "WMT", "JPM", "XOM", "GLW", "MU", "MRVL"]


def main() -> None:
    from scan import pre_post as pp

    print("=" * 70)
    print("PREMARKET / POSTMARKET SCAN DIAGNOSIS")
    print("=" * 70)

    # Universe loaders (what the cron's None-universe path resolves to)
    try:
        u600 = pp._load_sp600_or_sp500()
        u500 = pp._load_sp500()
        print(f"_load_sp600_or_sp500(): {len(u600)} symbols  (sample {u600[:5]})")
        print(f"_load_sp500(): {len(u500)} symbols")
        print(f"sp500.txt exists: {(ROOT / 'sp500.txt').exists()}")
    except Exception as e:
        print(f"universe load error: {type(e).__name__}: {e}")

    for session in ("regular", "premarket", "postmarket"):
        print(f"\n### session_label = {session!r}")
        try:
            df, meta = pp.run_scan(session, list(UNIVERSE), session_label=session)
            rows = 0 if df is None else len(df)
            print(f"  final rows: {rows}")
            print(f"  meta: downloaded={meta.get('downloaded_count')} "
                  f"skipped={meta.get('skipped_count')} elapsed={meta.get('elapsed_s')}")
            if rows:
                cols = list(df.columns)[:8]
                print(f"  columns: {cols}")
                if "Ticker" in df.columns:
                    print(f"  tickers: {df['Ticker'].head(5).tolist()}")
        except Exception as e:
            print(f"  run_scan error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
