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

    # Inspect the actual fetched frames — downloaded_count only counts keys, not
    # whether each frame is usable (non-empty, has Close, enough history).
    try:
        from scan.headless_common import build_filtered_price_data, fetch_headless_prices

        price_data, skipped, elapsed = fetch_headless_prices(
            list(UNIVERSE), period="60d", interval="1d",
            use_parallel=True, parallel_chunk=800, parallel_workers=4,
        )
        print(f"\nfetch_headless_prices: {len(price_data)} frames, {len(skipped)} skipped, {elapsed:.2f}s")
        nonempty = {k: v for k, v in price_data.items() if v is not None and len(v) > 0}
        print(f"  non-empty frames: {len(nonempty)} / {len(price_data)}")
        for k, v in list(price_data.items())[:4]:
            cols = list(getattr(v, "columns", []))
            print(f"    {k}: rows={0 if v is None else len(v)} cols={cols[:6]}")
        filtered = build_filtered_price_data(
            price_data, min_price=5.0, max_price=1000.0, min_dollar_vol=2_000_000,
        )
        print(f"  after price/liquidity filter: {len(filtered)} frames")
    except Exception as e:
        print(f"fetch/filter probe error: {type(e).__name__}: {e}")

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
