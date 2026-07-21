"""Diagnose the empty 'Top market gappers' in the morning digest.

Checks, against prod, what the digest actually has to work with: whether a
snapshot DataFrame loads, whether it carries a GapPct column, and what each
gappers path returns. Read-only. Run via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from scheduler import morning_digest as md

    print("=" * 70)
    print("MORNING DIGEST — gappers diagnosis")
    print("=" * 70)

    df = md._latest_snapshot_df()
    if df is None:
        print("snapshot df: None  <-- root cause (no snapshot at digest time)")
    else:
        cols = list(getattr(df, "columns", []))
        print(f"snapshot df: {len(df)} rows")
        print(f"  columns: {cols}")
        print(f"  has GapPct: {'GapPct' in cols}")
        if "GapPct" in cols:
            import pandas as pd

            g = pd.to_numeric(df["GapPct"], errors="coerce")
            print(f"  GapPct non-null: {int(g.notna().sum())} / {len(g)}")
            print(f"  GapPct sample: {g.dropna().head(5).tolist()}")
        sym_col = md._symbol_column(df)
        print(f"  symbol column: {sym_col!r}")

    live = md._market_gappers(df)
    fb = md._gappers_from_snapshot(df) if df is not None else []
    print(f"\n_market_gappers(df) -> {len(live)} rows")
    for r in live[:5]:
        print(f"    {r}")
    print(f"_gappers_from_snapshot(df) -> {len(fb)} rows")
    for r in fb[:5]:
        print(f"    {r}")

    print("\nfallback wired:", hasattr(md, "_gappers_from_snapshot"))

    # Is a snapshot present at all in recent runs?
    try:
        from db.runs import list_runs

        runs = list_runs(limit=10) or []
        snaps = [r for r in runs if r.get("is_snapshot")]
        print(f"\nrecent runs: {len(runs)}  (snapshots among them: {len(snaps)})")
        for r in runs[:6]:
            print(f"    id={r.get('id')} snap={r.get('is_snapshot')} "
                  f"rows={r.get('row_count')} name={r.get('name')!r} at={r.get('created_at')}")
    except Exception as e:
        print(f"list_runs failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
