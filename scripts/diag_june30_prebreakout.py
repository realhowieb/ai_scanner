"""Diagnostic: why is the PreBreakout 5D heatmap cell near Jun 30 so negative?

The heatmap draws from signal_track_daily (ranking, horizon_days, day,
avg_excess, n_picks) — precomputed by the cron. Recomputing live is misleading
because old snapshots age out, so this dumps the STORED daily series directly:
each day's mean excess and how many picks it was averaged over. A deep-red day
built on n_picks=1-2 is a thin-sample blip, not a broad signal failure.

Also lists the raw snapshot runs around late June so coverage gaps are visible.
Read-only; run headless via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZON = 5


def main() -> None:
    print("=" * 72)
    print("PreBreakout 5D — stored daily heatmap series (signal_track_daily)")
    print("=" * 72)

    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        print("no Neon connection")
        return
    cur = conn.cursor()

    for ranking in ("prebreakout", "breakout"):
        cur.execute(
            """
            SELECT day, avg_excess, n_picks, computed_at
            FROM signal_track_daily
            WHERE ranking = %s AND horizon_days = %s
            ORDER BY day ASC
            """,
            (ranking, HORIZON),
        )
        rows = cur.fetchall() or []
        print(f"\n--- {ranking} {HORIZON}D: {len(rows)} stored days ---")
        print(f"{'day':<12}{'avg_excess':>12}{'n_picks':>9}   computed_at")
        for r in rows:
            day = r["day"] if isinstance(r, dict) else r[0]
            exc = r["avg_excess"] if isinstance(r, dict) else r[1]
            n = r["n_picks"] if isinstance(r, dict) else r[2]
            ca = r["computed_at"] if isinstance(r, dict) else r[3]
            exc_s = f"{exc * 100:+.2f}%" if exc is not None else "—"
            flag = "  <-- deep red" if (exc is not None and exc < -0.03) else ""
            print(f"{str(day):<12}{exc_s:>12}{str(n):>9}   {ca}{flag}")

    # Snapshot coverage around late June (are there even scans those days?)
    print("\n--- snapshot runs, Jun 20 – Jul 10 (coverage) ---")
    cur.execute(
        """
        SELECT created_at::date AS d, COUNT(*) AS n, MAX(row_count) AS max_rows
        FROM runs
        WHERE is_snapshot = TRUE
          AND created_at::date BETWEEN DATE '2026-06-20' AND DATE '2026-07-10'
        GROUP BY created_at::date ORDER BY d
        """
    )
    for r in (cur.fetchall() or []):
        d = r["d"] if isinstance(r, dict) else r[0]
        n = r["n"] if isinstance(r, dict) else r[1]
        mx = r["max_rows"] if isinstance(r, dict) else r[2]
        print(f"  {d}: {n} snapshot(s), max_rows={mx}")

    cur.close()


if __name__ == "__main__":
    main()
