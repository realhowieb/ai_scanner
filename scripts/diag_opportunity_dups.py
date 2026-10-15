"""Read-only: where do duplicate opportunity observations come from?

Counts signal_outcomes source='opportunity' rows per ET day and signal_type
since 2026-10-06, and lists every (ticker, fired_at) group with more than one
row. Run via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        print("no Neon connection")
        return 1
    cur = conn.cursor()
    cur.execute(
        "SELECT (fired_at AT TIME ZONE 'America/New_York')::date AS d, signal_type, "
        "COUNT(*), COUNT(DISTINCT fired_at), COUNT(DISTINCT UPPER(ticker)) "
        "FROM signal_outcomes WHERE source = 'opportunity' AND fired_at >= '2026-10-06' "
        "GROUP BY 1, 2 ORDER BY 1, 2")
    print("day | signal_type | rows | snapshots | tickers")
    for r in cur.fetchall():
        print(" | ".join(str(x) for x in (r.values() if isinstance(r, dict) else r)))
    cur.execute(
        "SELECT UPPER(ticker) AS t, fired_at, COUNT(*) AS n, "
        "STRING_AGG(signal_type || ':' || ticker || ':' || source_event_id::text || ':' || created_at::text, ' ; ' ORDER BY id) "
        "FROM signal_outcomes WHERE source = 'opportunity' "
        "GROUP BY 1, 2 HAVING COUNT(*) > 1 ORDER BY 2")
    print("\nduplicate groups (ticker | fired_at | n | members type:ticker:event:created)")
    for r in cur.fetchall():
        print(" | ".join(str(x) for x in (r.values() if isinstance(r, dict) else r)))
    cur.execute(
        "SELECT fired_at, signal_type, COUNT(*), MIN(created_at), MAX(created_at) "
        "FROM signal_outcomes WHERE source = 'opportunity' AND fired_at >= '2026-10-12' "
        "GROUP BY 1, 2 ORDER BY 1, 2")
    print("\nsnapshots since 2026-10-12 (fired_at | type | rows | first write | last write)")
    for r in cur.fetchall():
        print(" | ".join(str(x) for x in (r.values() if isinstance(r, dict) else r)))
    cur.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
