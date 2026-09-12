"""Read-only health check for the signal_outcomes pipeline in Neon.

Answers: are signals being frozen, and are their forward outcomes being
backfilled? Prints row counts, the fired_at range, completed-vs-pending, a
per-source breakdown, and the 30-day scorecard. Changes nothing.

Run with the Neon secret set (NEON_DATABASE_URL / DATABASE_URL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from db.engine import get_neon_conn
    from db.signal_outcomes import summarize_recent_outcomes

    conn = get_neon_conn()
    if conn is None:
        print("[check] No Neon connection (NEON_DATABASE_URL / DATABASE_URL unset).")
        return 1

    cur = conn.cursor()

    def scalar(sql):
        cur.execute(sql)
        row = cur.fetchone()
        return (row[0] if not isinstance(row, dict) else list(row.values())[0]) if row else None

    total = scalar("SELECT COUNT(*) FROM signal_outcomes")
    if total is None:
        print("[check] signal_outcomes table not found or unreadable.")
        return 1
    completed = scalar("SELECT COUNT(*) FROM signal_outcomes WHERE outcome_computed_at IS NOT NULL")
    pending = scalar("SELECT COUNT(*) FROM signal_outcomes WHERE outcome_computed_at IS NULL")
    last7 = scalar("SELECT COUNT(*) FROM signal_outcomes WHERE fired_at >= NOW() - make_interval(days => 7)")
    oldest = scalar("SELECT MIN(fired_at) FROM signal_outcomes")
    newest = scalar("SELECT MAX(fired_at) FROM signal_outcomes")
    last_backfill = scalar("SELECT MAX(outcome_computed_at) FROM signal_outcomes")

    print("=== signal_outcomes health ===")
    print(f"total rows:         {total}")
    print(f"completed (scored): {completed}")
    print(f"pending (maturing): {pending}")
    print(f"fired last 7 days:  {last7}")
    print(f"oldest fired_at:    {oldest}")
    print(f"newest fired_at:    {newest}")
    print(f"last backfill run:  {last_backfill}")

    cur.execute(
        "SELECT source AS src, COUNT(*) AS n, "
        "COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL) AS scored "
        "FROM signal_outcomes GROUP BY source ORDER BY n DESC"
    )
    print("\nby source (source | rows | scored):")
    for r in cur.fetchall() or []:
        if isinstance(r, dict):
            src, n, done = r.get("src"), r.get("n"), r.get("scored")
        else:
            src, n, done = r[0], r[1], r[2]
        print(f"  {src or '(none)'}: {n} | {done}")
    cur.close()
    conn.close()

    print("\n=== 30-day scorecard (what the brief renders) ===")
    print(json.dumps(summarize_recent_outcomes(days_back=30), indent=2, default=str))

    # Verdict
    print("\n=== verdict ===")
    if total == 0:
        print("EMPTY: no signals frozen yet — freeze step may not have run since deploy.")
    elif completed == 0:
        print("FREEZING, NOT YET SCORED: rows exist but no forward outcomes backfilled "
              "yet (need signals >=5 trading days old + a backfill run).")
    else:
        print(f"HEALTHY: {completed}/{total} signals scored; pipeline is accruing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
