"""P2-54: read-only Neon query statistics (pg_stat_statements).

Reports whether pg_stat_statements is installed, then the queries with the most
total time, the highest mean time and the most calls in this database, plus the
largest tables and current connections. Read-only: it only runs SELECTs.

pg_stat_statements stores normalized query text (literals become $1, $2 ...), so
no values reach the output; the text is still passed through redact() and cut
to a short preview because workflow logs are public. Stats reset when the Neon
compute restarts, so run it after a trading day.

    DATABASE_URL=... python scripts/neon_query_stats.py --top 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREVIEW = 140


def _clean(text: str) -> str:
    from ui.log_privacy import redact

    one_line = " ".join(str(text or "").split())
    one_line = redact(one_line)
    return one_line if len(one_line) <= PREVIEW else one_line[: PREVIEW - 1] + "…"


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall() or []


def collect(conn, top: int) -> dict:
    cur = conn.cursor()
    out: dict = {}
    ext = _rows(cur, "SELECT extversion FROM pg_extension WHERE extname = 'pg_stat_statements'")
    out["pg_stat_statements"] = ext[0][0] if ext else None
    out["neon_extension"] = bool(_rows(cur, "SELECT 1 FROM pg_extension WHERE extname = 'neon'"))
    out["connections"] = [
        {"state": r[0] or "-", "count": int(r[1])}
        for r in _rows(cur, "SELECT state, count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() GROUP BY state ORDER BY 2 DESC")
    ]
    out["largest_tables"] = [
        {"table": r[0], "size_mb": round(float(r[1]) / 1e6, 1), "rows_est": int(r[2] or 0)}
        for r in _rows(cur, "SELECT c.relname, pg_total_relation_size(c.oid), c.reltuples::bigint "
                            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                            "WHERE c.relkind = 'r' AND n.nspname = 'public' "
                            "ORDER BY 2 DESC LIMIT 10")
    ]
    if not out["pg_stat_statements"]:
        cur.close()
        return out
    base = ("SELECT s.query, s.calls, s.total_exec_time, s.mean_exec_time, s.rows "
            "FROM pg_stat_statements s JOIN pg_database d ON d.oid = s.dbid "
            "WHERE d.datname = current_database() AND s.query NOT ILIKE '%%pg_stat_statements%%' ")

    def block(order):
        return [
            {"query": _clean(r[0]), "calls": int(r[1]), "total_ms": round(float(r[2]), 1),
             "mean_ms": round(float(r[3]), 2), "rows": int(r[4])}
            for r in _rows(cur, base + f"ORDER BY {order} DESC LIMIT %s", (int(top),))
        ]

    out["by_total_time"] = block("s.total_exec_time")
    out["by_mean_time"] = block("s.mean_exec_time")
    out["by_calls"] = block("s.calls")
    reset = _rows(cur, "SELECT stats_reset FROM pg_stat_statements_info")
    out["stats_since"] = str(reset[0][0]) if reset else None
    cur.close()
    return out


def render(r: dict) -> str:
    lines = ["## Neon query stats (P2-54)", ""]
    v = r.get("pg_stat_statements")
    lines.append(f"- pg_stat_statements: **{'installed v' + v if v else 'NOT installed'}**"
                 f" · neon extension: {'yes' if r.get('neon_extension') else 'no'}"
                 + (f" · stats since {r['stats_since']}" if r.get("stats_since") else ""))
    lines.append("- connections: " + ", ".join(f"{c['state']}={c['count']}" for c in r.get("connections", [])))
    lines += ["", "**Largest tables**", "", "| Table | MB | Rows (est.) |", "|---|---:|---:|"]
    lines += [f"| {t['table']} | {t['size_mb']} | {t['rows_est']} |" for t in r.get("largest_tables", [])]
    for key, title in (("by_total_time", "Most total time"), ("by_mean_time", "Slowest on average"),
                       ("by_calls", "Most calls")):
        if key not in r:
            continue
        lines += ["", f"**{title}**", "", "| Calls | Total ms | Mean ms | Rows | Query |", "|---:|---:|---:|---:|---|"]
        lines += [f"| {q['calls']} | {q['total_ms']} | {q['mean_ms']} | {q['rows']} | `{q['query'].replace('|', '/')}` |"
                  for q in r[key]]
    if not v:
        lines += ["", "Run `CREATE EXTENSION IF NOT EXISTS pg_stat_statements;` in the Neon SQL editor."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "neon_query_stats.json"))
    args = ap.parse_args(argv)
    url = (os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        sys.exit("DATABASE_URL is not set.")
    import psycopg

    with psycopg.connect(url, connect_timeout=10) as conn:
        conn.read_only = True
        report = collect(conn, max(1, min(args.top, 50)))
    text = render(report)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
