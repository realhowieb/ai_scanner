#!/usr/bin/env python3
"""Run 38 — generate the scanner-performance scoreboard from real outcome data.

Reads matured rows from the populated `signal_outcomes` table (and, when present,
Run 36 canonical observations-with-outcomes), runs the pure
`analytics.scanner_performance` engine, and writes a machine-readable scoreboard
artifact. Read-only, non-fatal: with no DB / no matured data it reports
INSUFFICIENT_DATA honestly and never fabricates results.

    python -m scripts.scanner_scoreboard --out artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def _fetch_signal_outcome_rows(days_back: int = 365, limit: int = 50000) -> List[Dict[str, Any]]:
    """Matured signal_outcomes rows (guarded, Neon). [] when unavailable."""
    try:
        from db.engine import get_neon_conn
    except Exception:
        return []
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source, signal_type, ticker, fired_at,
                   return_1d, return_3d, return_5d, mfe_5d, mae_5d
            FROM signal_outcomes
            WHERE outcome_computed_at IS NOT NULL
              AND fired_at >= NOW() - make_interval(days => %s)
            ORDER BY fired_at DESC
            LIMIT %s
            """,
            (int(days_back), int(limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    cols = ("source", "signal_type", "ticker", "fired_at", "return_1d",
            "return_3d", "return_5d", "mfe_5d", "mae_5d")
    out = []
    for r in rows:
        out.append(dict(r) if isinstance(r, dict) else dict(zip(cols, r)))
    return out


def _fetch_canonical_records() -> List[Dict[str, Any]]:
    """Run 36 canonical observations-with-outcomes (empty until populated)."""
    try:
        from db.hsf_observations import load_recent_observations
        return load_recent_observations(limit=20000) or []
    except Exception:
        return []


def main() -> int:
    from analytics.scanner_performance import (
        build_scoreboard_report,
        from_canonical_observations,
        from_signal_outcomes_rows,
    )

    ap = argparse.ArgumentParser(description="Scanner performance scoreboard")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--days-back", type=int, default=365)
    ap.add_argument("--primary-horizon", default="5d")
    args = ap.parse_args()

    records = from_signal_outcomes_rows(_fetch_signal_outcome_rows(days_back=args.days_back))
    records += from_canonical_observations(_fetch_canonical_records())

    if not records:
        report = {
            "schema": "hsf-scanner-scoreboard-1.0",
            "status": "INSUFFICIENT_DATA",
            "reason": ("No matured outcome data available (signal_outcomes empty / "
                       "no DB, and Run 36 canonical store not yet populated). No "
                       "scoreboard is fabricated."),
            "n_records": 0,
        }
    else:
        report = {"status": "OK",
                  **build_scoreboard_report(records, primary_horizon=args.primary_horizon)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scanner_scoreboard.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[scanner_scoreboard] status={report.get('status')} "
          f"records={report.get('n_records', 0)} -> {out_dir}")
    if report.get("status") == "OK":
        for name, s in sorted(report["overall"]["scanners"].items()):
            print(f"  {name:32} n={s['signals']:5} class={s['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
