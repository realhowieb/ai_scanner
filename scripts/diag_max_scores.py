"""Diagnostic: find where out-of-range BreakoutScores in stored snapshots come from.

Scans recent runs (snapshots and labeled runs), reports each run's max
BreakoutScore, and dumps the offending rows' scoring inputs for anything above
a sanity threshold — to locate which scan path is bypassing input clipping.
Read-only. Run headless via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLD = 100.0
INPUT_COLS = [
    "Ticker", "BreakoutScore", "PctChange", "GapPct", "Trend10D%", "Trend20D%",
    "VolRel20", "RelVol", "RSvsSPY", "BreakoutPos20D", "Last",
]


def main() -> None:
    from db.runs import list_runs, load_run_results
    from ui.app_runtime import normalize_results_to_df

    runs = list_runs(limit=400) or []
    print(f"inspecting {len(runs)} recent runs")
    flagged = 0
    for r in runs:
        try:
            raw = load_run_results(r["id"])
            df = normalize_results_to_df(raw) if raw else None
        except Exception as e:
            print(f"run {r.get('id')}: load failed {type(e).__name__}")
            continue
        if df is None or len(df) == 0 or "BreakoutScore" not in df.columns:
            continue
        try:
            mx = float(df["BreakoutScore"].max())
        except Exception:
            continue
        tag = "SNAPSHOT" if r.get("is_snapshot") else (r.get("label") or "run")
        line = (
            f"run={r.get('id')} {str(r.get('created_at'))[:16]} [{tag}] "
            f"name={str(r.get('name'))[:40]!r} max_score={mx:.1f}"
        )
        if mx > THRESHOLD:
            flagged += 1
            print("⚠️ " + line)
            cols = [c for c in INPUT_COLS if c in df.columns]
            bad = df[df["BreakoutScore"] > THRESHOLD][cols].head(5)
            print(bad.to_string(index=False))
        else:
            print("   " + line)
    print(f"\nflagged {flagged} run(s) with max BreakoutScore > {THRESHOLD:g}")


if __name__ == "__main__":
    main()
