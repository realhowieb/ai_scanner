#!/usr/bin/env python3
"""Run 48 — signal effectiveness analysis runner (read-only, no production change).

Loads canonical research observations (with matured outcomes), runs the pure
`analytics.signal_effectiveness` framework, and writes machine-readable artifacts
+ a Run 49 recommendation manifest. Honestly reports the effectiveness verdict —
INSUFFICIENT LIVE DATA when cohorts/outcomes have not accumulated enough.

    python -m scripts.run48_effectiveness [--limit N] [--out DIR] [--include-degraded]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics.signal_effectiveness import build_analysis_report

ROOT = Path(__file__).resolve().parents[1]


def _load(limit: int):
    try:
        from db.hsf_observations import load_recent_observations
        return load_recent_observations(limit=limit) or []
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 48 signal effectiveness")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    ap.add_argument("--include-degraded", action="store_true",
                    help="also analyze DEGRADED scans (diagnostic; not primary)")
    args = ap.parse_args()

    observations = _load(args.limit)
    report = build_analysis_report(observations, healthy_only=not args.include_degraded)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run48_analysis.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "run48_recommendations.json").write_text(
        json.dumps(report["recommendations"], indent=2, default=str))

    d = report["dataset"]
    print("RUN 48 — SIGNAL EFFECTIVENESS")
    print(f"records={d['total_records']} by_cohort={d['by_cohort']}")
    print(f"matured_by_horizon={d['matured_by_horizon']}")
    print(f"usable_paired_n={d['usable_paired_n']} evidence={d['evidence_level']}")
    print(f"EFFECTIVENESS VERDICT: {report['effectiveness_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
