#!/usr/bin/env python3
"""Build the bounded Stair-stepper outcome-research artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics.stair_step_research import CONTEXT, build_research_report, render_markdown
from db.hsf_observations import load_recent_observations


def write_report(*, limit: int, out_dir: Path) -> dict:
    observations = load_recent_observations(
        limit=limit,
        context=CONTEXT,
        attach_outcomes="full",
    )
    report = build_research_report(observations)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stair_stepper_validation.json"
    markdown_path = out_dir / "stair_stepper_validation.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/automation/stair_stepper"),
    )
    args = parser.parse_args()
    report = write_report(limit=max(1, args.limit), out_dir=args.out)
    coverage = report["data_coverage"]
    verdict = report["best_window_verdict"]
    print("Day Trade Stair-Stepper Outcome Validation")
    print(f"Observations: {coverage['total_observations']}")
    print(f"Matured horizon pairs: {coverage['matured_horizon_pairs']}")
    print(f"Verdict: {verdict['status']}")
    print(f"Best window: {verdict['best_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
