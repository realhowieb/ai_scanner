#!/usr/bin/env python3
"""Produce read-only DT quality-gate diagnostics from historical observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics.day_trade_tier_diagnostic import PROFILES, diagnose_row, summarize


def build_diagnostic(observations: list[dict], *, profile: str) -> dict:
    rows = []
    missing = 0
    mismatches = 0
    for observation in observations:
        inputs = observation.get("diagnostic_inputs")
        if not isinstance(inputs, dict):
            missing += 1
            continue
        candidate = profile == "rejected_v2"
        row = diagnose_row(inputs, profile=profile,
                           score=None if candidate else observation.get("score"),
                           quality=None if candidate else observation.get("setup_quality"))
        if observation.get("direction") != row["direction"]:
            mismatches += 1
        if row["score"] != row["score_components"].get("calculated_score"):
            mismatches += 1
        if row["direction"] in ("bullish", "bearish"):
            rows.append({"timestamp": observation.get("timestamp"),
                         "ticker": observation.get("ticker"), **row})
    report = summarize(rows, profile=profile)
    report["observations"] = rows
    report["observations_received"] = len(observations)
    report["missing_diagnostic_inputs"] = missing
    report["score_or_direction_mismatches"] = mismatches
    if missing or mismatches:
        report["status"] = "INCOMPLETE_OR_MISMATCHED_INPUTS"
    return report


def render_markdown(report: dict) -> str:
    lines = ["# DT Score Quality-Tier Diagnostic", "", "## Executive Summary", ""]
    if report["status"] != "OK":
        lines += [f"Status: **{report['status']}**. The original indicator inputs are required to identify the held-out blocker.",
                  "", "No gate counts or pileup cause can be inferred from aggregate validation metrics.", "",
                  "## Gate Funnel", "Unavailable without observations.", "",
                  "## Strong Rejection Reasons", "Unavailable without observations.", "",
                  "## Conflict Distribution", "Unavailable without observations.", "",
                  "## Most Common Conflicts", "Unavailable without observations.", "",
                  "## Agreement Analysis", "Unavailable without observations.", "",
                  "## Confirmation Analysis", "Unavailable without observations.", "",
                  "## Score Pileup Analysis", "Unavailable without observations.", "",
                  "## Hypothetical Gate Sensitivity", "Unavailable without observations.", "",
                  "## Recommendation", "Rerun the held-out window with original indicator inputs before proposing classifier changes.", ""]
        return "\n".join(lines)
    n = report["directional_n"]
    lines += [f"Profile: {report['profile']}; directional observations: {n}; quality tiers: {report['quality']}.",
              "", "## Gate Funnel", ""]
    for gate, count in report["gate_funnel"].items():
        lines.append(f"- {gate}: {count}/{n}")
    lines += ["", "## Strong Rejection Reasons", ""]
    for gate, item in sorted(report["rejection_reasons"].items(), key=lambda x: -x[1]["count"]):
        lines.append(f"- {gate}: {item['count']} ({item['pct']:.1%})")
    lines += ["", f"All except conflict pass: {report['rejection_combinations']['only_conflict_fails']}",
              "", "## Conflict Distribution", ""]
    for group, distribution in report["conflict_distribution"].items():
        lines.append(f"- {group}: " + ", ".join(f"{bucket}={item['count']}" for bucket, item in distribution.items()))
    lines += ["", "## Most Common Conflicts", ""]
    for name, item in report["conflict_frequency"].items():
        lines.append(f"- {name}: {item['count']} ({item['pct']:.1%}); score factors: {', '.join(item['also_score_factors'])}")
    lines += ["", "## Agreement Analysis", "", "```json", json.dumps(report["agreement"], indent=2), "```",
              "", "## Confirmation Analysis", "", "```json", json.dumps(report["confirmation"], indent=2), "```",
              "", "## Score Pileup Analysis", "", "```json", json.dumps(report["score_77_1"], indent=2), "```",
              "", "## Hypothetical Gate Sensitivity", "", "```json", json.dumps(report["gate_sensitivity"], indent=2), "```",
              "", "## Recommendation", ""]
    if report["gate_sensitivity"]["current"] == 0:
        candidates = [(name, count) for name, count in report["gate_sensitivity"].items() if name.startswith("without_")]
        leader, count = max(candidates, key=lambda pair: pair[1])
        lines.append(f"Largest single-gate blocker: {leader} ({count} hypothetical Strong). Review the associated indicator semantics before considering a new threshold.")
    else:
        lines.append(f"Strong occurs in {report['gate_sensitivity']['current']} observations; inspect the measured gate funnel before proposing changes.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, help="JSON list of observations with diagnostic_inputs")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="rejected_v2")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.observations:
        observations = json.loads(args.observations.read_text())
    else:
        from scripts.validate_day_trade_score import _load_intraday_observations
        observations = _load_intraday_observations() or []
    report = build_diagnostic(observations, profile=args.profile)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dt_tier_diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "dt_tier_diagnostic.md").write_text(render_markdown(report))
    print(f"DT tier diagnostic: {report['status']}; directional={report.get('directional_n', 0)}")
    return 0 if report["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
