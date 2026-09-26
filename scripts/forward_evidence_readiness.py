#!/usr/bin/env python3
"""Run 56 — forward-evidence readiness monitor runner (READ-ONLY, anti-peeking).

Loads production research observations and outcome records, runs the pure
`analytics.forward_readiness.monitor`, and writes:

    artifacts/research/forward_evidence_readiness.json
    artifacts/research/forward_evidence_readiness.md

It reports evidence QUANTITY and QUALITY only, never effectiveness. It writes
nothing to the store and does not trigger Run 55. When the state is
READY_FOR_RUN55_RERUN it prints (and writes to $GITHUB_OUTPUT)
RUN55_RERUN_RECOMMENDED=true, and takes no other action.

    python -m scripts.forward_evidence_readiness [--out DIR] [--input SNAPSHOT.json]
        [--maturation-report maturation_report.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from analytics.forward_readiness import COHORTS, HORIZONS, UNMATURED_REASONS, monitor

ROOT = Path(__file__).resolve().parents[1]
_ICON = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL", "NOT_APPLICABLE": "— N/A"}


def _v(x: Any, suffix: str = "") -> str:
    return "—" if x is None else f"{x}{suffix}"


def render_markdown(r: Dict[str, Any]) -> str:
    e = r["epoch"]
    L: List[str] = [
        "# Forward Evidence Readiness", "",
        f"Generated {r['generated_at']} · schema `{r['schema']}` · READ-ONLY · no effectiveness statistics", "",
        f"## State: **{r['state']}**", "",
        f"- Reason: {r['state_reason']}",
        f"- Limiting factor: **{r['limiting_factor']}**",
        f"- LONG readiness: **{r['long_readiness']}** · SHORT readiness: **{r['short_readiness']}**",
        f"- Estimated trading days until ready: **{r['estimated_trading_days_until_ready']}** ({r['estimate_basis']})",
        f"- RUN55_RERUN_RECOMMENDED = **{str(r['RUN55_RERUN_RECOMMENDED']).lower()}**", "",
        "## Forward epoch", "",
        f"- Start: `{e['forward_epoch_start_timestamp']}` · first forward scan run: `{e['forward_epoch_start_scan_run']}`",
        f"- Run 55 evaluation commit `{e['run55_evaluation_commit'][:10]}` · criteria commit "
        f"`{e['run55_criteria_commit'][:10]}` · workflow run {e['run55_workflow_run']}",
        f"- Scanner scoring version: {e['scanner_scoring_version']}",
        f"- Schemas: observation `{e['research_schema_version']}`, outcome `{e['outcome_schema_version']}`", "",
        "## Gates (pre-registered)", "", "| Gate | Status | Detail |", "|---|---|---|"]
    for k, g in r["gates"].items():
        L.append(f"| {k} | {_ICON[g['status']]} | {g['detail']} |")

    L += ["", "## Run 55 baseline vs forward epoch", "", "| Metric | Run 55 | Forward epoch | Gate |", "|---|---|---|---|"]
    for row in r["run55_comparison"]:
        L.append(f"| {row['metric']} | {_v(row['run55'])} | {_v(row['forward'])} | {row['gate']} |")

    t, s = r["time_coverage"], r["scan_coverage"]
    L += ["", "## Coverage", "",
          f"- Trading days: {t['forward_trading_days']} ({t['completed_forward_trading_days']} completed) · "
          f"calendar days {t['calendar_days']} · first {_v(t['first_forward_observation'])} · "
          f"latest {_v(t['latest_forward_observation'])}",
          f"- Scan runs: {s['total_forward_scan_runs']} total, {s['regular_session_scan_runs']} regular-session, "
          f"{s['successful_scan_runs']} successful ({s['successful_definition']}) · unique symbols {s['unique_symbols']}",
          "", "| Cohort | Observations | Symbols | Scan runs | Matured | Unmatured |", "|---|---|---|---|---|---|"]
    for c in COHORTS:
        x = r["cohorts"][c]
        L.append(f"| {c} | {x['observations']} | {x['unique_symbols']} | {x['unique_scan_runs']} | "
                 f"{x['matured_observations']} | {x['unmatured_observations']} |")

    L += ["", "## Horizon maturation (regular session, time-eligible observations)", "",
          "| Horizon | Cohort | Eligible | Matured | Maturation % | Directional % | MFE % | MAE % | Clusters | "
          + " | ".join(UNMATURED_REASONS) + " |",
          "|" + "---|" * (9 + len(UNMATURED_REASONS))]
    for h in HORIZONS:
        for c in COHORTS:
            x = r["horizons"][h][c]
            L.append(f"| {h} | {c} | {x['eligible_observations']} | {x['matured_observations']} | "
                     f"{_v(x['maturation_pct'])} | {_v(x['directional_return_coverage_pct'])} | "
                     f"{_v(x['mfe_coverage_pct'])} | {_v(x['mae_coverage_pct'])} | {x['matured_scan_run_clusters']} | "
                     + " | ".join(str(x["unmatured_reasons"][k]) for k in UNMATURED_REASONS) + " |")
    L += ["", f"_{r['data_quality']['unmatured_reason_note']}._", "",
          "## Maturation parity", "",
          "| Horizon | CANDIDATE % | NEAR_MISS % | CONTROL % | Parity gap (pp) | Class | Measurable |",
          "|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        p = r["maturation_parity"][h]
        L.append(f"| {h} | {_v(p['candidate_maturation_pct'])} | {_v(p['near_miss_maturation_pct'])} | "
                 f"{_v(p['control_maturation_pct'])} | {_v(p['maturation_parity_gap'])} | "
                 f"{p.get('parity_classification', '—')} | {'yes' if p['measurable'] else 'no'} |")

    L += ["", "## Directions", "", "| Direction | Observations | Matured (+60m) | Scan runs | Matured clusters (+60m) |",
          "|---|---|---|---|---|"]
    for k, x in r["directions"].items():
        L.append(f"| {k} | {x['observations']} | {x['matured_observations_60m']} | {x['scan_runs']} | "
                 f"{x['matured_scan_run_clusters_60m']} |")

    dq = r["data_quality"]
    L += ["", "## Data quality", ""]
    for k in ("pre_epoch_observations_excluded", "legacy_untagged_excluded", "duplicate_observation_ids",
              "conflicting_observation_ids", "duplicate_outcomes", "conflicting_outcomes",
              "orphan_forward_outcomes", "point_in_time_violations", "direction_transform_mismatches",
              "invalid_market_values", "cohort_overlap", "scoring_version_drift",
              "maturation_capacity_binding"):
        L.append(f"- {k}: {dq[k]}")
    L.append(f"- latest maturation run: {dq['maturation_run_report']}")
    L += ["", "## Research metadata completeness (Run 57, informational)", "",
          "| Cohort | Block % | Tier % | Regime % | Scoring version % | Commit SHA % | Provider % |",
          "|---|---|---|---|---|---|---|"]
    for c, m in (r.get("metadata_completeness") or {}).items():
        L.append(f"| {c} | {_v(m['metadata_block_coverage_pct'])} | {_v(m['tier_metadata_coverage_pct'])} | "
                 f"{_v(m['regime_metadata_coverage_pct'])} | {_v(m['scoring_version_coverage_pct'])} | "
                 f"{_v(m['commit_sha_coverage_pct'])} | {_v(m['provider_coverage_pct'])} |")
    L += ["", "## Progress", ""]
    L += [f"- {k}: {v}" for k, v in r["progress"].items()]
    L += ["", f"_{r['anti_peeking']}_", ""]
    return "\n".join(L)


def _load(input_path):
    if input_path:
        data = json.loads(Path(input_path).read_text())
        return data.get("observations") or [], data.get("outcomes_by_id") or {}
    from scripts.audit_research_cohorts import _load_live
    return _load_live()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 56 forward-evidence readiness (read-only)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "research"))
    ap.add_argument("--input", default=None, help="replay a saved snapshot instead of the DB")
    ap.add_argument("--maturation-report", default=None,
                    help="latest maturation_report.json (run-level RATE_LIMITED etc.); optional")
    args = ap.parse_args()

    observations, outcomes = _load(args.input)
    mat = None
    if args.maturation_report and Path(args.maturation_report).exists():
        try:
            mat = json.loads(Path(args.maturation_report).read_text())
        except Exception:
            mat = None
    report = monitor(observations, outcomes, maturation_report=mat)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forward_evidence_readiness.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "forward_evidence_readiness.md").write_text(render_markdown(report))

    flag = "true" if report["RUN55_RERUN_RECOMMENDED"] else "false"
    print("RUN 56 — FORWARD EVIDENCE READINESS (read-only)")
    print(f"state={report['state']} limiting={report['limiting_factor']} "
          f"long={report['long_readiness']} short={report['short_readiness']}")
    print(f"failing={[k for k, g in report['gates'].items() if g['status'] == 'FAIL']}")
    print(f"estimated_trading_days_until_ready={report['estimated_trading_days_until_ready']}")
    print(f"RUN55_RERUN_RECOMMENDED={flag}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"RUN55_RERUN_RECOMMENDED={flag}\nstate={report['state']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
