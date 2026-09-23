#!/usr/bin/env python3
"""Run 53A — deterministic Run 54 readiness gate.

Evaluates the ten Run 54 preconditions from (a) the read-only cohort audit
(artifacts/automation/cohort_audit.json, produced by scripts.audit_research_cohorts
against production Neon) and (b) code-verified facts established in this run. Emits
a machine-readable assessment and an overall RUN54_READY = YES | CONDITIONAL | NO.

Pure/deterministic: given the same audit JSON + flags it always returns the same
verdict. It changes no scoring, ranking, or data. A gate whose evidence is not yet
available (no live audit run) is CONDITIONAL, never silently PASS.

    python -m scripts.run54_readiness [--audit PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

PASS, COND, FAIL = "PASS", "CONDITIONAL", "FAIL"

# Code-verified-in-this-run facts (deterministic, not dependent on the live DB):
#   long_direction_verified   — deterministic LONG winner/loser tests pass.
#   short_direction_verified  — deterministic SHORT winner/loser tests pass.
#   outcome_formula_verified  — directional_return/MFE/MAE proven correct after the
#                               long/short synonym fix (Run 53A).
CODE_FACTS = {
    "long_direction_verified": True,
    "short_direction_verified_deterministic": True,
    "short_live_evidence": "INSUFFICIENT",   # no live SHORT samples confirmed
    "outcome_formula_verified": True,
    "run52_fix_deployed": False,             # dev only; main lacks 3e5320e
}


def assess(audit: Optional[Dict[str, Any]], facts: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    facts = {**CODE_FACTS, **(facts or {})}
    gates: Dict[str, Dict[str, str]] = {}

    def gate(key, status, why):
        gates[key] = {"status": status, "detail": why}

    have_audit = bool(audit) and audit.get("total_observations", 0) > 0
    cohorts = (audit or {}).get("cohorts", {})
    overlap = (audit or {}).get("cohort_overlap_within_run", {})
    conflicting = (audit or {}).get("conflicting_duplicates")

    # 1. Modern cohort tagging trustworthy.
    if not have_audit:
        gate("modern_cohort_tagging", COND, "no live audit yet — run audit_research_cohorts against Neon")
    else:
        explicit = sum(c.get("explicitly_tagged", 0) for c in cohorts.values())
        legacy = sum(c.get("legacy_inferred", 0) for c in cohorts.values())
        gate("modern_cohort_tagging", PASS if explicit > 0 else FAIL,
             f"explicit={explicit} legacy_inferred={legacy} (legacy separable via explicit tag)")

    # 2. Candidate/Near-Miss/Control separation valid (no candidate∩control overlap).
    if not have_audit:
        gate("cohort_separation", COND, "no live audit yet")
    else:
        cc = overlap.get("candidate_control", 0)
        gate("cohort_separation", PASS if cc == 0 else FAIL,
             f"candidate∩control overlaps within run = {cc}")

    # 3. No material PIT leakage.
    if not have_audit:
        gate("pit_integrity", COND, "no live audit yet")
    else:
        pit = sum(c.get("point_in_time_violations", 0) for c in cohorts.values())
        gate("pit_integrity", PASS if pit == 0 else FAIL, f"point_in_time_violations = {pit}")

    # 4. Outcome calculations correct.
    gate("outcome_formula", PASS if facts["outcome_formula_verified"] else FAIL,
         "directional_return/MFE/MAE verified after long/short synonym fix (Run 53A)")

    # 5. LONG direction verified.
    gate("long_direction", PASS if facts["long_direction_verified"] else FAIL,
         "deterministic LONG winner/loser tests pass")

    # 6. SHORT direction verified OR explicitly insufficient (excluded from claims).
    if facts["short_direction_verified_deterministic"]:
        gate("short_direction", COND,
             f"deterministic SHORT tests pass; live SHORT evidence = {facts['short_live_evidence']} "
             "(exclude SHORT from unsupported live conclusions)")
    else:
        gate("short_direction", FAIL, "SHORT direction not verified")

    # 7. Maturation producing usable outcomes. Pre-fix outcomes have NULL
    #    directional_return/MFE/MAE (the bug); usable outcomes require the Run 53A
    #    direction fix AND the Run 52 anti-join deployed, then fresh maturation.
    gate("maturation_usable", COND,
         "raw_return present, but pre-Run-53A outcomes have NULL directional/MFE/MAE; "
         f"run52_fix_deployed={facts['run52_fix_deployed']} — deploy fixes then re-mature")

    # 8. Legacy data excludable deterministically.
    gate("legacy_excludable", PASS,
         "explicit research_cohort tag + auditor explicitly_tagged/legacy_inferred split")

    # 9. No severe conflicting-duplicate problem.
    if not have_audit or conflicting is None:
        gate("no_conflicting_duplicates", COND, "no live audit yet")
    else:
        gate("no_conflicting_duplicates", PASS if conflicting == 0 else FAIL,
             f"conflicting_duplicates = {conflicting}")

    # 10. Sample sizes reported honestly.
    gate("honest_sample_sizes", PASS,
         "auditor reports per-cohort n, matured/unmatured, and all counts")

    statuses = [g["status"] for g in gates.values()]
    if FAIL in statuses:
        overall = "NO"
    elif COND in statuses:
        overall = "CONDITIONAL"
    else:
        overall = "YES"
    return {"schema": "run54-readiness-1.0", "run54_ready": overall, "gates": gates,
            "have_live_audit": have_audit}


def render_text(report: Dict[str, Any]) -> str:
    lines = [f"RUN54_READY = {report['run54_ready']}",
             f"(live audit present: {report['have_live_audit']})", ""]
    for k, g in report["gates"].items():
        lines.append(f"[{g['status']:<11}] {k}: {g['detail']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic Run 54 readiness gate")
    ap.add_argument("--audit", default=str(ROOT / "artifacts" / "automation" / "cohort_audit.json"))
    args = ap.parse_args()
    audit = None
    try:
        p = Path(args.audit)
        if p.exists():
            audit = json.loads(p.read_text())
    except Exception:
        audit = None
    report = assess(audit)
    print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
