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
    "run52_fix_deployed": True,
    "maturation_backlog_status": "INSUFFICIENT_POST_FIX_EVIDENCE",
}


def _supported_comparisons(cohorts: Dict[str, Any], minimum: int = 30) -> list:
    """Modern-only cohort comparisons with complete outcomes at one horizon/direction."""
    candidate = cohorts.get("CANDIDATE", {}).get("modern_maturity_by_horizon_direction", {})
    control = cohorts.get("CONTROL", {}).get("modern_maturity_by_horizon_direction", {})
    supported = []
    for horizon in sorted(set(candidate) & set(control)):
        for direction in sorted(set(candidate[horizon]) & set(control[horizon])):
            cn = candidate[horizon][direction].get("analysis_eligible", 0)
            xn = control[horizon][direction].get("analysis_eligible", 0)
            if min(cn, xn) >= minimum:
                supported.append({"horizon": horizon, "direction": direction,
                                  "candidate_n": cn, "control_n": xn})
    return supported


def assess(audit: Optional[Dict[str, Any]], facts: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    facts = {**CODE_FACTS, **(facts or {})}
    gates: Dict[str, Dict[str, str]] = {}

    def gate(key, status, why):
        gates[key] = {"status": status, "detail": why}

    have_audit = bool(audit) and audit.get("total_observations", 0) > 0
    cohorts = (audit or {}).get("cohorts", {})
    overlap = (audit or {}).get("cohort_overlap_within_run", {})

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

    # 4. Required field contract satisfied for modern rows.
    if not have_audit:
        gate("required_field_contract", COND, "no live field-level audit yet")
    else:
        missing = []
        for cohort, values in cohorts.items():
            for row in values.get("modern_required_field_missingness", []):
                if row.get("missing_count", 0):
                    missing.append(f"{cohort}.{row['field']}={row['missing_count']}")
        gate("required_field_contract", PASS if not missing else FAIL,
             "modern minimum contract complete" if not missing else "; ".join(missing))

    # 5. Legacy data excludable deterministically.
    gate("legacy_excludable", PASS,
         "explicit research_cohort tag deterministically separates modern rows")

    # 6. Outcome calculations valid.
    gate("outcome_calculations", PASS if facts["outcome_formula_verified"] else FAIL,
         "directional_return/MFE/MAE deterministic tests pass")

    # 7. LONG direction verified in code and, when available, live examples.
    long_live = sum(
        1 for v in cohorts.values()
        for example in v.get("direction_examples", {}).get("LONG", [])
        if example.get("directional_return") is not None)
    gate("long_live_direction", PASS if facts["long_direction_verified"] and long_live else COND,
         f"deterministic verification={facts['long_direction_verified']}; live_examples={long_live}")

    # 8. SHORT verified or conclusions explicitly restricted.
    short_live = sum(
        1 for v in cohorts.values()
        for example in v.get("direction_examples", {}).get("SHORT", [])
        if example.get("directional_return") is not None)
    short_status = PASS if short_live else COND
    gate("short_live_direction", short_status,
         f"live_examples={short_live}; unsupported SHORT conclusions must be excluded")

    # 9. Maturation pipeline producing usable modern outcomes.
    backlog = facts["maturation_backlog_status"]
    modern_eligible = sum(
        cell.get("analysis_eligible", 0)
        for values in cohorts.values()
        for directions in values.get("modern_maturity_by_horizon_direction", {}).values()
        for cell in directions.values())
    maturation_status = (PASS if backlog == "BACKLOG_DRAINING" and modern_eligible
                         else FAIL if have_audit and modern_eligible == 0
                         else COND)
    gate("maturation_usable", maturation_status,
         f"usable modern horizon rows={modern_eligible}; backlog={backlog}; "
         f"run52_fix_deployed={facts['run52_fix_deployed']}")

    # 10. At least one legitimate modern Candidate-vs-Control comparison.
    supported = _supported_comparisons(cohorts)
    gate("adequate_scoped_sample", PASS if supported else FAIL if have_audit else COND,
         f"supported comparisons={supported}" if supported
         else "no horizon/direction has >=30 complete modern Candidate and Control outcomes")

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
