#!/usr/bin/env python3
"""Run 61 — HSF Autonomous Research Mode v1 certification (deliberate release event).

    python -m scripts.autonomy_certification [--live] [--out artifacts/health]

Runs every certification gate (A–X, plus informational gates) from
`analytics.autonomy_certification`, and writes:

    artifacts/health/autonomy_certification.json
    artifacts/health/autonomy_certification.md

`--live` takes a fresh READ-ONLY production health snapshot and an observe-only
recovery plan (the ledger is read, never written) for Gate V and the production
snapshot. Without it, the committed artifacts/health snapshot is used. This
script never changes repository variables, secrets, the ledger or research data,
and never enables recovery.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from analytics import autonomy_certification as ac
from analytics import recovery_policy as rp
from analytics import system_health as sh

ROOT = Path(__file__).resolve().parents[1]


def commit_sha() -> Optional[str]:
    sha = (os.getenv("GITHUB_SHA") or "").strip()
    if sha:
        return sha
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = ROOT / ".git" / ref
            if p.exists():
                return p.read_text().strip()
            packed = (ROOT / ".git" / "packed-refs").read_text().splitlines()
            return next((ln.split()[0] for ln in packed if ln.endswith(ref)), None)
        return head
    except Exception:
        return None


def production_state(live: bool) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(health, plan, meta). Read-only in every mode."""
    meta: Dict[str, Any] = {"source": "live" if live else "committed artifacts/health"}
    if live:
        from scripts import system_health as shs
        now = _dt.datetime.now(_dt.timezone.utc)
        health = sh.evaluate(shs.collect(now))
        try:
            from db.recovery_ledger import list_recent
            ledger = list_recent(days=7)
            meta["ledger"] = {"available": True, "events_7d": len(ledger)}
        except Exception as e:
            ledger = []
            meta["ledger"] = {"available": False, "error": type(e).__name__}
        plan = rp.build_plan(health, ledger, os.environ, now=now)
    else:
        health = json.loads((ROOT / "artifacts" / "health" / "system_health.json").read_text())
        plan = json.loads((ROOT / "artifacts" / "health" / "recovery_plan.json").read_text())
        meta["ledger"] = {"available": plan.get("ledger_available"), "events_7d": None}
    return health, plan, meta


def render(cert: Dict[str, Any]) -> str:
    L = [f"# {cert['release']} — Certification", "",
         f"**Verdict: {cert['verdict']}**  ", f"Engineering certification: **{cert['engineering_certification']}**  ",
         f"Autonomy level: **{cert['autonomy_level']}**  ",
         f"Production autonomy mode: **{cert['production']['production_autonomy_mode']}**  ",
         f"Certified commit: `{cert['certified_commit']}` · generated {cert['generated_at']}", "",
         f"Mandatory gates: {cert['mandatory_passed']}/{cert['mandatory_gates']} passed, "
         f"{cert['mandatory_failed']} failed", "",
         "| Gate | Title | Status | Mandatory | Reference | Notes |", "|---|---|---|---|---|---|"]
    for k, g in cert["gates"].items():
        L.append(f"| {k} | {g['title']} | {g['status']} | {'yes' if g['mandatory'] else 'no (informational)'} | "
                 f"{g['reference']} | {g.get('notes', '')} |")
    p = cert["production"]
    L += ["", "## Current production snapshot (real, not simulated)", "",
          f"- System status: **{p['system_status']}** · health score {p['health_score']} · human action {p['human_action']}",
          f"- Autonomy readiness (health model): {p['autonomy_readiness']}",
          f"- Production autonomy mode: **{p['production_autonomy_mode']}** ({p['production_mode_reason']})",
          f"- Active incidents: {p['active_incidents'] or 'none'}",
          f"- Circuit breaker: {p['circuit_breaker']} · recovery ledger: {p['ledger']}",
          f"- Forward evidence: {p['forward_evidence']}", "",
          "Activation (human decision): repository variables `HSF_AUTONOMY_ENABLED=true` and "
          "`HSF_AUTONOMY_MODE=recover`.", "",
          "## Defects found and fixed during certification", ""]
    L += [f"- {d}" for d in cert["defects_fixed"]]
    L += ["", "_Operational certification only: no effectiveness statistics are computed or shown._", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="HSF Autonomous Research Mode v1 certification")
    ap.add_argument("--live", action="store_true", help="read-only live production snapshot for Gate V")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "health"))
    args = ap.parse_args()
    health, plan, meta = production_state(args.live)
    arts = sorted((ROOT / "artifacts" / "health").glob("*.json")) + sorted((ROOT / "artifacts" / "health").glob("*.md")) \
        + sorted((ROOT / "artifacts" / "research").glob("forward_evidence_readiness.*"))
    arts = [a for a in arts if not a.name.startswith("autonomy_certification")]
    res = ac.run_all(current_health=health, current_plan=plan, extra_artifacts=arts)
    cfg = rp.autonomy_config(os.environ)
    cert = {
        **res,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "certified_commit": commit_sha(),
        "production": {
            "source": meta["source"], "system_status": health.get("system_status"),
            "health_score": health.get("health_score"), "human_action": health.get("human_action"),
            "autonomy_readiness": (health.get("autonomy_readiness") or {}).get("state"),
            "production_autonomy_mode": cfg["production_state"], "production_mode_reason": cfg["reason"],
            "active_incidents": [i["incident_id"] for i in health.get("incidents") or []],
            "circuit_breaker": (plan.get("circuit_breaker") or {}).get("state"),
            "ledger": meta["ledger"], "forward_evidence": health.get("forward_evidence_status"),
        },
        "activation_required": {"HSF_AUTONOMY_ENABLED": "true", "HSF_AUTONOMY_MODE": "recover"},
        "defects_fixed": [
            "Sibling incidents mapped to the same recovery action (MATURATION_STALE / MATURATION_FAILING / "
            "CONSECUTIVE_FAILURES → RETRY_MATURATION) could re-launch it inside its cooldown and bypass "
            "per-incident attempt limits. Fixed: per-action cooldown (recovery_policy.action_cooldown), "
            "shared executions recorded against every sibling incident, and sibling bookkeeping rows "
            "flagged `shared_execution` (set before persisting) and excluded from circuit-breaker failure "
            "counts. Regression tests: tests/test_autonomous_recovery.py::SiblingIncidentRegressionTests.",
        ],
    }
    sh.assert_clean(cert)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autonomy_certification.json").write_text(json.dumps(cert, indent=2, default=str))
    md = render(cert)
    if sh.forbidden_text(md):
        raise ValueError("anti-peeking violation in certification markdown")
    (out / "autonomy_certification.md").write_text(md)
    for k, g in cert["gates"].items():
        print(f"{k:<7} {g['status']:<5} {g['title']}")
    print(f"mandatory {cert['mandatory_passed']}/{cert['mandatory_gates']} · production {cfg['production_state']}")
    print(cert["verdict"])
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"verdict={cert['verdict']}\n")
    return 0 if cert["verdict"] == ac.CERTIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())
