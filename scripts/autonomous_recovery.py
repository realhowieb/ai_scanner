#!/usr/bin/env python3
"""Run 60 — safe self-healing controller (operational recovery only).

    python -m scripts.autonomous_recovery                 # PLAN ONLY (default)
    python -m scripts.autonomous_recovery --execute       # execute allowlisted recovery IF the kill
                                                          # switch permits (HSF_AUTONOMY_ENABLED=true
                                                          # and HSF_AUTONOMY_MODE=recover); else plan
    python -m scripts.autonomous_recovery --verify        # regenerate health and report which
                                                          # incidents in the last plan cleared
    python -m scripts.autonomous_recovery --reset-circuit --by <name>   # human-only circuit reset

Outputs (safe to publish; no secrets): artifacts/health/recovery_plan.{json,md},
recovery_results.json, human_escalation.{json,md}.

Security: actions come only from `analytics.recovery_policy.ALLOWLIST`; each maps to
a constant workflow file + constant inputs. Incident text never reaches a command;
no process spawning or shell execution is used. The controller never dispatches itself.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from analytics import recovery_controller as rc
from analytics import recovery_policy as rp
from analytics import system_health as sh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "health"
_WF_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*\.yml$")
SELF_WORKFLOW = "autonomous-recovery.yml"


def _scrub(text: Any) -> str:
    s = str(text or "")
    for k in ("GH_TOKEN", "GITHUB_TOKEN", "DATABASE_URL", "ALPACA_API_SECRET_KEY", "ALPACA_API_KEY_ID"):
        v = os.getenv(k)
        if v:
            s = s.replace(v, "***")
    return s[:300]


def fresh_health(persist: bool = True) -> Dict[str, Any]:
    from scripts import system_health as shs
    now = _dt.datetime.now(_dt.timezone.utc)
    inputs = shs.collect(now)
    report = sh.evaluate(inputs)
    if persist:
        try:
            from db.system_health import save_snapshot
            save_snapshot(report)
        except Exception:
            pass
    return report


def _validated_workflow(action: str, spec: Mapping[str, Any]) -> str:
    wf = spec.get("workflow")
    if action not in rp.ALLOWLIST or wf not in rp.KNOWN_WORKFLOWS or not _WF_RE.match(str(wf)) \
            or wf == SELF_WORKFLOW:
        raise ValueError("workflow not allowlisted")
    return wf


def execute_action(action: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Fixed mapping: inline health regeneration or a constant workflow dispatch."""
    if rp.ALLOWLIST.get(action) != spec:
        return {"ok": False, "error": "action/spec mismatch"}
    if spec["executor"] == "inline_health":
        fresh_health(persist=True)
        return {"ok": True}
    wf = _validated_workflow(action, spec)
    from scripts.system_health import GH_API, _gh
    session, repo = _gh()
    active = session.get(f"{GH_API}/repos/{repo}/actions/workflows/{wf}/runs",
                         params={"status": "in_progress", "per_page": 5}, timeout=15).json()
    if active.get("workflow_runs"):
        return {"ok": False, "skipped": True, "error": f"{wf} already running; not dispatching a duplicate"}
    t0 = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=5)
    inputs = {k: str(v) for k, v in spec["inputs"].items()}
    r = session.post(f"{GH_API}/repos/{repo}/actions/workflows/{wf}/dispatches",
                     json={"ref": "main", "inputs": inputs}, timeout=15)
    if r.status_code != 204:
        return {"ok": False, "error": _scrub(f"dispatch HTTP {r.status_code}: {r.text[:120]}")}
    deadline = time.monotonic() + 60 * max(1, int(spec.get("wait_min") or 10))
    run = None
    while time.monotonic() < deadline:
        time.sleep(20)
        rs = session.get(f"{GH_API}/repos/{repo}/actions/workflows/{wf}/runs",
                         params={"event": "workflow_dispatch", "per_page": 5}, timeout=15).json()
        cands = [x for x in rs.get("workflow_runs") or [] if x.get("created_at", "") >= t0.isoformat()[:19]]
        if cands:
            run = cands[0]
            if run.get("status") == "completed":
                return {"ok": run.get("conclusion") == "success", "run_id": run.get("id"),
                        "error": None if run.get("conclusion") == "success" else f"run concluded {run.get('conclusion')}"}
    return {"ok": False, "run_id": (run or {}).get("id"), "error": f"{wf} did not complete within wait window"}


def _write(name: str, obj: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(obj if isinstance(obj, str) else rc.dumps(obj))


def _ledger():
    from db.recovery_ledger import append_event, list_recent
    return list_recent(days=7), append_event


def main() -> int:
    ap = argparse.ArgumentParser(description="HSF safe self-healing (operational recovery only)")
    ap.add_argument("--execute", action="store_true", help="execute allowlisted recovery if the kill switch permits")
    ap.add_argument("--verify", action="store_true", help="regenerate health and report cleared incidents")
    ap.add_argument("--health", default=None, help="use a saved health report instead of collecting")
    ap.add_argument("--reset-circuit", action="store_true")
    ap.add_argument("--by", default=None, help="human performing the circuit reset")
    args = ap.parse_args()
    now = _dt.datetime.now(_dt.timezone.utc)

    if args.reset_circuit:
        if not args.by or not re.fullmatch(r"[A-Za-z0-9 ._@-]{2,64}", args.by):
            raise SystemExit("--reset-circuit requires --by <name>")
        from db.recovery_ledger import append_event
        append_event({"recovery_id": rc._rid("reset", now.isoformat()), "incident_id": "system:AUTONOMY_CIRCUIT_OPEN",
                      "incident_type": "AUTONOMY_CIRCUIT_OPEN", "action": None, "policy": "HUMAN",
                      "result": "CIRCUIT_RESET", "started_at": now.isoformat(), "completed_at": now.isoformat(),
                      "reason": f"reset by {args.by}"})
        print("circuit reset recorded")
        return 0

    health = json.loads(Path(args.health).read_text()) if args.health else fresh_health(persist=False)
    env = dict(os.environ) if args.execute else {**os.environ, "HSF_AUTONOMY_MODE": "observe"}
    try:
        ledger, append = _ledger()
        ledger_ok = True
    except Exception as e:
        ledger, append, ledger_ok = [], None, False
        print(f"[recovery] ledger unavailable ({_scrub(e)}); forcing plan-only")
        env["HSF_AUTONOMY_MODE"] = "observe"
    results: List[Dict[str, Any]] = []

    def append_fn(ev):
        results.append(dict(ev))
        if append is not None:
            try:
                append(ev)
            except Exception as e:
                print(f"[recovery] ledger append failed: {_scrub(e)}")

    cycle = rc.run_cycle(health, ledger, env, execute_fn=execute_action,
                         verify_fn=lambda: fresh_health(persist=True), append_fn=append_fn, now=now)
    plan = cycle["plan"]
    plan["ledger_available"] = ledger_ok
    _write("recovery_plan.json", plan)
    _write("recovery_plan.md", rc.render_plan_md(plan))
    _write("recovery_results.json", {"generated_at": now.isoformat(), "events": results})
    _write("human_escalation.json", cycle["escalation"])
    _write("human_escalation.md", rc.render_escalation_md(cycle["escalation"]))
    if args.verify:
        after = fresh_health(persist=True)
        still = sorted({i["incident_id"] for i in after.get("incidents") or []})
        _write("recovery_verification.json", {"generated_at": after["generated_at"],
                                              "system_status": after["system_status"],
                                              "remaining_incidents": still})
    sh.assert_clean({"plan": plan, "escalation": cycle["escalation"]})
    a = plan["autonomy"]
    print(f"AUTONOMY: {a['production_state']} ({a['reason']}) · circuit {plan['circuit_breaker']['state']}")
    for d in plan["decisions"]:
        print(f"  {d['policy']:<12} {d['incident_id']} → {d.get('action') or '—'} ({d['reason']})")
    print(f"  would execute: {plan['would_execute'] or 'nothing'} · executed: {[e['action'] for e in cycle['executed']]}")
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"human_action_required={str(cycle['escalation']['human_action_required']).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
