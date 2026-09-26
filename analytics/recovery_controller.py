"""Run 60 — recovery controller: DETECT → CLASSIFY → DECIDE → RECOVER → VERIFY → ESCALATE.

Pure orchestration with injected I/O so every path is testable:
    execute_fn(action_id, spec) -> {"ok": bool, "error": str|None, "run_id": ...}
    verify_fn() -> fresh Run 59 health report
    append_fn(event) -> None          (append-only recovery ledger)

Success is incident-specific: a command exiting 0 is never enough. After every
action the health model is regenerated and the target incident must be CLEARED
(or measurably IMPROVED by a deterministic criterion); otherwise the attempt is
VERIFICATION_FAILED. Attempt limits, cooldowns and the circuit breaker come from
`analytics.recovery_policy`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from analytics import recovery_policy as rp

MATURATION_ACTIONS = ("RETRY_MATURATION", "RETRY_TRANSIENT_PROVIDER_OPERATION")
UNIVERSE_ACTIONS = ("RETRY_UNIVERSE_PROBE", "RETRY_UNIVERSE_REFRESH")
ARTIFACT_OF_ACTION = {
    "REGENERATE_FORWARD_READINESS": "forward_evidence_readiness",
    "RERUN_PARITY_AUDIT": "maturation_parity_audit",
    "REGENERATE_SYSTEM_HEALTH": "previous_system_health",
}


def _criticals(health: Mapping[str, Any]) -> set:
    return {i["incident_id"] for i in health.get("incidents") or [] if i.get("severity") == "CRITICAL"}


def _incident_ids(health: Mapping[str, Any]) -> set:
    return {i["incident_id"] for i in health.get("incidents") or []}


def verify(decision: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic, incident-specific verification (no false success)."""
    target = decision["incident_id"]
    action = decision["action"]
    subs = after.get("subsystems") or {}
    cleared = target not in _incident_ids(after)
    checks: Dict[str, Any] = {"target_incident_absent": cleared}
    if action in UNIVERSE_ACTIONS:
        u = subs.get("universe") or {}
        count = (u.get("metrics") or {}).get("symbol_count") or 0
        checks["universe_non_empty"] = count > 0
        checks["universe_not_suspicious"] = u.get("detail_state") not in ("BROKEN", "SUSPICIOUS", "STALE")
        ok = cleared and checks["universe_non_empty"] and checks["universe_not_suspicious"]
        return {"result": "CLEARED" if ok else "NOT_CLEARED", "checks": checks}
    if action in MATURATION_ACTIONS:
        m_after = (subs.get("maturation") or {}).get("metrics") or {}
        m_before = ((before.get("subsystems") or {}).get("maturation") or {}).get("metrics") or {}
        codes_after = {f.get("code") for f in (subs.get("maturation") or {}).get("findings") or []}
        newer = str(m_after.get("last_success") or "") > str(m_before.get("last_success") or "")
        checks["new_successful_run"] = newer
        checks["stale_or_failing_absent"] = not (codes_after & {"MATURATION_STALE", "MATURATION_FAILING"})
        d0, d1 = m_before.get("deferred_symbols") or 0, m_after.get("deferred_symbols") or 0
        checks["deferred_before_after"] = [d0, d1]
        if cleared and newer and checks["stale_or_failing_absent"]:
            return {"result": "CLEARED", "checks": checks}
        if not cleared and newer and decision.get("incident_type") == "BACKLOG_GROWING" and d0 and d1 <= 0.75 * d0:
            return {"result": "IMPROVED", "checks": checks}
        return {"result": "NOT_CLEARED", "checks": checks}
    if action in ARTIFACT_OF_ACTION:
        arts = {a["artifact"]: a for a in ((subs.get("artifact_freshness") or {}).get("metrics") or {}).get("artifacts") or []}
        fresh = (arts.get(ARTIFACT_OF_ACTION[action]) or {}).get("status") in ("FRESH", "NOT_APPLICABLE")
        checks["artifact_fresh"] = fresh
        return {"result": "CLEARED" if cleared and fresh else "NOT_CLEARED", "checks": checks}
    return {"result": "CLEARED" if cleared else "NOT_CLEARED", "checks": checks}


def _rid(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _event(decision: Mapping[str, Any], *, result: str, now: _dt.datetime, started: _dt.datetime,
           completed: Optional[_dt.datetime] = None, attempt: Optional[int] = None,
           verification: Optional[Mapping[str, Any]] = None, error: Optional[str] = None,
           health_before: Optional[Mapping[str, Any]] = None, health_after: Optional[Mapping[str, Any]] = None,
           reason: Optional[str] = None, new_critical: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "recovery_id": _rid(decision.get("incident_id"), decision.get("action"), started.isoformat(), result),
        "incident_id": decision.get("incident_id"), "incident_type": decision.get("incident_type"),
        "action": decision.get("action"), "policy": decision.get("policy"),
        "risk_class": decision.get("risk_class"), "detected_at": now.isoformat(),
        "started_at": started.isoformat(), "completed_at": (completed or started).isoformat(),
        "attempt": attempt, "result": result,
        "verification_result": (verification or {}).get("result"),
        "verification_checks": (verification or {}).get("checks"),
        "reason": reason or decision.get("reason"), "error": (error or None) and str(error)[:300],
        "health_before": {"system_status": (health_before or {}).get("system_status"),
                          "generated_at": (health_before or {}).get("generated_at")} if health_before else None,
        "health_after": {"system_status": (health_after or {}).get("system_status"),
                         "generated_at": (health_after or {}).get("generated_at")} if health_after else None,
        "new_critical_incidents": new_critical or [],
    }


def run_cycle(health: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]], env: Mapping[str, str], *,
              execute_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
              verify_fn: Callable[[], Mapping[str, Any]],
              append_fn: Callable[[Mapping[str, Any]], Any],
              now: Optional[_dt.datetime] = None,
              clock: Optional[Callable[[], _dt.datetime]] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))
    ledger = list(ledger)
    plan = rp.build_plan(health, ledger, env, now=now)
    written: List[Dict[str, Any]] = []

    def append(ev):
        append_fn(ev)
        ledger.append(ev)
        written.append(ev)

    circuit = plan["circuit_breaker"]
    already_open = any(e.get("result") == "CIRCUIT_OPENED" for e in ledger) and \
        any("not reset" in r for r in circuit["reasons"])
    if circuit["open"] and not already_open:
        append(_event({"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "incident_type": "AUTONOMY_CIRCUIT_OPEN",
                       "policy": "ESCALATE", "risk_class": None, "action": None},
                      result="CIRCUIT_OPENED", now=now, started=now, reason="; ".join(circuit["reasons"])))

    by_id = {d["incident_id"]: d for d in plan["decisions"]}
    executed: List[Dict[str, Any]] = []
    stop = False
    for iid in plan["would_execute"]:
        d = by_id[iid]
        spec = rp.ALLOWLIST[d["action"]]
        attempt = (d.get("attempts") or {}).get("attempt_count", 0) + 1
        if not plan["autonomy"]["execution_permitted"] or circuit["open"] or stop:
            why = ("observe mode: would execute" if not plan["autonomy"]["execution_permitted"] else
                   "circuit open" if circuit["open"] else "stopped: previous recovery created new critical incidents")
            append(_event(d, result="SKIPPED", now=now, started=now, reason=why))
            continue
        started = clock()
        try:
            res = dict(execute_fn(d["action"], spec) or {})
        except Exception as e:  # executor failures are recorded, never raised
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if res.get("skipped"):   # e.g. an equivalent run is already active: not an attempt
            append(_event(d, result="SKIPPED", now=now, started=started, completed=clock(),
                          reason=res.get("error") or "skipped"))
            continue
        after = verify_fn() if res.get("ok") else None
        ver = verify(d, health, after) if after else {"result": "NOT_RUN", "checks": {}}
        new_crit = sorted(_criticals(after) - _criticals(health)) if after else []
        result = ("FAILED" if not res.get("ok") else
                  "SUCCESS" if ver["result"] in ("CLEARED", "IMPROVED") else "VERIFICATION_FAILED")
        ev = _event(d, result=result, now=now, started=started, completed=clock(), attempt=attempt,
                    verification=ver, error=res.get("error"), health_before=health, health_after=after,
                    new_critical=new_crit)
        append(ev)
        executed.append(ev)
        if new_crit:
            stop = True   # recovery created new critical incidents → breaker opens next cycle
        if result != "SUCCESS" and attempt >= spec["max_attempts"]:
            append(_event(d, result="ESCALATED", now=now, started=clock(), attempt=attempt,
                          reason=f"attempt limit {attempt}/{spec['max_attempts']} reached without verified recovery"))

    for d in plan["decisions"]:
        hint = d.get("result_hint")
        if hint in ("COOLDOWN", "ATTEMPT_LIMIT"):
            append(_event(d, result=hint, now=now, started=now))
        elif d["policy"] == "ESCALATE" and not any(
                e.get("incident_id") == d["incident_id"] and e.get("result") in ("ESCALATED", "PROHIBITED")
                and str(e.get("started_at", "")) >= (now - _dt.timedelta(hours=24)).isoformat() for e in ledger):
            append(_event(d, result="PROHIBITED" if d.get("risk_class") == "PROHIBITED" and d.get("action")
                          else "ESCALATED", now=now, started=now))
    return {"plan": plan, "executed": executed, "ledger_events_written": written,
            "escalation": escalation(plan, ledger, health)}


def escalation(plan: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]], health: Mapping[str, Any]) -> Dict[str, Any]:
    incidents = {i["incident_id"]: i for i in health.get("incidents") or []}
    items = []
    for d in plan["decisions"]:
        if d["policy"] != "ESCALATE":
            continue
        tried = [e for e in ledger if e.get("incident_id") == d["incident_id"]
                 and e.get("result") in rp.ATTEMPT_RESULTS]
        spec = rp.ALLOWLIST.get(d.get("action") or "", {})
        inc = incidents.get(d["incident_id"], {})
        items.append({
            "incident_id": d["incident_id"], "incident_type": d["incident_type"], "subsystem": d["subsystem"],
            "severity": d.get("severity"), "detected": inc.get("detected_at"),
            "automatic_attempts": f"{len(tried)}/{spec.get('max_attempts', 0)}" if d.get("action") else "0/0",
            "recovery": ("FAILED" if tried else "NOT ATTEMPTED"), "why_escalated": d["reason"],
            "what_hsf_tried": [f"{e.get('action')} → {e.get('result')} (verification {e.get('verification_result')})"
                               for e in tried],
            "recommended_human_action": inc.get("recommended_action") or "inspect the incident",
        })
    if plan["circuit_breaker"]["open"]:
        items.insert(0, {"incident_id": "system:AUTONOMY_CIRCUIT_OPEN", "incident_type": "AUTONOMY_CIRCUIT_OPEN",
                         "subsystem": "autonomy", "severity": "CRITICAL", "automatic_attempts": "stopped",
                         "recovery": "HALTED", "why_escalated": "; ".join(plan["circuit_breaker"]["reasons"]),
                         "what_hsf_tried": [], "recommended_human_action":
                         "review the recovery ledger, fix the cause, then reset the circuit "
                         "(python -m scripts.autonomous_recovery --reset-circuit --by <name>)"})
    return {"schema": "hsf-human-escalation-1.0", "generated_at": plan["generated_at"],
            "human_action_required": bool(items), "system_status": health.get("system_status"),
            "items": items}


def render_escalation_md(esc: Mapping[str, Any]) -> str:
    if not esc["items"]:
        return "# HSF HUMAN ACTION\n\nNone required. Automatic recovery has nothing to escalate.\n"
    L = ["# HSF HUMAN ACTION REQUIRED", "", f"System state: **{esc['system_status']}** · generated {esc['generated_at']}", ""]
    for i in esc["items"]:
        L += [f"## {i['incident_type']}", "", f"- Incident: `{i['incident_id']}` ({i.get('severity')})",
              f"- Detected: {i.get('detected')}", f"- Automatic attempts: {i['automatic_attempts']}",
              f"- Recovery: **{i['recovery']}**", f"- Why escalated: {i['why_escalated']}"]
        if i["what_hsf_tried"]:
            L.append("- What HSF already tried:")
            L += [f"  {k}. {t}" for k, t in enumerate(i["what_hsf_tried"], 1)]
        L += [f"- **Recommended human action:** {i['recommended_human_action']}", ""]
    return "\n".join(L)


def render_plan_md(plan: Mapping[str, Any]) -> str:
    a, c = plan["autonomy"], plan["circuit_breaker"]
    L = ["# HSF Recovery Plan", "",
         f"Generated {plan['generated_at']} · health {plan['health_generated_at']} ({plan['system_status']})", "",
         f"- Production autonomy: **{a['production_state']}** ({a['reason']})",
         f"- Circuit breaker: **{c['state']}**" + (f" — {'; '.join(c['reasons'])}" if c["reasons"] else ""),
         f"- Would execute: {plan['would_execute'] or 'nothing'} · executing: {plan['to_execute'] or 'nothing'}",
         f"- Escalations: {plan['escalations'] or 'none'}", "",
         "| Incident | Policy | Action | Risk | Attempts | Cooldown | Reason |", "|---|---|---|---|---|---|---|"]
    for d in plan["decisions"]:
        st = d.get("attempts") or {}
        L.append(f"| {d['incident_id']} | {d['policy']} | {d.get('action') or '—'} | {d.get('risk_class') or '—'} | "
                 f"{st.get('attempt_count', '—')}/{d.get('attempt_limit') or '—'} | {d.get('cooldown_min') or '—'} min | "
                 f"{d['reason']} |")
    if not plan["decisions"]:
        L.append("| — | NO_ACTION | — | — | — | — | no incidents |")
    return "\n".join(L) + "\n"


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
