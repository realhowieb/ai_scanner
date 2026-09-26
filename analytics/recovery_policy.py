"""Run 60 — deterministic recovery policy engine (pure, no I/O).

Consumes a Run 59 `hsf-system-health-1.0` report plus the recovery ledger and
produces a recovery PLAN: exactly one policy per incident:
    NO_ACTION · WATCH · AUTO_RECOVER · ESCALATE

OPERATIONAL AUTONOMY ONLY. The engine can only ever choose an action from
`ALLOWLIST`. Each action maps to a *fixed* workflow file with *fixed* inputs;
incident text is never used to build a command. Anything that could touch
scoring, ranking, thresholds, cohorts, outcomes, research data, secrets,
schedules, provider logic or code is PROHIBITED and escalates.

State (attempts, cooldowns, circuit breaker) is derived only from the
append-only recovery ledger (`db/recovery_ledger.py`).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

from analytics import market_calendar as mc

SCHEMA = "hsf-recovery-plan-1.0"
POLICIES = ("NO_ACTION", "WATCH", "AUTO_RECOVER", "ESCALATE")
RISK_CLASSES = ("SAFE", "BOUNDED", "PROHIBITED")
RESULTS = ("SUCCESS", "FAILED", "SKIPPED", "COOLDOWN", "ATTEMPT_LIMIT", "PROHIBITED",
           "VERIFICATION_FAILED", "ESCALATED", "CIRCUIT_OPENED", "CIRCUIT_RESET")
ATTEMPT_RESULTS = ("SUCCESS", "FAILED", "VERIFICATION_FAILED")

# ---- Allowlist (the ONLY actions that can ever execute) ---------------------------------
# `workflow` / `inputs` are constants: never derived from incident content.
ALLOWLIST: Dict[str, Dict[str, Any]] = {
    "REGENERATE_SYSTEM_HEALTH": {
        "risk": "SAFE", "executor": "inline_health", "workflow": None, "inputs": {},
        "max_attempts": 2, "cooldown_min": 15, "wait_min": 0,
        "idempotency": "read-only health collection; snapshot table is append-only",
        "verification": "fresh health report generated; target artifact FRESH",
        "fallback": "ESCALATE",
    },
    "REGENERATE_FORWARD_READINESS": {
        "risk": "SAFE", "executor": "dispatch", "workflow": "forward-evidence-readiness.yml", "inputs": {},
        "max_attempts": 2, "cooldown_min": 30, "wait_min": 15,
        "idempotency": "read-only monitor; writes only its own artifact; never resets the Run 56 epoch",
        "verification": "workflow success and readiness artifact FRESH",
        "fallback": "ESCALATE",
    },
    "RERUN_PARITY_AUDIT": {
        "risk": "SAFE", "executor": "dispatch", "workflow": "maturation-parity-audit.yml", "inputs": {},
        "max_attempts": 2, "cooldown_min": 60, "wait_min": 25,
        "idempotency": "read-only audit; maturation runs in DRY RUN (no writes)",
        "verification": "workflow success and parity artifact FRESH",
        "fallback": "ESCALATE",
    },
    "RETRY_MATURATION": {
        "risk": "BOUNDED", "executor": "dispatch", "workflow": "mature-observations.yml",
        "inputs": {"slack_min": "15", "dry_run": "false"},
        "max_attempts": 2, "cooldown_min": 45, "wait_min": 25, "trading_days_only": True,
        "idempotency": "outcomes are first-write-wins on (observation_id, horizon); already-matured "
                       "horizons are skipped; cap/batch/formulas/retirement unchanged",
        "verification": "a new successful maturation run and the target maturation incident cleared "
                        "(or deferred/backlog reduced)",
        "fallback": "ESCALATE",
    },
    "RETRY_TRANSIENT_PROVIDER_OPERATION": {
        "risk": "BOUNDED", "executor": "dispatch", "workflow": "mature-observations.yml",
        "inputs": {"slack_min": "15", "dry_run": "false"},
        "max_attempts": 2, "cooldown_min": 60, "wait_min": 25, "trading_days_only": True,
        "idempotency": "re-runs the provider-dependent maturation after its own 429/backoff retries "
                       "were exhausted; same first-write-wins guarantees; no provider switch",
        "verification": "rate-limited / provider-error symbols no longer reported",
        "fallback": "ESCALATE",
    },
    "RETRY_UNIVERSE_PROBE": {
        "risk": "SAFE", "executor": "inline_health", "workflow": None, "inputs": {},
        "max_attempts": 2, "cooldown_min": 30, "wait_min": 0,
        "idempotency": "read-only Alpaca assets probe; the last-known-good cache is kept on failure",
        "verification": "universe source is live, non-empty and not SUSPICIOUS/BROKEN",
        "fallback": "ESCALATE",
    },
    "RETRY_UNIVERSE_REFRESH": {
        "risk": "BOUNDED", "executor": "dispatch", "workflow": "refresh-universe.yml", "inputs": {},
        "max_attempts": 2, "cooldown_min": 120, "wait_min": 10,
        "idempotency": "refresh scripts rewrite sp500/nasdaq lists only when the fetch is plausible "
                       "(min count + mega-cap sentinels); otherwise last-known-good is kept",
        "verification": "workflow success; universe subsystem not BROKEN/SUSPICIOUS",
        "fallback": "ESCALATE",
    },
    "RETRY_SAFE_SCANNER_RUN": {
        "risk": "BOUNDED", "executor": "dispatch", "workflow": "scheduled-scans.yml",
        "inputs": {"force": "false", "session": "auto"},
        "max_attempts": 1, "cooldown_min": 60, "wait_min": 15, "trading_days_only": True,
        # Not proven: a rerun in a different hour creates a NEW scan instance (new
        # hour-bucketed observation ids, an extra runs row, and alert/digest side
        # effects guarded only by throttles). Until proven, the guard escalates.
        "idempotency_proven": False,
        "idempotency": "observation ids dedupe only within the same hour bucket; scan-history and "
                       "alert side effects are not proven idempotent",
        "window_min": 30,
        "verification": "the missed slot has a successful run; no new research-capture incidents",
        "fallback": "ESCALATE",
    },
}
KNOWN_WORKFLOWS = frozenset(a["workflow"] for a in ALLOWLIST.values() if a["workflow"])

# Categories that can never be automated, whatever the request looks like.
PROHIBITED_ACTIONS = {
    "CHANGE_SCORING": "scoring formulas / weights", "CHANGE_RANKING": "ranking",
    "CHANGE_THRESHOLD": "tier / health / readiness thresholds",
    "CHANGE_CONFLICT_PENALTY": "conflict penalties", "REASSIGN_COHORT": "research cohort membership",
    "CHANGE_CANDIDATE_SELECTION": "candidate / near-miss / control selection",
    "CHANGE_UNIVERSE_RULES": "universe eligibility rules", "CHANGE_PREBREAKOUT": "PreBreakout logic",
    "REWRITE_OUTCOMES": "outcome values / formulas", "REWRITE_OBSERVATIONS": "research observations",
    "DELETE_DATA": "database deletion", "RESTORE_DATABASE": "database restoration / schema rebuild",
    "RESET_EPOCH": "Run 56 forward epoch", "CHANGE_READINESS_GATES": "Run 56 readiness gates",
    "CHANGE_PARITY_RULES": "Run 58 parity rules", "CHANGE_SECRETS": "production / GitHub secrets",
    "CHANGE_SCHEDULE": "workflow schedule definitions", "SWITCH_PROVIDER": "provider logic",
    "DEPLOY_CODE": "deploying new code", "SUPPRESS_INCIDENT": "incident suppression",
    "LOWER_HEALTH_THRESHOLD": "health thresholds", "RUN_FORMAL_EVALUATION": "formal effectiveness evaluation",
}

# ---- Incident → decision map (code driven; unknown codes escalate) ------------------------
NO_ACTION_CODES = {"RATE_LIMIT_RECOVERED", "DATA_AVAILABILITY_LIMITED"}
WATCH_CODES = {"MATURATION_CAP_BINDING", "SCHEDULE_GAP", "RETIREMENT_SPIKE", "FORWARD_PARITY_WARNING",
               "DATABASE_SLOW", "WORKFLOW_NEVER_RUN", "MATURATION_REPORT_MISSING", "UNIVERSE_HYGIENE",
               "ABNORMAL_DURATION"}
AUTO_CODES = {
    "MATURATION_STALE": "RETRY_MATURATION",
    "MATURATION_FAILING": "RETRY_MATURATION",
    "BACKLOG_GROWING": "RETRY_MATURATION",
    "RATE_LIMIT_PRESSURE": "RETRY_TRANSIENT_PROVIDER_OPERATION",
    "ALPACA_PERSISTENT_FAILURES": "RETRY_TRANSIENT_PROVIDER_OPERATION",
    "UNIVERSE_FALLBACK_CACHE": "RETRY_UNIVERSE_PROBE",
    "ALPACA_ASSETS_UNAVAILABLE": "RETRY_UNIVERSE_PROBE",
    "STALE_ARTIFACT_FORWARD_EVIDENCE_READINESS": "REGENERATE_FORWARD_READINESS",
    "STALE_ARTIFACT_MATURATION_PARITY_AUDIT": "RERUN_PARITY_AUDIT",
    "STALE_ARTIFACT_PREVIOUS_SYSTEM_HEALTH": "REGENERATE_SYSTEM_HEALTH",
    "MISSED_SCAN": "RETRY_SAFE_SCANNER_RUN",
    "FAILED_SCAN": "RETRY_SAFE_SCANNER_RUN",
    "STALE_SCANNER": "RETRY_SAFE_SCANNER_RUN",
}
# Workflow-level incidents resolve through the workflow named in the evidence,
# validated against a fixed map (never executed as a string).
WORKFLOW_ACTIONS = {
    "mature-observations.yml": "RETRY_MATURATION",
    "forward-evidence-readiness.yml": "REGENERATE_FORWARD_READINESS",
    "system-health.yml": "REGENERATE_SYSTEM_HEALTH",
    "refresh-universe.yml": "RETRY_UNIVERSE_REFRESH",
    "maturation-parity-audit.yml": "RERUN_PARITY_AUDIT",
    "scheduled-scans.yml": "RETRY_SAFE_SCANNER_RUN",
}
ESCALATE_REASONS = {
    "DATABASE_UNAVAILABLE": "database failures are human-controlled (no schema rebuild / restore / deletion)",
    "TABLE_UNREADABLE": "database integrity uncertain",
    "DUPLICATE_KEYS": "database integrity uncertain; never repaired automatically",
    "REQUIRED_FIELD_NULLS": "research data quality; never rewritten automatically",
    "UNIVERSE_EMPTY": "no safe automatic fix; a corrupt universe must never be accepted",
    "UNIVERSE_SIZE_JUMP": "suspicious universe; needs human review before trusting new scans",
    "FORWARD_PARITY_CRITICAL": "research-design decision (Run 58); not an operational fault",
    "FORWARD_EVIDENCE_BLOCKED": "research data-quality block; not operational",
    "FORMAL_EVALUATION_READY": "formal evaluation always requires explicit human approval",
    "ZERO_OBSERVATIONS": "research capture fault; rerunning cannot recover past point-in-time state",
    "COHORT_MISSING_IN_SCAN": "research capture fault",
    "MISSING_COHORT_LABEL": "research data; never rewritten",
    "DUPLICATE_OBSERVATIONS": "research data; never rewritten",
    "MALFORMED_OBSERVATIONS": "research data; never rewritten",
    "METADATA_INCOMPLETE": "capture-code issue; needs a human fix",
    "STALE_ARTIFACT_LATEST_SCANNER_OBSERVATION": "scanner capture stale; see scanner incidents",
}

# ---- Circuit breaker ----------------------------------------------------------------------
CB_WINDOW = _dt.timedelta(hours=6)
CB_MAX_FAILURES = 3
CB_MAX_ACTION_REQUIRED = 2
CB_MAX_CONSECUTIVE_VERIFY_FAILS = 2
ATTEMPT_WINDOW = _dt.timedelta(hours=24)


def _parse(v: Any) -> Optional[_dt.datetime]:
    if v is None:
        return None
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def autonomy_config(env: Mapping[str, str]) -> Dict[str, Any]:
    """Kill switch. Default is OFF / observe: plans are produced, nothing executes."""
    enabled = str(env.get("HSF_AUTONOMY_ENABLED", "false")).strip().lower() == "true"
    mode = str(env.get("HSF_AUTONOMY_MODE", "observe")).strip().lower()
    mode = mode if mode in ("observe", "recover") else "observe"   # no other mode exists
    execute = enabled and mode == "recover"
    return {"enabled": enabled, "mode": mode, "execution_permitted": execute,
            "production_state": "RECOVERY_ENABLED" if execute else "OBSERVE_ONLY",
            "reason": ("HSF_AUTONOMY_ENABLED=true and HSF_AUTONOMY_MODE=recover" if execute else
                       "kill switch off (HSF_AUTONOMY_ENABLED != true)" if not enabled else
                       "HSF_AUTONOMY_MODE=observe (plan only)")}


def decide_requested_action(action: str) -> Dict[str, Any]:
    """Gate for any explicitly requested action (e.g. from a future planner)."""
    a = str(action or "").strip().upper()
    if a in PROHIBITED_ACTIONS:
        return {"action": a, "risk_class": "PROHIBITED", "policy": "ESCALATE", "allowed": False,
                "reason": f"prohibited: self-healing may never modify {PROHIBITED_ACTIONS[a]}"}
    if a in ALLOWLIST:
        return {"action": a, "risk_class": ALLOWLIST[a]["risk"], "policy": "AUTO_RECOVER", "allowed": True,
                "reason": "allowlisted"}
    return {"action": a, "risk_class": "PROHIBITED", "policy": "ESCALATE", "allowed": False,
            "reason": "not in the recovery allowlist (no dynamic actions)"}


def _code(incident: Mapping[str, Any]) -> str:
    return str(incident.get("incident_id", "")).split(":", 1)[-1]


def _events_for(ledger: Sequence[Mapping[str, Any]], incident_id: str, action: str) -> List[Mapping[str, Any]]:
    evs = [e for e in ledger if e.get("incident_id") == incident_id and e.get("action") == action]
    return sorted(evs, key=lambda e: str(e.get("started_at") or e.get("detected_at") or ""))


def attempt_state(ledger: Sequence[Mapping[str, Any]], incident_id: str, action: str,
                  now: _dt.datetime) -> Dict[str, Any]:
    """Attempts in the current episode: within ATTEMPT_WINDOW and after the last
    verified clear of this incident."""
    evs = _events_for(ledger, incident_id, action)
    cleared = [e for e in evs if e.get("verification_result") == "CLEARED"]
    since = _parse(cleared[-1].get("completed_at")) if cleared else None
    tries = [e for e in evs if e.get("result") in ATTEMPT_RESULTS
             and (_parse(e.get("started_at")) or now) >= now - ATTEMPT_WINDOW
             and (since is None or (_parse(e.get("started_at")) or now) > since)]
    last = tries[-1] if tries else None
    cd = ALLOWLIST.get(action, {}).get("cooldown_min", 60)
    last_t = _parse(last.get("completed_at") or last.get("started_at")) if last else None
    next_ok = last_t + _dt.timedelta(minutes=cd) if last_t else None
    return {"attempt_count": len(tries),
            "first_attempt_at": tries[0].get("started_at") if tries else None,
            "last_attempt_at": last.get("started_at") if last else None,
            "last_result": last.get("result") if last else None,
            "next_allowed_attempt": next_ok.isoformat() if next_ok else None,
            "in_cooldown": bool(next_ok and now < next_ok)}


def action_cooldown(ledger: Sequence[Mapping[str, Any]], action: str, now: _dt.datetime) -> Dict[str, Any]:
    """Run 61 fix: cooldown is per ACTION, across every incident that maps to it, so a
    sibling incident (e.g. MATURATION_FAILING vs MATURATION_STALE) cannot re-launch the
    same recovery inside its cooldown."""
    tries = [e for e in ledger if e.get("action") == action and e.get("result") in ATTEMPT_RESULTS]
    last_t = max((_parse(e.get("completed_at") or e.get("started_at")) for e in tries
                  if _parse(e.get("completed_at") or e.get("started_at"))), default=None)
    nxt = last_t + _dt.timedelta(minutes=ALLOWLIST.get(action, {}).get("cooldown_min", 60)) if last_t else None
    return {"last_action_attempt_at": last_t.isoformat() if last_t else None,
            "next_allowed_action_attempt": nxt.isoformat() if nxt else None,
            "in_cooldown": bool(nxt and now < nxt)}


def circuit_state(health: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]],
                  now: _dt.datetime) -> Dict[str, Any]:
    """Global breaker. Once opened it stays open until a human appends CIRCUIT_RESET."""
    reasons: List[str] = []
    ordered = sorted(ledger, key=lambda e: str(e.get("started_at") or e.get("completed_at") or ""))
    opened = [e for e in ordered if e.get("result") == "CIRCUIT_OPENED"]
    reset = [e for e in ordered if e.get("result") == "CIRCUIT_RESET"]
    if opened and (not reset or str(reset[-1].get("started_at")) < str(opened[-1].get("started_at"))):
        reasons.append(f"circuit opened at {opened[-1].get('started_at')} and not reset by a human")
    recent = [e for e in ordered if (_parse(e.get("started_at")) or now) >= now - CB_WINDOW]
    # Count distinct executions: sibling bookkeeping rows (`shared_execution`) repeat
    # one execution's result for other incidents and must not inflate the breaker.
    fails = [e for e in recent if e.get("result") in ("FAILED", "VERIFICATION_FAILED")
             and not e.get("shared_execution")]
    if len(fails) >= CB_MAX_FAILURES:
        reasons.append(f"{len(fails)} recovery failures within {CB_WINDOW}")
    streak = 0
    for e in reversed([e for e in ordered if e.get("result") in ATTEMPT_RESULTS and not e.get("shared_execution")]):
        if e.get("result") == "VERIFICATION_FAILED":
            streak += 1
        else:
            break
    if streak >= CB_MAX_CONSECUTIVE_VERIFY_FAILS:
        reasons.append(f"{streak} consecutive verification failures")
    subs = health.get("subsystems") or {}
    ar = [n for n, s in subs.items() if s.get("status") == "ACTION_REQUIRED"]
    if len(ar) >= CB_MAX_ACTION_REQUIRED:
        reasons.append(f"{len(ar)} subsystems ACTION_REQUIRED: {sorted(ar)}")
    db = subs.get("database") or {}
    db_codes = {f.get("code") for f in db.get("findings") or []}
    if db.get("status") in ("UNKNOWN", "ACTION_REQUIRED") or db_codes & {"DUPLICATE_KEYS", "TABLE_UNREADABLE"}:
        reasons.append("database integrity uncertain")
    for e in recent:
        if e.get("new_critical_incidents"):
            reasons.append(f"recovery {e.get('recovery_id')} was followed by new critical incidents "
                           f"{e.get('new_critical_incidents')}")
    return {"open": bool(reasons), "state": "AUTONOMY_CIRCUIT_OPEN" if reasons else "CLOSED", "reasons": reasons}


def scanner_guard(incident: Mapping[str, Any], health: Mapping[str, Any], now: _dt.datetime) -> Dict[str, Any]:
    """All five Part 11 conditions must be PROVEN; otherwise escalate."""
    spec = ALLOWLIST["RETRY_SAFE_SCANNER_RUN"]
    ev = incident.get("evidence")
    slots = [_parse(x) for x in (ev if isinstance(ev, list) else [])]
    slots = [s for s in slots if s]
    latest = max(slots) if slots else None
    runs = ((health.get("subsystems") or {}).get("scanner") or {}).get("metrics") or {}
    last_ok = _parse(runs.get("most_recent_success"))
    conds = {
        "idempotency_proven": bool(spec.get("idempotency_proven")),
        "duplicate_ids_prevented_for_rerun": False,  # only within the same hour bucket
        "within_market_time_window": bool(latest and _dt.timedelta(0) <= now - latest
                                          <= _dt.timedelta(minutes=spec["window_min"])),
        "no_hindsight": bool(latest and mc.is_trading_day(latest.astimezone(mc.ET).date())),
        "no_successful_equivalent": not bool(latest and last_ok and last_ok >= latest),
    }
    failed = [k for k, v in conds.items() if not v]
    return {"conditions": conds, "allowed": not failed, "failed_conditions": failed}


def decide(incident: Mapping[str, Any], health: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]],
           now: _dt.datetime, circuit: Mapping[str, Any]) -> Dict[str, Any]:
    code = _code(incident)
    base = {"incident_id": incident.get("incident_id"), "incident_type": code,
            "subsystem": incident.get("subsystem"), "severity": incident.get("severity")}

    def out(policy, reason, action=None, requires_human=False, extra=None):
        spec = ALLOWLIST.get(action or "", {})
        d = {**base, "policy": policy, "reason": reason, "action": action,
             "risk_class": spec.get("risk") if action else ("PROHIBITED" if policy == "ESCALATE" else None),
             "attempt_limit": spec.get("max_attempts"), "cooldown_min": spec.get("cooldown_min"),
             "verification": spec.get("verification"), "fallback": spec.get("fallback", "ESCALATE"),
             "idempotency": spec.get("idempotency"), "requires_human": requires_human}
        if action:
            d["workflow"] = spec.get("workflow")
            d["attempts"] = attempt_state(ledger, incident.get("incident_id"), action, now)
        d.update(extra or {})
        return d

    if code in NO_ACTION_CODES:
        return out("NO_ACTION", "informational; already recovered or explained")
    if code in ESCALATE_REASONS:
        return out("ESCALATE", ESCALATE_REASONS[code], requires_human=True)
    action = AUTO_CODES.get(code)
    if code in ("WORKFLOW_STALE", "CONSECUTIVE_FAILURES"):
        wf = str((incident.get("evidence") or {}).get("workflow", "")) if isinstance(incident.get("evidence"), dict) else ""
        if wf not in WORKFLOW_ACTIONS:
            return out("ESCALATE", "workflow not recognized by the fixed recovery map", requires_human=True)
        if code == "CONSECUTIVE_FAILURES" and incident.get("severity") == "CRITICAL":
            return out("ESCALATE", "≥ 3 consecutive workflow failures need a human", requires_human=True)
        action = WORKFLOW_ACTIONS[wf]
    if code in WATCH_CODES and action is None:
        return out("WATCH", "expected or self-clearing; no recovery needed now")
    if action is None:
        return out("ESCALATE", "unknown incident type: no allowlisted recovery (no dynamic actions)",
                   requires_human=True)
    gate = decide_requested_action(action)
    if not gate["allowed"]:
        return out("ESCALATE", gate["reason"], requires_human=True)
    spec = ALLOWLIST[action]
    if circuit.get("open"):
        return out("ESCALATE", "AUTONOMY_CIRCUIT_OPEN: " + "; ".join(circuit["reasons"]), action, True)
    if action == "RETRY_SAFE_SCANNER_RUN":
        g = scanner_guard(incident, health, now)
        if not g["allowed"]:
            return out("ESCALATE", "scanner rerun not provably safe: " + ", ".join(g["failed_conditions"]),
                       action, True, {"scanner_guard": g})
    today = now.astimezone(mc.ET).date()
    if spec.get("trading_days_only") and not mc.is_trading_day(today):
        nxt = mc.next_expected_scan(now)
        return out("WATCH", f"not a trading day; next legitimate opportunity {nxt.isoformat() if nxt else 'n/a'}",
                   action)
    st = attempt_state(ledger, incident.get("incident_id"), action, now)
    if st["attempt_count"] >= spec["max_attempts"]:
        return out("ESCALATE", f"attempt limit reached ({st['attempt_count']}/{spec['max_attempts']})",
                   action, True, {"result_hint": "ATTEMPT_LIMIT"})
    acd = action_cooldown(ledger, action, now)
    if st["in_cooldown"] or acd["in_cooldown"]:
        until = max(x for x in (st["next_allowed_attempt"], acd["next_allowed_action_attempt"]) if x)
        return out("WATCH", f"cooldown until {until}", action, extra={"result_hint": "COOLDOWN"})
    return out("AUTO_RECOVER", f"allowlisted {spec['risk']} recovery", action)


def build_plan(health: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]], env: Mapping[str, str],
               now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Deterministic plan: same inputs → same output (excluding nothing)."""
    now = now or _parse(health.get("generated_at")) or _dt.datetime.now(_dt.timezone.utc)
    cfg = autonomy_config(env)
    circuit = circuit_state(health, ledger, now)
    decisions = [decide(i, health, ledger, now, circuit) for i in health.get("incidents") or []]
    decisions.sort(key=lambda d: (POLICIES.index(d["policy"]) * -1, str(d["incident_id"])))
    auto = [d for d in decisions if d["policy"] == "AUTO_RECOVER"]
    # One execution per action per cycle (several incidents can share an action).
    seen, to_execute = set(), []
    for d in auto:
        if d["action"] not in seen:
            seen.add(d["action"])
            to_execute.append(d)
    esc = [d for d in decisions if d["policy"] == "ESCALATE"]
    return {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "health_generated_at": health.get("generated_at"),
        "system_status": health.get("system_status"),
        "autonomy": cfg,
        "circuit_breaker": circuit,
        "decisions": decisions,
        "to_execute": [d["incident_id"] for d in to_execute] if cfg["execution_permitted"] and not circuit["open"]
        else [],
        "would_execute": [d["incident_id"] for d in to_execute],
        "escalations": [d["incident_id"] for d in esc],
        "human_action_required": bool(esc) or circuit["open"],
        "allowlist": sorted(ALLOWLIST),
    }
