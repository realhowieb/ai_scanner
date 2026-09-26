# Run 60 — Safe self-healing & failure recovery

**Operational autonomy: YES. Model autonomy: NO.**

HSF can now detect a known operational problem, decide from code whether an
approved recovery exists, execute it within bounds, verify that the incident
actually cleared, and escalate when it cannot safely resolve it. It can never
change what the scanner predicts or what the research concludes.

**Autonomy readiness: RECOVERY_READY** (Run 59 health model, workflow run
36231358548). **Production mode: OBSERVE_ONLY**: the kill switch is off by
default, and recovery runs are plan-only until a human enables them.

## Architecture

```
system-health.yml (17:05 / 23:45 UTC)          autonomous-recovery.yml (17:25 / 00:05 UTC, + manual)
  └─ Run 59 health model ───────────────────►    1. regenerate health (read-only collectors)
                                                 2. recovery plan   (analytics/recovery_policy.py)
                                                 3. kill switch + circuit breaker
                                                 4. allowlisted action (fixed workflow + fixed inputs)
                                                 5. VERIFY: regenerate health, incident-specific checks
                                                 6. append-only ledger (hsf_recovery_events)
                                                 7. human_escalation.{json,md} if unresolved
```

| Module | Role |
|---|---|
| `analytics/recovery_policy.py` | Pure policy engine: allowlist, risk classes, incident→decision map, attempts, cooldowns, circuit breaker, scanner guard, kill switch |
| `analytics/recovery_controller.py` | Pure loop with injected I/O (execute / verify / append), verification rules, escalation artifact |
| `db/recovery_ledger.py` | Append-only `hsf_recovery_events` (append and read only; no update or delete API) |
| `scripts/autonomous_recovery.py` | CLI: plan (default), `--execute`, `--verify`, `--reset-circuit --by <name>` |
| `.github/workflows/autonomous-recovery.yml` | Scheduled after the health runs, plus manual dispatch; `actions: write` for fixed dispatches only |

## Policy engine

Every incident resolves to exactly one of **NO_ACTION, WATCH, AUTO_RECOVER or
ESCALATE**. Each decision records:
- incident_id, incident_type, subsystem, policy, reason;
- action, risk_class, attempt_limit, cooldown, verification, fallback;
- idempotency, requires_human, and the current attempt state.

The decision table is code, not judgment:

| Incident | Decision |
|---|---|
| RATE_LIMIT_RECOVERED, DATA_AVAILABILITY_LIMITED | NO_ACTION |
| MATURATION_CAP_BINDING, SCHEDULE_GAP, RETIREMENT_SPIKE, FORWARD_PARITY_WARNING, DATABASE_SLOW, WORKFLOW_NEVER_RUN, MATURATION_REPORT_MISSING, UNIVERSE_HYGIENE, ABNORMAL_DURATION | WATCH |
| MATURATION_STALE, MATURATION_FAILING, BACKLOG_GROWING | AUTO → RETRY_MATURATION |
| RATE_LIMIT_PRESSURE, ALPACA_PERSISTENT_FAILURES | AUTO → RETRY_TRANSIENT_PROVIDER_OPERATION |
| UNIVERSE_FALLBACK_CACHE, ALPACA_ASSETS_UNAVAILABLE | AUTO → RETRY_UNIVERSE_PROBE |
| STALE_ARTIFACT_{FORWARD_EVIDENCE_READINESS, MATURATION_PARITY_AUDIT, PREVIOUS_SYSTEM_HEALTH} | AUTO → REGENERATE_FORWARD_READINESS / RERUN_PARITY_AUDIT / REGENERATE_SYSTEM_HEALTH |
| WORKFLOW_STALE, CONSECUTIVE_FAILURES (WARNING) | The workflow named in the evidence, via a fixed map; unrecognized → ESCALATE |
| MISSED_SCAN, FAILED_SCAN, STALE_SCANNER | Scanner guard → currently always ESCALATE (see below) |
| Database, universe corruption, research-capture/data issues, forward-parity CRITICAL, evidence blocked, formal-evaluation-ready, ≥ 3 consecutive failures | ESCALATE |
| Anything else | ESCALATE (no dynamic actions) |

**Market calendar.** BOUNDED maturation and provider actions run only on
trading days. On weekends and holidays they are WATCH, with the next legitimate
opportunity named.

## Allowlist: the only executable actions

| Action | Risk | Executor | Max attempts | Cooldown | Verification |
|---|---|---|---|---|---|
| REGENERATE_SYSTEM_HEALTH | SAFE | inline, read-only | 2 | 15 min | fresh report; artifact FRESH |
| REGENERATE_FORWARD_READINESS | SAFE | dispatch `forward-evidence-readiness.yml` | 2 | 30 min | success; artifact FRESH |
| RERUN_PARITY_AUDIT | SAFE | dispatch `maturation-parity-audit.yml` (dry-run trace) | 2 | 60 min | success; artifact FRESH |
| RETRY_UNIVERSE_PROBE | SAFE | inline health (Alpaca assets probe) | 2 | 30 min | source live, non-empty, not SUSPICIOUS / BROKEN / STALE |
| RETRY_MATURATION | BOUNDED | dispatch `mature-observations.yml` (`slack_min=15`, `dry_run=false`) | 2 | 45 min | new successful run, incident cleared (or BACKLOG_GROWING deferred down ≥ 25%) |
| RETRY_TRANSIENT_PROVIDER_OPERATION | BOUNDED | same maturation dispatch, after its own 429/backoff retries were exhausted | 2 | 60 min | rate-limited / provider-error symbols gone |
| RETRY_UNIVERSE_REFRESH | BOUNDED | dispatch `refresh-universe.yml` | 2 | 120 min | success; universe not BROKEN / SUSPICIOUS |
| RETRY_SAFE_SCANNER_RUN | BOUNDED | dispatch `scheduled-scans.yml` (`force=false`, `session=auto`) | 1 | 60 min | guarded: currently always escalates |

**Dispatch safety.**
- Workflow and input values are constants. The workflow id is validated against
  the fixed set with a strict regex.
- `autonomous-recovery.yml` can never dispatch itself, so there is no recursion.
- An already-running equivalent run is detected; that is a SKIP and does not
  consume an attempt.

## Prohibited: always ESCALATE

`PROHIBITED_ACTIONS` covers 22 classes:
- scoring, ranking, thresholds, conflict penalties;
- cohort reassignment and candidate / near-miss / control selection;
- universe rules and PreBreakout;
- outcome and observation rewriting, deletion, database restore or schema rebuild;
- the Run 56 epoch and gates, and the Run 58 parity rules;
- secrets, schedules, provider switching and code deployment;
- incident suppression, lowering health thresholds, and running formal evaluation.

Any requested action that is not allowlisted is also rejected
(`decide_requested_action`).

## Idempotency

| Path | Guarantee |
|---|---|
| Maturation | First-write-wins on `(observation_id, horizon)`; matured horizons are skipped. Cap, batch size, formulas and retirement are unchanged. |
| Readiness / parity / health | Read-only. Artifacts are overwritten and snapshots appended. The epoch is never reset. |
| Universe refresh | Scripts rewrite the sp500 / nasdaq lists only on a plausible fetch (count floor plus mega-cap sentinels); otherwise last-known-good is kept. |
| Scanner | **Not proven.** Observation ids deduplicate only within the same hour bucket. A later rerun creates a new scan instance, an extra runs row, and alert/digest side effects guarded only by throttles. |

**Scanner guard.** All five conditions must be proven:
1. idempotency;
2. duplicate-id prevention for the rerun;
3. the rerun is within 30 minutes of the missed slot;
4. no hindsight;
5. no successful equivalent run already exists.

Condition 1 is false by design, so **scanner reruns always ESCALATE**. A 10:00
ET scan is never replayed at 15:30 ET.

## Attempts, cooldowns, circuit breaker, kill switch

- **Attempts:** counted per (incident, action) within 24 hours and since the
  last verified clear. Hitting the limit produces ESCALATE and an ESCALATED
  ledger event.
- **Cooldown:** per action. A recovery that is due during its cooldown becomes
  WATCH, and a COOLDOWN event is written. Repeated health checks never relaunch it.
- **Circuit breaker:** opens on any of:
  - ≥ 3 recovery failures in 6 hours;
  - ≥ 2 consecutive verification failures;
  - ≥ 2 subsystems ACTION_REQUIRED;
  - database status UNKNOWN or ACTION_REQUIRED, or duplicate keys / unreadable tables;
  - a recovery followed by new critical incidents (further actions stop in the
    same cycle).

  Once open it **stays open until a human appends a CIRCUIT_RESET**:
  `python -m scripts.autonomous_recovery --reset-circuit --by <name>`. While it
  is open, every AUTO decision escalates. Monitoring continues throughout.
- **Kill switch:** execution requires both `HSF_AUTONOMY_ENABLED=true` **and**
  `HSF_AUTONOMY_MODE=recover`. Anything else is OBSERVE_ONLY: the full decision
  path runs, SKIPPED ("would execute") events are logged, and nothing executes.
  No mode permits model or scoring changes.
- **Ledger unavailable:** plan only, because bounds cannot be enforced without it.

## Verification (no false success)

After every executed action, health is regenerated and checked for that specific
incident:
- **Maturation:** a new successful run, the target incident gone, and no
  MATURATION_STALE or MATURATION_FAILING. For BACKLOG_GROWING, deferred symbols
  must fall by at least 25% (IMPROVED).
- **Universe:** the incident gone *and* the universe non-empty and not
  suspicious. An exit-0 refresh that leaves a 0-symbol or shrunken universe is
  not success.
- **Artifacts:** the target artifact is FRESH.

Command failure is recorded as FAILED, exit 0 with the incident still present as
VERIFICATION_FAILED, and either consumes an attempt.

## Ledger (`hsf_recovery_events`)

Fields: recovery_id, incident_id, incident_type, action, policy, risk_class,
detected_at, started_at, completed_at, attempt, result, verification_result,
verification_checks, reason, error (scrubbed), health_before, health_after and
new_critical_incidents.

Results: SUCCESS, FAILED, SKIPPED, COOLDOWN, ATTEMPT_LIMIT, PROHIBITED,
VERIFICATION_FAILED, ESCALATED, CIRCUIT_OPENED, CIRCUIT_RESET. Nothing deletes
rows.

## Escalation

`artifacts/health/human_escalation.{json,md}` (also in the workflow summary):
- one section per escalated incident: detection time, attempts (x/y), recovery
  outcome, why it escalated, what HSF already tried (action → result →
  verification), and the recommended human action;
- a circuit-breaker section when the circuit is open.

## Production state (2026-09-26 08:57 UTC, Saturday)

- **Recovery run 36231285780 (observe):** plan = `maturation:MATURATION_CAP_BINDING → WATCH`.
  Nothing would execute and nothing escalated. Circuit CLOSED, ledger available.
- **Friday's 400-cap backlog:** correctly WATCH. No weekend recovery; Monday's
  2,000-cap scheduled maturation is the next opportunity.
- **System health:** DEGRADED (90), human action WATCH, autonomy
  **RECOVERY_READY**.
- **Why OBSERVE_ONLY:** the repository variables `HSF_AUTONOMY_ENABLED` and
  `HSF_AUTONOMY_MODE` are unset. Enabling them is a human decision.

## How to enable / disable / dry-run

- **Enable:** GitHub → Settings → Secrets and variables → Actions → **Variables**:
  `HSF_AUTONOMY_ENABLED=true`, `HSF_AUTONOMY_MODE=recover`.
- **Disable immediately:** set `HSF_AUTONOMY_ENABLED=false` (or delete it). Health
  monitoring and plans continue.
- **Dry run:** Actions → Autonomous Recovery → Run workflow → mode `observe`, or
  locally `python -m scripts.autonomous_recovery` (plan only).
- **Reset the circuit** (human only): `python -m scripts.autonomous_recovery --reset-circuit --by <name>`.

## Security

- No process spawning or shell execution, and no dynamic command construction
  (tests scan the modules).
- Incident text and evidence are never executed. A test injects shell
  metacharacters and path traversal, and both escalate.
- Workflow ids are regex-validated against a fixed set, and inputs are
  constants.
- Tokens and DB/Alpaca secrets are scrubbed from any error text. Artifacts are
  safe to publish.

## Run 61 certification scope (AUTONOMOUS is never self-declared)

1. **Controlled enablement trial:** turn on `recover` for about two trading
   weeks. Require ≥ 1 real verified recovery per allowlisted class that
   occurs, 0 prohibited attempts, 0 false successes, and 0 circuit-breaker
   misfires.
2. **Chaos drills in a sandbox branch:** a forced maturation failure, a stale
   artifact, an Alpaca outage (mocked 429s), DB unreachable, and verification
   failure → circuit → human reset. Confirm the ledger and escalation for each.
3. **Notification delivery:** HUMAN_ACTION_REQUIRED sent via the existing
   Resend SMTP, with dedupe.
4. **Scanner idempotency:** either prove it (same-slot dedupe of runs rows and
   alerts) or keep scanner recovery permanently human.
5. **Scan coverage telemetry:** persist attempted/processed counts
   (PARTIAL_SCAN) and instrument the yfinance fallback.
6. **Calendar:** cross-check the calendar with Alpaca, and add yearly updates.
7. **Independent review** of the allowlist, prohibited classes and ledger;
   sign-off recorded in the repo.
