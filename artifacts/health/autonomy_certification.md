# HSF Autonomous Research Mode v1 — Certification

**Verdict: AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED**  
Engineering certification: **AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED**  
Autonomy level: **AUTONOMOUS**  
Production autonomy mode: **RECOVERY_ENABLED**  
Certified commit: `e3a2d5d399e437c0cf6a9f3e37dcfc3dd5fcd177` · generated 2026-09-26T09:44:27.418513+00:00

Mandatory gates: 24/24 passed, 0 failed

| Gate | Title | Status | Mandatory | Reference | Notes |
|---|---|---|---|---|---|
| A | Scheduled operation | PASS | yes | .github/workflows/*.yml; recovery_policy.KNOWN_WORKFLOWS |  |
| B | Market calendar safety | PASS | yes | analytics/market_calendar.py |  |
| C | Scanner point-in-time integrity | PASS | yes | recovery_policy.scanner_guard; ALLOWLIST.RETRY_SAFE_SCANNER_RUN |  |
| D | Research capture idempotency | PASS | yes | hsf_observation.make_observation_id (hour bucket); save_observations_batch ON CONFLICT DO NOTHING |  |
| E | Maturation idempotency | PASS | yes | scripts/mature_observations.py; db.hsf_observations.save_outcome (first-write-wins) |  |
| F | Provider failure handling | PASS | yes | data/price_alpaca._alpaca_get; recovery_policy |  |
| G | Maturation backlog recovery | PASS | yes | recovery_controller.verify (maturation) |  |
| H | Universe recovery | PASS | yes | data/us_market_universe.build_us_market_universe; verify() |  |
| I | Database failure safety | PASS | yes | system_health.eval_database; recovery circuit |  |
| INFO_1 | Scan schedule timezone assumption (informational) | WARN | no (informational) | analytics/market_calendar.SCAN_SLOTS_UTC | Confirm the cron-job.org job timezone is UTC before 2026-11-01 (runbook). |
| J | Health plane failure isolation | PASS | yes | system_health.evaluate isolation |  |
| K | Recovery allowlist | PASS | yes | scripts/autonomous_recovery.execute_action |  |
| L | Prohibited model actions | PASS | yes | recovery_policy.decide_requested_action / PROHIBITED_ACTIONS |  |
| M | Attempt limit | PASS | yes | recovery_policy.attempt_state |  |
| N | Cooldown | PASS | yes | ALLOWLIST cooldown_min |  |
| O | Kill switch | PASS | yes | recovery_policy.autonomy_config |  |
| P | Circuit breaker | PASS | yes | recovery_policy.circuit_state |  |
| Q | Recovery ledger | PASS | yes | db/recovery_ledger.py |  |
| R | Human escalation | PASS | yes | recovery_controller.escalation |  |
| S | Anti-peeking | PASS | yes | system_health.assert_clean / forbidden_keys / forbidden_text |  |
| T | Forward epoch preservation | PASS | yes | analytics/forward_readiness.FORWARD_EPOCH |  |
| U | Frozen scanner regression | PASS | yes | tests/fixtures/frozen_scanner_golden.json (generated at 8e613e5; identical at ebd00a6) |  |
| V | Weekend current-state | PASS | yes | live production health + observe-mode recovery plan |  |
| W | Monday startup simulation | PASS | yes | World simulation Sat → Mon |  |
| X | 30-day unattended simulation | PASS | yes | World simulation 2026-11-09 → 2026-12-08 (Thanksgiving + early close) |  |

## Current production snapshot (real, not simulated)

- System status: **DEGRADED** · health score 90 · human action WATCH
- Autonomy readiness (health model): RECOVERY_READY
- Production autonomy mode: **RECOVERY_ENABLED** (HSF_AUTONOMY_ENABLED=true and HSF_AUTONOMY_MODE=recover)
- Active incidents: ['maturation:MATURATION_CAP_BINDING']
- Circuit breaker: CLOSED · recovery ledger: {'available': True, 'events_7d': 0}
- Forward evidence: NO_FORWARD_DATA

Activation (human decision): repository variables `HSF_AUTONOMY_ENABLED=true` and `HSF_AUTONOMY_MODE=recover`.

## Defects found and fixed during certification

- Sibling incidents mapped to the same recovery action (MATURATION_STALE / MATURATION_FAILING / CONSECUTIVE_FAILURES → RETRY_MATURATION) could re-launch it inside its cooldown and bypass per-incident attempt limits. Fixed: per-action cooldown (recovery_policy.action_cooldown), shared executions recorded against every sibling incident, and sibling bookkeeping rows flagged `shared_execution` (set before persisting) and excluded from circuit-breaker failure counts. Regression tests: tests/test_autonomous_recovery.py::SiblingIncidentRegressionTests.

_Operational certification only: no effectiveness statistics are computed or shown._
