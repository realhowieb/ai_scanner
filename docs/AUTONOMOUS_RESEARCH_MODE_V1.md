# HSF Autonomous Research Mode v1 — Runbook

**Operational autonomy: YES. Model autonomy: NO.**

HSF Autonomous Research Mode v1 means HSF can run this loop without routine human
intervention:

schedule → scan → capture → mature → monitor → detect → classify → recover safe
failures → verify → escalate the rest → keep collecting forward evidence.

It does **not** trade, place orders, modify models, tune scores or thresholds,
redesign cohorts, or interpret research results.

**Certification.** Run 61, `scripts/autonomy_certification.py`, gates A–X.
Artifacts are `artifacts/health/autonomy_certification.{json,md}`, and the
certified commit SHA is recorded inside the JSON.

**Engineering certification is separate from production mode.** Live recovery
also requires the kill switch below.

## Normal operation: what HSF does each trading day

All times are UTC (cron-job.org → `scheduled-scans.yml` on `main`; GitHub
schedules for the rest).

| When (UTC) | What | Workflow |
|---|---|---|
| 12:35 13:35 16:35 19:35 20:35 21:35 | Scheduled scans (pre-market, regular, post-market). Regular-session scans capture CANDIDATE / NEAR_MISS / CONTROL research observations with point-in-time metadata | `scheduled-scans.yml` (cron-job.org dispatch) |
| every 30 min, 13–22 | Outcome maturation (batched Alpaca, 429-safe, first-write-wins, 2,000-symbol cap, 6-day retirement) | `mature-observations.yml` |
| 23:15 | Forward-evidence readiness (Run 56, anti-peeking) | `forward-evidence-readiness.yml` |
| 17:05, 23:45 | System health: 10 subsystems, incidents, human action | `system-health.yml` |
| 17:25, 00:05 | Autonomous recovery: plan → (recover if enabled) → verify → escalate | `autonomous-recovery.yml` |
| Sunday 10:30 | Universe list refresh (plausibility-guarded) | `refresh-universe.yml` |

Weekends and NYSE holidays (2025–2027 calendar, including early closes) are
expected idle time. They create no incidents and no recovery.

## What HSF fixes by itself (the allowlist; nothing else can execute)

| Action | Risk | When | Limit |
|---|---|---|---|
| REGENERATE_SYSTEM_HEALTH | SAFE | stale health snapshot | 2 attempts, 15 min cooldown |
| REGENERATE_FORWARD_READINESS | SAFE | stale Run 56 readiness artifact / workflow | 2, 30 min |
| RERUN_PARITY_AUDIT | SAFE | Run 58 parity audit older than 7 days | 2, 60 min |
| RETRY_UNIVERSE_PROBE | SAFE | Alpaca assets probe fell back to cache | 2, 30 min |
| RETRY_MATURATION | BOUNDED | stale / failing / growing-backlog maturation (**trading days only**) | 2, 45 min |
| RETRY_TRANSIENT_PROVIDER_OPERATION | BOUNDED | provider 429 / errors after the built-in retries were exhausted (trading days only) | 2, 60 min |
| RETRY_UNIVERSE_REFRESH | BOUNDED | stale weekly universe refresh | 2, 120 min |
| RETRY_SAFE_SCANNER_RUN | BOUNDED | **Never executes today.** Scanner reruns always escalate, because rerun idempotency is not proven and point-in-time integrity beats completeness | 1, 60 min |

**Cooldowns are per action.** Sibling incidents that map to the same action
share one cooldown, and each counts the shared attempt. That is the Run 61 fix.

**Success means the incident cleared.** Every action is followed by a fresh
health report, and it counts as success only if the specific incident cleared
(or, for backlog, measurably improved).

## What HSF will never change

These are prohibited and always escalate:
- scoring formulas and weights, ranking;
- tier thresholds, conflict penalties;
- candidate / near-miss / control selection, cohort definitions;
- universe eligibility rules, PreBreakout;
- outcome, directional-return and MFE/MAE formulas;
- research observations and outcome values (no rewrite, no delete);
- database restore or schema rebuild;
- the Run 56 epoch and readiness gates, the Run 58 parity rules;
- secrets, workflow schedules, provider switching, code deployment;
- incident suppression, health thresholds;
- running the formal effectiveness evaluation.

## When Howard is needed (escalation)

Check `artifacts/health/human_escalation.md` (Autonomous Recovery run summary and
artifact) or **Admin → 🩺 System Health**. Human action is required when:
- the database is unavailable or its integrity is uncertain;
- the universe is empty or suspicious;
- the scanner missed or failed a scan (never auto-rerun);
- research capture has problems (zero observations, a missing cohort,
  duplicates, missing metadata);
- a recovery exhausted its 2 attempts, or verification failed;
- the circuit breaker opened;
- forward parity is CRITICAL or forward evidence is DATA_QUALITY_BLOCKED (a
  research-design decision);
- **Run 56 reports READY_FOR_RUN55_RERUN**: formal evaluation needs your approval.

## How to disable recovery (kill switch)

GitHub → Settings → Secrets and variables → Actions → **Variables**:
- **Off:** set `HSF_AUTONOMY_ENABLED` to `false`, or delete it. Monitoring and
  recovery plans continue; nothing executes.
- **On:** requires both `HSF_AUTONOMY_ENABLED=true` and `HSF_AUTONOMY_MODE=recover`.
  Any other value, including a typo such as `ture`, is OBSERVE_ONLY.

## How to inspect health

- **App:** Admin tab → System Health. It shows status, score, human action,
  subsystem cards, forward progress, incidents, providers and workflows.
- **Artifacts:** the `system-health` workflow artifact, `system_health.{json,md}`.
- **CLI (read-only):** `python -m scripts.system_health --out artifacts/health`.

## How to inspect recovery history

- The append-only Neon table `hsf_recovery_events`. Each row has the incident,
  action, attempt, result, timestamps, health before/after, verification and
  errors.
- Per run: the `autonomous-recovery` artifact (`recovery_plan`,
  `recovery_results`, `recovery_verification`, `human_escalation`).

## Circuit-open state

1. **The breaker opens on any of:**
   - ≥ 3 failed executions in 6 hours;
   - ≥ 2 consecutive verification failures;
   - ≥ 2 subsystems ACTION_REQUIRED;
   - database integrity uncertain;
   - a recovery followed by new critical incidents.
2. **While open:** every recovery escalates, and monitoring continues.
3. **To recover:**
   - Read `human_escalation.md` and the ledger, then fix the cause.
   - Confirm a fresh health run no longer shows it.
   - Run `python -m scripts.autonomous_recovery --reset-circuit --by <your name>`.
4. **It never resets itself.** If the problem still exists at the next check, it
   re-opens.

## How the formal effectiveness evaluation begins

1. Run 56 readiness reaches **READY_FOR_RUN55_RERUN**. Health then shows
   `FORMAL_EVALUATION_READY` (HUMAN_ACTION_REQUIRED).
2. You approve it and manually dispatch **Signal Effectiveness Analysis** with the
   frozen Run 55 criteria.
3. HSF never runs it automatically and never changes scoring afterwards. Any
   model change goes through a separate research-validation lifecycle, with its
   own new forward epoch.

## Post-certification change control

**Re-certification is required** (dispatch **Autonomy Certification**; the
verdict must be `AUTONOMOUS_RESEARCH_MODE_V1_CERTIFIED`) after changes to:
- recovery policy, the allowlist, prohibited classes, attempt limits, cooldowns or
  the circuit breaker (`analytics/recovery_*`, `db/recovery_ledger.py`,
  `scripts/autonomous_recovery.py`);
- health severity, incident or calendar logic (`analytics/system_health.py`,
  `analytics/market_calendar.py`);
- scanner idempotency or research persistence (`analytics/observation_capture.py`,
  `analytics/research_cohorts.py`, `analytics/research_metadata.py`,
  `db/hsf_observations.py`);
- maturation retry behaviour (`scripts/mature_observations.py`, `data/price_alpaca.py`);
- autonomy-relevant workflow schedules or triggers.

UI-only changes do not need re-certification. Model and scoring changes belong to
the research-validation lifecycle, not this one.

## Known assumption (informational gate INFO_1)

Expected scan slots are the cron-job.org **UTC** times. Before **2026-11-01**
(DST ends), confirm the cron-job.org job timezone is UTC. If it is
America/New_York, update `SCAN_SLOTS_UTC` (and re-certify), or the health plane
will report false missed scans in winter. The yearly calendar update is due
before 2028.

**Release identity:** HSF Autonomous Research Mode v1. The certified commit is in
`artifacts/health/autonomy_certification.json` → `certified_commit`.
