# Run 59 — HSF system health control plane

One command answers **"Does HSF need me right now?"**:

```
python -m scripts.system_health --out artifacts/health [--persist]
```

It produces `artifacts/health/system_health.json` (schema `hsf-system-health-1.0`,
the contract Run 60 will consume) and `system_health.md`. The Streamlit
**Admin → 🩺 System Health** view renders the same model from the persisted
snapshot. The workflow `system-health.yml` runs on weekdays at 17:05 and 23:45 UTC
and can be triggered manually.

The control plane observes, classifies, prioritizes and recommends. It performs
**no self-healing**, which is Run 60. It changes no scoring, ranking, cohort,
outcome, maturation or Run 56 epoch logic, and it exposes no effectiveness
statistics.

## Current production status (workflow run 36230345355, 2026-09-26 08:38 UTC, Saturday)

**System Status: DEGRADED · Health Score 90/100 · Human Action: WATCH ·
Autonomy: OBSERVABLE**

| Subsystem | Status | Why |
|---|---|---|
| Universe | HEALTHY | 11,533 tradable US-listed symbols (live Alpaca) |
| Scanner | HEALTHY | 18/18 expected slots in the last 3 trading days ran successfully |
| Research Capture | HEALTHY | 3,363 observations (CANDIDATE 1,300 · NEAR_MISS 663 · CONTROL 1,400); 0 untagged / duplicates / malformed |
| Maturation | DEGRADED | Latest *scheduled* run (Fri 23:41 UTC, old 400 cap) deferred 1,213 ready symbols |
| Market Data | HEALTHY | Alpaca assets endpoint OK. The latest maturation report predates request telemetry (schema 1.1) |
| Cohort Parity | COLLECTING | Forward parity not yet measurable. Historical CRITICAL is **explained** (MARKET_DATA_AVAILABILITY_EFFECT, HIGH) |
| Forward Evidence | NO_FORWARD_DATA | Day 0/20, runs 0/100. Expected: no scan has run since the epoch |
| Workflows | HEALTHY | 6/6 |
| Database | HEALTHY | Connected, 161 ms |
| Artifact Freshness | HEALTHY | 4/4 fresh |

**The one incident:** `maturation:MATURATION_CAP_BINDING` (WARNING, WATCH, not an
automation candidate).
- **It is accurate.** The backlog is pending until Monday's first scheduled run
  under the 2,000 cap. The Run 58 dry run showed that cap reaches all 1,593
  ready symbols.
- **It should clear on its own** once that run reports 0 deferred symbols.
- **Next expected scan:** 2026-09-28 12:35 UTC (Monday 08:35 ET).
- **Collection:** 10.6 s, 0 collector errors, snapshot persisted.

## Health model

- **Subsystems.** There are ten: universe, scanner, research capture,
  maturation, market data, cohort parity, forward evidence, workflows, database
  and artifact freshness. Each reports `status`, `detail_state`, `reason`,
  `observed_value`, `expected_value`, `last_updated`, `recommended_action`,
  `human_action`, `findings` and `metrics`.
- **Statuses.**
  - HEALTHY, DEGRADED and ACTION_REQUIRED are as named.
  - **WAITING** means expected research or market waiting, such as COLLECTING
    or NO_FORWARD_DATA. It counts as healthy and creates no incidents.
  - **UNKNOWN** means telemetry is missing. It is never faked as HEALTHY.
- **System status.** ACTION_REQUIRED > DEGRADED > UNKNOWN > HEALTHY. Any
  CRITICAL finding makes its subsystem ACTION_REQUIRED, and any WARNING makes it
  DEGRADED.
- **Human action.** NO_ACTION < WATCH < AUTOMATIC_RECOVERY_CANDIDATE <
  HUMAN_ACTION_REQUIRED. Formal-evaluation readiness is HEALTHY but
  **HUMAN_ACTION_REQUIRED**: a human must approve it, and it never runs
  automatically.
- **Incidents.** Every finding becomes an incident with an ID, severity,
  subsystem, detection time, age, evidence, recommended action and automation
  flag. They are sorted by severity, then age. Expected waiting never creates one.
- **Health score.** 100 minus 25 per ACTION_REQUIRED, 10 per DEGRADED and 5 per
  UNKNOWN subsystem. It is secondary: a database outage still reads
  ACTION_REQUIRED even at a score of 75 (tested).
- **Isolation.** Each collector and evaluator is isolated. One failure becomes
  UNKNOWN for that subsystem only, and the rest of the report still renders
  (tested).

## Calendar awareness

`analytics/market_calendar.py` holds the NYSE full-day holidays and early closes
for 2025–2027. It also records the cron-job.org dispatch slots (12:35, 13:35,
16:35, 19:35, 20:35 and 21:35 UTC), which were verified reliable to ±1 minute.

Freshness is measured in **completed trading days**, with a 3-hour grace after
the close, rather than "hours since". A missed scan is an expected slot on a
trading day with no run in [−10 min, +45 min]. Weekends and holidays have no
expected slots, so they create no findings (tested for a Saturday, a Sunday and
Thanksgiving).

## Detections

| Subsystem | Findings |
|---|---|
| Universe | UNIVERSE_EMPTY (BROKEN) · UNIVERSE_FALLBACK_CACHE (STALE) · UNIVERSE_SIZE_JUMP ≥ 10% (SUSPICIOUS) / ≥ 30% (BROKEN) · UNIVERSE_HYGIENE |
| Scanner | MISSED_SCAN · FAILED_SCAN (≥ 3 consecutive is critical) · STALE_SCANNER (> 3.5 h while open; or no success on the last trading day) · ABNORMAL_DURATION (> 20 min, or stuck > 45 min). Result-row count is deliberately **not** a signal. |
| Research capture | ZERO_OBSERVATIONS (a successful regular-session scan captured nothing) · COHORT_MISSING_IN_SCAN (except a same-hour second scan, whose rows are deduplicated by design) · MISSING_COHORT_LABEL · DUPLICATE_OBSERVATIONS · MALFORMED_OBSERVATIONS · METADATA_INCOMPLETE (< 95% Run 57 metadata) |
| Maturation | MATURATION_STALE · MATURATION_FAILING · RUNTIME_PRESSURE (> 15 min) · RATE_LIMIT_PRESSURE / RATE_LIMIT_RECOVERED · BACKLOG_GROWING / MATURATION_CAP_BINDING · RETIREMENT_SPIKE · DATA_AVAILABILITY_LIMITED (INFO, explained by Run 58) · MATURATION_REPORT_MISSING |
| Market data | Alpaca (requests, 429s, retries, failures, empty responses, assets endpoint, last success) · yfinance listed separately as **not instrumented** |
| Cohort parity | Historical (Run 58: gaps, classes, root cause, whether it is explained) is kept separate from forward-epoch parity (Run 56). Forward WARNING → DEGRADED; forward CRITICAL → ACTION_REQUIRED (a research-design decision) |
| Forward evidence | NO_FORWARD_DATA / COLLECTING (WAITING) · DATA_QUALITY_BLOCKED (ACTION_REQUIRED) · READY_FOR_FORMAL_EVALUATION (human approval). Reports progress only: days, runs, coverage, gate PASS/WARN/FAIL |
| Workflows | Six known workflows with cadence: WORKFLOW_STALE · CONSECUTIVE_FAILURES · SCHEDULE_GAP (only within the history actually returned) · WORKFLOW_NEVER_RUN |
| Database | DATABASE_UNAVAILABLE · TABLE_UNREADABLE · DATABASE_SLOW (> 2 s) · DUPLICATE_KEYS · REQUIRED_FIELD_NULLS (> 1%) |
| Artifacts | Run 56 readiness, Run 58 parity audit (≤ 7 days), latest scanner capture, previous health snapshot |

## Anti-peeking

The forward-evidence evaluator copies only whitelisted progress fields from the
Run 56 report. `assert_clean` rejects both effectiveness-like **keys** (return,
win, pnl, mfe, mae, correlation, effect, …) and effectiveness **phrases** (win
rate, mean/median return, effect size, candidate/control return) in the JSON. The
script re-checks the Markdown.

A test injects `win_rate` and `mean_return` into the readiness input and proves
neither reaches the output. The Streamlit view renders the same guarded model.

## Autonomy readiness: OBSERVABLE

**Blockers to RECOVERY_READY** (Run 60):
- no automatic re-dispatch of stale or missed workflows;
- no automatic maturation retry escalation;
- no notification layer for HUMAN_ACTION_REQUIRED;
- no recovery audit trail or guardrails (retry limits, kill switch).

**Additional blockers to AUTONOMOUS:**
- scanner coverage telemetry (symbols attempted/processed) is not persisted, so
  PARTIAL_SCAN is undetectable;
- the yfinance fallback is not instrumented;
- the calendar only covers 2025–2027;
- formal evaluation is human-approved by design.

## Fixes found while validating on production

The first production run surfaced two false positives, both fixed and tested:
1. PARTIAL_SCAN was inferred from small saved result counts. Few candidates is a
   market outcome, not a scanner fault.
2. COHORT_MISSING_IN_SCAN fired on a second scan within the same hour, where
   duplicate-ID deduplication is by design.

The view's test also now skips cleanly where `streamlit` is not installed.

## Recommended Run 60 scope

Consume `system_health.json` to act on `automation_candidates`:
- **Guarded re-dispatch** of scheduled-scans, mature-observations,
  forward-evidence-readiness and maturation-parity-audit. These are idempotent,
  and need rate limits, max attempts per day, a kill switch and an audit log.
- **Notifications** (email via the existing Resend SMTP) whenever
  `human_action == HUMAN_ACTION_REQUIRED`.
- **Persist scan coverage telemetry**, so PARTIAL_SCAN becomes detectable.

Never automate formal evaluation, experiment-design changes, or database repair.
