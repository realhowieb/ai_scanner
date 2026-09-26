# HSF SYSTEM HEALTH

**System Status: DEGRADED**  
Health Score: 90/100  
Human Action: **WATCH**  
Generated: 2026-09-26T08:38:17.940346+00:00

- Universe: **HEALTHY** — 11533 tradable US-listed symbols (live)
- Scanner: **HEALTHY** — 18/18 expected slots in the last 3 days ran successfully
- Research Capture: **HEALTHY** — 3363 research observations in the last 3 days
- Maturation: **DEGRADED** — last successful maturation 2026-09-25T23:41:57+00:00
- Market Data: **HEALTHY** — Alpaca assets endpoint OK; no request telemetry in the latest maturation report (schema hsf-maturation-1.1, pre-hardening)
- Cohort Parity: **COLLECTING** — forward parity not yet measurable; historical parity CRITICAL (explained: MARKET_DATA_AVAILABILITY_EFFECT)
- Forward Evidence: **NO_FORWARD_DATA** — NO_FORWARD_DATA: day 0/20, runs 0/100
- Workflows: **HEALTHY** — 6/6 workflows healthy
- Database: **HEALTHY** — connected, 161.0 ms
- Artifact Freshness: **HEALTHY** — 4/4 artifacts fresh

## Forward Evidence

- Status: **NO_FORWARD_DATA** (epoch 2026-09-26T07:23:11+00:00)
- Trading Days: 0 / 20 (minimum 10)
- Scan Runs: 0 / 100 (minimum 50)
- Lowest cohort maturation coverage: None% (≥ 80%, +60m ≥ 70%)
- Gates: {'A_trading_days': 'FAIL', 'B_scan_runs': 'FAIL', 'C_cohort_clusters': 'FAIL', 'D_horizon_maturation': 'FAIL', 'E_maturation_parity': 'FAIL', 'F_directional_integrity': 'FAIL', 'G_effective_clusters': 'FAIL', 'H_research_integrity': 'PASS'}
- Formal Evaluation: **NOT READY**

## Incidents

None requiring human action.
1. **maturation:MATURATION_CAP_BINDING** — WARNING · age 8.9h · human action WATCH · automation candidate NO
   1213 ready symbols deferred by the per-run cap. Recommended: cohort-neutral only while the cap does not bind (Run 58); review MAX_SYMBOLS

Next expected scan: 2026-09-28T12:35:00+00:00 (trading day today: False, market open: False)

## Autonomy Readiness

**OBSERVABLE**

Blockers to RECOVERY_READY:
- no automatic re-dispatch of stale/missed workflows (scan, maturation, readiness, parity audit)
- no automatic maturation retry escalation
- no incident notification layer (email/Slack) for HUMAN_ACTION_REQUIRED
- no recovery audit trail / guardrails (rate limits, max retries, kill switch)

Additional blockers to AUTONOMOUS:
- scanner coverage telemetry (symbols attempted/processed) not persisted
- yfinance fallback usage not instrumented
- calendar covers 2025-2027 only (needs yearly update or Alpaca calendar cross-check)
- formal evaluation requires explicit human approval by design (never automatic)

## Provider Health

| Provider | Instrumented | Requests | 429s | Retries | Failures | Empty | Last success |
|---|---|---|---|---|---|---|---|
| alpaca | True | None | None | None | 0 | None | None |
| yfinance | False | — | — | — | — | — | — |

## Workflow Freshness

| Workflow | Cadence | Last success | Last failure | Status |
|---|---|---|---|---|
| Scheduled market scan | trading_day | 2026-09-25T23:55:11+00:00 | None | HEALTHY |
| Mature observations | trading_day | 2026-09-26T06:47:07+00:00 | None | HEALTHY |
| Universe refresh | weekly | 2026-09-20T14:16:52+00:00 | None | HEALTHY |
| Research readiness (Run 56) | trading_day | 2026-09-26T07:57:34+00:00 | None | HEALTHY |
| Maturation parity audit (Run 58) | manual | 2026-09-26T08:13:12+00:00 | None | HEALTHY |
| System health (Run 59) | trading_day | 2026-09-26T08:34:53+00:00 | None | HEALTHY |

## Artifact Freshness

| Artifact | Generated | Age (h) | Expected | Status |
|---|---|---|---|---|
| Run 56 readiness | 2026-09-26T07:57:34+00:00 | 0.7 | ≤ 1 completed trading day | FRESH |
| Run 58 maturation parity | 2026-09-26T08:13:57.542686+00:00 | 0.4 | ≤ 7 days | FRESH |
| Latest scanner research capture | 2026-09-25T19:40:02.460183+00:00 | 13.0 | ≤ 1 completed trading day | FRESH |
| Previous health snapshot | 2026-09-26T08:35:36.744422+00:00 | 0.0 | ≤ 1 completed trading day | FRESH |

## Research Data Quality

- Recent observations: 3363 by cohort {'CONTROL': 1400, 'CANDIDATE': 1300, 'NEAR_MISS': 663}
- Untagged 0 · duplicates 0 · malformed 0 · Run 57 metadata None% of 0

_Operational observability only: no effectiveness statistics are computed or shown._
