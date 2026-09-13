# HSF Market Intelligence — V1.0 Release Candidate

Operational reference for deploying and running HSF. Produced by Run 30
(production hardening). **No secret values appear in this document** — only
variable names and purpose.

## Product surfaces (V1)
- **Market Brief** — what matters now
- **Scanner** — discover
- **My Watchlist** — what matters to me (Run 27, batch-composed)
- **Stock Intelligence** — why a ticker matters
- **Alerts** — notify on meaningful state change
- **Intelligence Performance** — what HSF has learned historically (admin/read-only)

Intelligence loop: Observe → Score → Explain → Freeze → Track → Detect change →
Notify → Verify operation → Measure alert follow-through → Measure opportunity
evolution → Analyze historical evidence → Personalize → Surface.

## Release identifier
Release = the deployed commit SHA. Admins see it under the Alerts page →
Intelligence health expander (`config_validation.build_info`), sourced from
`HSF_COMMIT_SHA` (or `GIT_COMMIT` / `RENDER_GIT_COMMIT` / git fallback).

## Environment variables (names + purpose only — never commit values)
**Required**
- `NEON_DATABASE_URL` **or** `DATABASE_URL` — primary Postgres/Neon database. Absent → pages degrade to empty state; persistence unavailable.

**Optional (feature degrades gracefully if absent)**
- `COOKIE_PASSWORD` — signs the auth session cookie. Absent → login may not persist across reloads.
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` — primary market data. Absent → fallback provider used where available.
- `RESEND_API_KEY` — transactional email for alerts. Absent → email skipped; in-app alert feed still works.
- `ADMIN_USERS` — comma-separated admin usernames. Absent → admin diagnostics hidden.
- `DB_CONNECT_TIMEOUT` — seconds for the DB connect timeout (default 10).
- `CRON_FORCE=1` — bypasses the weekend/holiday scan skip for manual workflow runs.

Validate at deploy time with `config_validation.validate_config()` (returns
`ok` + `missing_required`, no values).

## Test command (canonical regression gate)
```bash
.venv/bin/python -m pytest -q
```
Lint: `python3.9 -m ruff check . --select E9,F,I`. CI mirrors this in
`.github/workflows/smoke.yml`.

## Background jobs (cron pipeline)
Entry point: `scheduler/cron_runner.py::main` (scheduled via
`.github/workflows/scheduled-scans.yml`). Ordered, each step best-effort and
isolated (a failure logs and never aborts later steps):
1. session scan + snapshot save — reads providers, writes scan results/snapshot.
2. alert outcome scoring (price pipeline) — idempotent on unscored events.
3. `freeze_latest_opportunities` — freezes HSF opportunities (idempotent per snapshot/ticker).
4. `run_intelligence_alert_evaluation` — detect state changes, match users, deliver (dedupe + `ON CONFLICT`).
5. `mature_alert_outcomes` (Run 24) — idempotent per alert_id+horizon.
6. `mature_opportunity_outcomes` (Run 25) — idempotent per observation_id+horizon, first-observation immutable.
7. `score_pending_signal_outcomes` — settles 1/3/5D price windows (≥8-day gate).

**Idempotency** is enforced by unique keys / `ON CONFLICT DO NOTHING` / persisted
keys — reruns and overlapping runs cannot duplicate logical records. **Historical
intelligence is never reset.**

## Deployment checklist
**Before deploy:** tests pass · required secrets configured · DB reachable ·
schema compatible (additive `CREATE TABLE/INDEX IF NOT EXISTS` only) ·
scheduled-scans workflow valid.
**After deploy:** login works · Market Brief loads · Scanner loads · watchlist
read/write works · Stock Intelligence loads · Alerts load · admin health shows
expected build SHA + config health.

## Health verification
Admin → Alerts → 🩺 Intelligence health: build SHA, config health (missing
required / degraded optional), Run 23 operational status (HEALTHY / STALE /
UNKNOWN / DEGRADED), latest evaluation metrics.

## Graceful degradation matrix
| Dependency down | Still works | User sees |
|---|---|---|
| Database | static shell | "Database temporarily unavailable"; empty intelligence |
| Primary market provider | persisted intelligence, watchlist state | fallback or "market data temporarily unavailable" |
| Email (Resend) | in-app alert feed, detection | email skipped silently |
| Model artifact | score + all non-model surfaces | model-derived detail omitted |
| Historical evidence | all primary pages | "building evidence / insufficient history" |
| Intelligence Performance | everything else | empty readiness (EARLY) |

## Known limitations (V1)
- Streamlit **cold start** is slower than warm targets (see performance report).
- Market-data **provider rate limits** can slow the Scanner under heavy use.
- Historical **evidence is early** — most performance cohorts read INSUFFICIENT_SAMPLE until data matures.
- **Market regime** is not frozen per observation → by-regime analytics report INSUFFICIENT_COVERAGE.
- Background-job **cadence** bounds evidence freshness (snapshots a few times per weekday).

## Rollback plan
1. Identify the last known-good commit SHA (admin health shows the deployed SHA).
2. Redeploy that commit (Streamlit Cloud: point to the prior commit/branch).
3. **Do not** run destructive schema rollback — all V1 migrations are additive and backward-compatible.
4. Verify DB reachable + login + Market Brief + admin health.
5. Confirm the scheduled workflow still points at `scheduler/cron_runner.py::main`.
6. Investigate the failure offline from logs.

## Data safety
All Run 30 schema changes are additive (`IF NOT EXISTS`). No historical table or
column is dropped/renamed. Opportunity observations, alerts, alert-quality and
opportunity outcomes, calibration data, and watchlists are preserved assets.
