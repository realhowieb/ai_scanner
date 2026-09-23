# Post-launch monitoring plan

Reuse existing artifacts/telemetry — no new vendor required.

## Per scheduled scan (from coverage_us_market.json / perf record)
- universe_source (alert if != live for N consecutive runs)
- coverage_percentage (alert if < 0.90 / DEGRADED)
- coverage_health (alert on DEGRADED/FAILED)
- attempted vs priced (skipped %); provider_trouble_events (alert on spike)
- rate_limit_events / timeout_events (provider throttling)
- total_runtime_seconds / symbols_per_second (alert if runtime > 2× median)
- snapshot_promoted (alert if a HEALTHY scan fails to promote)
- candidate_count (alert if 0 or wildly off baseline)

## Data pipeline
- observation capture written vs attempted (research_cohorts log line)
- cohort balance (candidate/near_miss/control present each scan)
- outcome maturation attached counts (maturation_report.json)
- run48 readiness evidence level (weekly)

## App / product
- Sentry exceptions (app + cron), rate + new-issue alerts
- alert volume per user (spam guard)
- Neon connection errors / latency

## Cadence
- Real-time: Sentry alerts.
- Per scan: coverage/perf artifacts (already uploaded).
- Weekly: run48 readiness + cohort balance + perf_history trend (once durable).

## Repeatability
`analytics.scan_reliability.compare_to_recent` flags universe-size / coverage /
runtime / throughput / provider-error anomalies vs recent HEALTHY runs.
