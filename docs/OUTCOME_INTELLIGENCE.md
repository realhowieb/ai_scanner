# Outcome Intelligence (`/v1/outcomes/*`)

One canonical, benchmark-relative evidence layer for HSF signals. It answers
"how have HSF signals actually performed?" from the existing outcome store and
never computes an outcome of its own.

```
signal_outcomes (source='opportunity')        frozen at signal time by the cron / Market Brief
   ↓  analytics.signal_outcomes               1/3/5-day returns, 5-day MFE/MAE, SPY same window
analytics.outcome_intelligence                canonical records, maturity, filters, metrics()
   ↓
api.outcomes (cache) → /v1/outcomes/*         Track Record, Stock page, AI ticker note
```

## Source of truth

| Field | Source | Point-in-time? |
|---|---|---|
| observation_id | `signal_outcomes.id` | yes |
| ticker, observed_at | `ticker`, `fired_at` (scan snapshot time) | yes |
| hsf_score, score_version, setup, signals, status | frozen `raw_signal` / `indicators` via `hsf_calibration.normalize_row` | yes (never today's scan) |
| prebreakout_prob | `prebreakout_prob` frozen column | yes |
| raw_return (1/3/5d) | `return_1d/3d/5d` | outcome, written after the window |
| benchmark_return | `benchmark_return_1d/3d/5d` (SPY) | outcome, same function and window |
| mfe / mae | `mfe_5d`, `mae_5d` (5-day window only) | outcome |
| entry/outcome price | not stored (always null) | gap |

## Methodology

- **Returns**: close of the first daily bar on/after the UTC date of `fired_at`
  to the close `h` bars later (`analytics.signal_outcomes.score_signal`).
- **Benchmark**: SPY scored by the same `score_signal` over the same dates; the
  existing track-record methodology. `excess = raw - SPY`. Rows matured before
  the columns existed are backfilled by the cron (`backfill_benchmark_returns`).
  Missing SPY is reported as coverage, never zero-filled.
- **MFE/MAE**: max High / min Low of the 5 bars after the entry bar vs the
  entry close. Only the 5-day window exists; 1/3-day fields are null.
- **Maturity** per horizon: `pending` (not computed), `unavailable` (computed,
  no price), `invalid` (computed at/before the observation), `matured`.
  Only `matured` enters a metric.
- **Certified**: matured AND passes the existing canonical eligibility rule
  (`opportunity_outcomes.is_eligible_opportunity_observation`: known score
  version, valid ticker/time/score 0-100).
- **Unit**: the cron freezes a ticker in many snapshots per day and all share
  one entry close, so the default `unit=signal_day` keeps the day's first
  observation (by time). `unit=observation` returns raw rows.
  `raw_observations` is always reported.

## Metric dictionary

| Metric | Formula | Null when |
|---|---|---|
| sample_size | records in the group, any maturity | never (0) |
| matured_count / pending_count / unavailable_count / invalid_count | per-horizon status counts | never (0) |
| distinct_days | distinct entry days among matured | never (0) |
| average_return / median_return | mean / median of matured raw_return | no matured |
| average_return_ci95 | mean ± 1.96·sd/√n | fewer than 30 matured |
| win_rate | matured with raw_return > 0 ÷ matured_count | no matured |
| win_rate_ci95 | Wilson 95% | no matured |
| average/median_benchmark_return | over matured with SPY | benchmark_count = 0 |
| average/median_excess_return | raw − SPY, over benchmark_count | benchmark_count = 0 |
| benchmark_beat_rate | excess > 0 ÷ benchmark_count | benchmark_count = 0 |
| average/median_mfe, _mae | over matured with MFE/MAE | horizon ≠ 5 or none present |
| evidence_quality | matured: <10 INSUFFICIENT, 10-29 LIMITED, 30-99 MODERATE, ≥100 STRONG (the existing calibration cut-offs) | never |

Intervals assume independent observations; same-day signals are correlated, so
treat them as optimistic (`distinct_days` shows how many independent days exist).

## Anti-cherry-picking

- Defaults are the complete dataset; every filter is explicit and echoed in
  `filters`. The default horizon (5) is pre-declared, not data-chosen.
- Every score bucket, horizon and setup is returned; setups are ordered by
  sample size, never by performance. Nothing is labelled "best".
- `/scores` includes a monotonicity check listing every inversion.
- Mixed score versions add a warning.

## Cache

| Key | TTL | Refresh / stale |
|---|---|---|
| `stamp` (row count, latest outcome and benchmark write) | 5 min | probe failure: last good dataset, `dataset.stale=true`; none → 503 |
| `("dataset", stamp)` | 6 h + 6 h stale-while-refresh | new stamp ⇒ new dataset |
| `("view", stamp, endpoint, params)` | 30 min (summary/scores/horizons/setups/timeseries), 10 min (symbol/query) | keyed by stamp, so new outcomes invalidate within 5 min |

Measured (synthetic, this sandbox): 10k rows build 0.2 s, any aggregate ≤ 0.12 s;
50k rows build 1.0 s, aggregates 0.27-0.5 s cold, cached afterwards. ~0.85 KB
per record resident. Past ~100k rows, move to a precomputed summary table.

## Observability

`hsf_api.outcomes` logs one JSON line per request (`outcomes_request`: endpoint,
cache hit/miss, ms, records, matured, pending, missing_benchmark,
missing_mfe_mae, stale) plus `outcomes_dataset_loaded`, `outcomes_load_failed`,
`outcomes_stamp_failed` and `outcomes_aggregation_failed`. No user data.

## Known gaps / leakage notes

- Entry/exit prices are not stored; outcomes can't be re-audited from the row alone.
- A postmarket snapshot enters at a close already printed at signal time (known, not future, data).
- `score_pending_signal_outcomes` stores nulls when bars are missing at scoring
  time; those rows stay `unavailable` (missing not at random).
- No point-in-time market regime is frozen on these rows, so there is no `/regimes`.
- Intraday Day Trader observations (`hsf_observations`, +5m..+60m) are a separate dataset, not included.
