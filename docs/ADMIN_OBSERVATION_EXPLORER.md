# Admin Observation Explorer

The Admin Observation Explorer is a read-only research observability surface for tracing an aggregate result back to the immutable observations that produced it.

## Data inventory

| Field | Availability | Source |
|---|---|---|
| Observation ID, symbol, timestamp, context | All observations | `hsf_observations` columns and record |
| Candidate / Near Miss / Control | Scanner research | `research_cohort` or `market_context.research_cohort` |
| Scan ID, session | Scanner research when captured | `market_context` / `research_metadata` |
| Rank at observation | Metadata-era records | `research_metadata.rank_at_observation` |
| Signals | When captured | `scanners[].name` |
| Forward return, MFE, MAE | Matured outcomes | `hsf_observation_outcomes.record` |
| Research epoch | Derived from immutable registered timestamp boundaries | `analytics.forward_readiness` |
| Control design | Run 59 / Run 59B controls | `market_context.control_design` |
| Canonical HSF Score | Not consistently persisted | Displayed only when present; no scanner score is substituted |

Scanner research and `day_trader:stair_stepper` observations are separate datasets. Scanner outcomes use the horizons actually stored by the scheduled research pipeline. Stair-Stepper views expose only its stored window/outcome values.

## Query behavior

- The Explorer checks Admin authorization before issuing queries.
- Filters are parameterized and pushed into PostgreSQL.
- Counts, timelines, cohort comparisons, integrity checks, histograms, and score buckets are database aggregations.
- Observation records use bounded server-side pagination, newest first.
- CSV export is limited to 5,000 rows and reports truncation.
- Pending or unavailable outcomes are never converted to zero returns.
- Mixed epochs are labeled diagnostic and cohort-effectiveness comparison is disabled.

## Run 59B integrity

The current epoch begins at the registered Run 59B boundary and uses `run59b_liquidity_matched_disjoint_v2`. The Explorer computes scan-local symbol overlap between Candidate, Near Miss, and Control cohorts. It reports an error when any pair overlaps; it never changes stored cohort membership.

## Indexes

The established observation schema initialization maintains:

- `idx_hsf_obs_symbol_ts (symbol, timestamp)`
- `idx_hsf_obs_timestamp (timestamp DESC)`
- `idx_hsf_obs_context_timestamp (context, timestamp DESC)`
- `idx_hsf_outcomes_horizon_observation (horizon, observation_id)`

No DDL runs from the Explorer page.

## Known data limitations

- Canonical point-in-time HSF Score is absent from many historical observations, so score filtering and score/outcome views correctly remain unavailable for those selections.
- Older observations may not contain scan ID, rank, session, or control-design metadata.
- Epoch assignment is timestamp-derived because historical records do not all persist an explicit epoch identifier.

The highest-value next observability improvement is to persist the canonical point-in-time HSF Score and score version on every new observation. That would unlock complete score calibration and score-versus-outcome analysis without reconstructing or substituting any value.
