# Cohort audit outcome transfer optimization

## Finding and boundary

`scripts/audit_research_cohorts.py::_load_live` selected at most 100,000 recent
observations but downloaded every historical outcome. The standalone cohort
report only uses `outcomes_by_id.get(observation_id)` for those observations, so
unmatched records added transfer without affecting its report.

The same loader is used by effectiveness, forward-readiness, and maturation
parity. In particular, `analytics/signal_evidence.py::readiness` counts orphan
and conflicting outcome records across all loaded outcomes;
`analytics/forward_readiness.py::monitor` checks unmatched forward outcomes.
Globally filtering the shared loader would hide evidence. Its default remains
the full historical read, including true orphans and outcomes outside the selected
observation window. No global integrity optimization is claimed here.

## Change

Only the standalone cohort CLI opts into `selected_outcomes_only=True`.
PostgreSQL uses `observation_id = ANY(%s)` with one bound array; SQLite uses one
bound JSON array with `json_each`. The result retains complete outcome records,
all horizons and statuses for selected IDs, and existing observation selection.
There is no per-symbol query or per-ID bind parameter. An empty selection skips
the outcome connection/read. Other callers and snapshot exports keep the old
full-history contract.

Every run rereads current outcomes. No persistent checkpoint/cache is introduced,
so late and mutable labels remain visible. No observations or outcomes are
deleted or rewritten; scoring and ML evaluation definitions are unchanged.
Existing non-fatal database fallback behavior is retained.

## Local measurement

See [aggregate benchmark](audits/cohort-transfer-benchmark.json). Reproduce using
a disposable PostgreSQL 17 server on localhost:55432:

```sh
HSF_TEST_POSTGRES=1 python -m pytest tests/test_db_traffic_postgres.py \
  -k cohort_selected -q -s
```

The synthetic fixture stores 220 observations and 441 outcomes (including a true
orphan) and selects 10 observations with 40 outcomes. Measurements cover the
outcome read only; observation loading and seeding are excluded.

| Measurement | Before | After |
| --- | ---: | ---: |
| Queries | 1 | 1 |
| Fetched outcome rows | 441 | 40 |
| Estimated DB → client payload bytes | 81,767 | 7,390 |
| Estimated client → DB payload bytes | 59 | 140 |
| Local execution time, single sample | 1.340 ms | 1.295 ms |

The fixture has approximately 91% fewer estimated inbound outcome bytes. This
is **not actual Neon billed savings** or a production forecast. Savings depend
on the fraction of outcomes belonging to selected observations. If all stored
outcomes belong to the selected window, inbound bytes do not decrease and the
ID array increases upload bytes. A large array may affect planner/latency cost;
monitor the existing job telemetry before tuning. Single local timings are not
a statistically controlled speed comparison.

Cohort reports match excluding `generated_at`; storage counts are unchanged.
SQLite fixtures prove preserved point-in-time violations, complete cohort report
equivalence, and shared-loader orphan/conflict evidence. The PostgreSQL test
checks actual array execution and payload counts. Regression tests also cover
empty selections, 70,000 IDs in one bound parameter, and the CLI-only opt-in.

Final validation: 2,972 passed, 24 skipped, and 324 subtests passed in 137.37
seconds with disposable PostgreSQL enabled; 106 focused audit tests passed.
The five new loader tests also pass with only requirements-dev.txt installed.
Repository-wide Ruff and diff checks passed. Local Python is 3.12; PR CI verifies
the repository's Python 3.13 environment.

Remaining full-history audit transfers need a separate redesign with equivalent
global orphan/conflict detection. Observation records remain full because the
cohort duplicate/conflict check hashes the entire record. SQL outcome ordering
was never contractual; illustrative example order can vary with query plans,
while population, counts, gates, and labels remain unchanged.
