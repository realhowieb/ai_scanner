# Neon transfer controls: promotion validation — 2026-10-10

PR #62 is merged into `dev` at `70c4fec`. Validation combined that revision with
`main@4e48049` (including PRs #59–61) without conflicts. This preserves the newer
production provenance and health fixes. Promotion changes are the transfer
controls, development dependencies, tests, and validation artifacts; ML scoring
definitions and historical records are unchanged.

## Measured local workloads

[Aggregate measurements](audits/neon-promotion-workloads.json) are reproducible
with `HSF_TEST_POSTGRES=1 python -m scripts.validate_neon_promotion` against a
disposable PostgreSQL 17 instance on **localhost:55432**. The script never reads
application database URLs and refuses to run without the explicit test opt-in.
Fixtures contain 1,100 observations and 2,200 outcomes. The temporary schema is
removed after validation; production is never contacted.

| Workload | Before | After | Equivalence and interpretation |
| --- | --- | --- | --- |
| Latest 100 observations with horizon membership | 1 query, 100 rows, 35,666 inbound estimated bytes | Same | Exact output equality and unchanged history counts. Aggregation now visits selected IDs. |
| Observation execution time, one local sample | 7.585 ms | 3.172 ms | Not a statistical comparison; cache/order effects and local hardware apply. No production speed claim. |
| Same read with slim projection | — | 33,306 inbound estimated bytes | Slim projection predates PR #62; existing field contract preserved. |
| Price-cache lookup, one hit and one miss | — | 1 query, 185 inbound / 192 outbound estimated bytes | Frames equal after JSON round-trip; correct missing symbol and family counters. |
| API sync endpoint through real request middleware | — | 1 query, 1 row, 1 inbound / 15 outbound estimated bytes, 1 connection open | Real localhost driver and request context propagation; endpoint is validation-only, not added to production. |
| Readiness retry after failed scan-index fetch | 4 calls / 2,947,780 inbound estimated bytes | 3 calls / 1,628,890 bytes | Synthetic recovery fixture; equal outputs. Normal successful path remains two reads. |
| Fresh price cache, 8,000 symbols × 250 bars | 117,000,000 modeled upload bytes | 0 unnecessary upload bytes | Serialized model plus scan-engine mixed/all-hit tests; reads still occur. |

Connection setup used for fixture seeding is outside measured scopes. Numbers
are application payload estimates, not PostgreSQL wire bytes or Neon billing.
Scan-engine market fetches and readiness data boundaries remain mocked in the
deterministic tests; no live market jobs or production workflows were dispatched.

The combined-code suite passed **2,966 tests, 24 skipped, 324 subtests** in
146.56 seconds, including opt-in PostgreSQL checks. The focused database/cache/
scan/maturation suite passed 47 tests. Repository-wide Ruff passed. Local Python
is 3.12; GitHub CI on the promotion PR remains the Python 3.13 check.

## Alert review

Verified that an intentionally tiny inbound budget emits an aggregate WARNING
with `budget_exceeded=true`. Existing defaults remain 100 MB inbound, 100 MB
outbound, 10,000 calls, 2 seconds per query, and 15 minutes per scope. These are
starter thresholds. This small synthetic workload cannot justify production
threshold tuning, so repository Variables and production API settings were not
changed. No operator messages were sent.

After release, collect at least one representative week per workflow and API
route, including cache misses and retries. Set each job's inbound/outbound/call
budget above its observed normal range (initial recommendation: twice its p95),
then investigate warnings without aborting collection. Account for payload
growth and sample counts. API scopes currently share defaults; route-specific
budgets require a later implementation, rather than reusing a scan-wide estimate
as an API limit. Slow-query warnings and job durations are separate signals.

## Provider baseline and release gate

**Provider baseline unavailable:** this cloud runtime has no configured Neon
credential, runtime binding, or Neon connector. No provider usage was fabricated.
Before deployment, record Neon project public/private transfer, compute CU-hours,
storage and restore-history usage, time window, release SHA, dataset sizes,
workflow attempts, and API cache activity. Keep organization/project identifiers
and credential material out of public artifacts. Neon consumption history, not
these fixtures, is required to establish actual savings.

The promotion PR is reviewable without this baseline, but deployment and a
before/after savings claim remain separate gates. No production deployment,
backfill, data deletion, scheduler dispatch, or Neon setting change was performed.
Review CI and obtain deployment approval before enabling the changes in production.
After release, compare equivalent provider windows normalized by work performed.

Unbounded full-history integrity reads remain the next optimization target.
Batching must preserve orphan/conflict evidence and detect mutable/late outcomes;
an observation-only checkpoint would change audit semantics. That redesign is
outside this promotion and has not been silently introduced.
