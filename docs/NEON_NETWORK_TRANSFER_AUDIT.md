# Neon network transfer audit — 2026-10-09

## Scope and evidence

Audited `dev` at `77799c2c9ac63475a795d9926e56ca1ef4fac6c4`. The repository already contains commit `109898f` (#53): run-result caching by stamp, slim maturation records, and once-daily Stair-stepper analysis. Those are **baseline fixes**, not savings delivered by this PR.

No production PostgreSQL connection, backfill, deployment, data deletion, scoring change, or scheduler dispatch was performed. The full US universe, observation retention, first-write-wins outcomes, research gates and scoring definitions remain intact. Disposable localhost PostgreSQL 17 and synthetic fixtures were used for validation.

The owner's exhausted 5 GB allowance is the incident fact. Source inspection proves the transfer mechanisms below, but neither Neon billing history nor historical row/payload counts were available. **An exact attribution of the billed 5 GB cannot be established from code alone.** The ranking is an evidence-based priority for measurement, not measured percentages of the incident.

Read-only GitHub Actions metadata was also inspected. A sample of the latest 100 runs contains seven successful main-branch maturation dispatches from 14:40–17:40 UTC on October 9, consistent with a half-hour cadence; three successful main-branch scan dispatches at 16:35, 17:53 and 18:52, one scan on a verification branch at 17:08, and a successful main-branch health run at 17:05. This limited window is not a daily/monthly frequency measurement. External scheduler configuration and duplicate dispatch intent cannot be determined from it. Workflows execute `main`, which has commits beyond this audited `dev` baseline; do not assume development tests describe every production change.

## Ranked contributors and code locations

| Priority | Mechanism and evidence | Direction / status |
| --- | --- | --- |
| 1 | Saved scan payload re-downloads: `api/today.py::run_df`, `db/runs.py::load_run_results`. Before #53, short cache expiry repeatedly fetched whole `results_json`; the code explicitly identifies this as most egress. Current stamp cache keeps unchanged payloads longer. Every API replica has its own cache. `db/runs.py::load_runs_results` also fetches whole saved runs in bulk. | DB → client; historically documented major contributor, actual incident share unmeasured. |
| 2 | Historical research audits: `scripts/audit_research_cohorts.py::_load_live` reads up to 100,000 complete observations **and every outcome without a WHERE or LIMIT**. Forward readiness and research integrity reuse this loader. `scripts/audit_research_metadata.py::_load` separately downloads up to 100,000 full observations. `scripts/analyze_stair_stepper.py::write_report` reads up to 20,000 observations plus full outcomes. Before #53 this last analysis ran each maturation cycle; now only in the post-close slot. | DB → client; current historical growth risk. Entire outcome history is deliberately retained for integrity/orphan checks. |
| 3 | Broad analytics/readiness reloads: `db/research_datasets.py::fetch_readiness_rows` has a lower date bound but no row limit; `fetch_scan_index` reads the historical join index. `scripts/ml_readiness.py::load` reloaded both if the second query failed. `api/ml_readiness.py` caches 30 min/stale 2 h. `api/outcomes.py::_load_dataset` can fetch 100,000 rows through `db/signal_outcomes.py::fetch_outcome_rows`; it has dataset/stamp caching. | DB → client; redundant recovery read fixed here; full-data ML semantics preserved. |
| 4 | Shared OHLCV cache: `db/prices.py::get_price_data_snapshot` returns serialized full daily history per fresh symbol. `scan/engine.py::run_breakout_scan` previously re-upserted **all** returned frames, including cache hits. This also advanced cache freshness without a new load. Eligibility is admin-context/large-universe dependent, so do not assume all cron runs use this path. | Reads: DB → client. Redundant writes: client → DB, **not** the same as billed public egress. Cache-hit reuploads fixed here. |
| 5 | Maturation: `scripts/mature_observations.py::main` repeatedly reads the latest 5,000 slim observations and horizon sets. `db/hsf_observations.py::load_recent_observations` previously aggregated all historical outcomes even for this small read. | Repeated DB → client reads remain; limiting the outcome aggregate improves server work, not returned-byte volume. |
| 6 | Monitoring: `scripts/system_health.py::_load_observations` reads 20,000 full records before filtering; outcome health reads a date window. System health runs twice daily according to workflow comments and includes readiness. Observation health is manual, default 50,000 rows. | DB → client; sample and row counts required to quantify. |
| 7 | Realtime polling and reconnects: `billing_service/realtime_alerts.py`, `api/scan_jobs.py` heartbeat/status reads, `db/core.py::get_conn`. Due-alert SQL is filtered and existing tests cover it. `db/engine.py` already reuses a connection per thread. `db/core.py` previously tried psycopg2 immediately after any psycopg3 failure. Scheduled scans lacked a concurrency group. | Small reads/connection protocol overhead; secondary unless frequency/concurrency is high. Double-driver failure retry and scan overlap fixed here. |

A literal `SELECT *` is not automatically a full-table download. The explorer's stars mostly refer to bounded CTE rows or operate inside the server before aggregation (`db/observation_explorer.py`); blanket replacement would not address the main payloads. Alert-event reads were changed to explicit contract columns. No runtime SQLAlchemy engine, ORM session, or `read_sql` call was found in Python application sources; SQLAlchemy is a declared dependency, while runtime access uses psycopg3, psycopg2 and SQLite.

Other inspected paths include HSF stock/scans/watchlist/research/outcome APIs, prebreakout/AI-confidence training and recalibration jobs, ML v3 evaluation, manual rescoring, observation capture, bulk outcome saves, recovery ledger, BTC outcome logging and all Actions workflows. Training/evaluation intentionally need historical feature/label sets; trimming them without equivalence evidence would change research semantics. Model downloads and price-cache reads should be counted separately from provider market-data traffic.

## Implemented optimizations and controls

- Retain a successful readiness observation download across scan-index retries within the same fixed `[lo, hi]` window. Three attempts remain; delays now use bounded exponential backoff with jitter, and no sleep occurs after the final failure. No durable cache of mutable labels is introduced.
- Limit horizon aggregation to the selected observations. Keep one SQL query and existing outcome membership semantics; both SQLite and PostgreSQL results are validated.
- Make recent-observation selection stable with `(timestamp DESC, observation_id DESC)` ordering. Equal-timestamp rows previously had undefined order and the SQL rewrite exposed different limit subsets. This is a deliberate deterministic tie-break, not a change to scoring.
- Upload only frames returned by the price loader, not fresh Neon cache hits or snapshot-only frames. Do not refresh a cache-hit timestamp. Fetch missing members even when some frames are cached, and retain successful frames on a later fetch failure. Full-universe requests are not reduced.
- Use explicit alert-event columns while retaining every existing contract field, pagination filter and retry rule. Future schema columns cannot silently enlarge responses.
- Preserve the existing warm per-thread pool and bulk upsert behavior. Add bounded connection timeouts to the secondary helper, and only fall back to psycopg2 when psycopg3 is unavailable. Never echo a DSN-bearing connection exception. Standalone billing deployments retain an import-safe connection path.
- Serialize scheduled scan workflows with a shared concurrency group and `cancel-in-progress: false`, matching existing maturation/health controls. This protects against overlap; it does not prove or eliminate all duplicate dispatches.
- Instrument API requests, API scan jobs, API cache refreshes and the major DB-backed workflow CLI jobs. Keep original module arguments and exit codes, including on failure. Upload aggregate JSON traffic reports for 14 days. No SQL logging, row values, tickers, user IDs, tokens or credentials are emitted by telemetry.

The existing `(observation_id, horizon)` outcome key is the stable, idempotent completion checkpoint. This PR preserves it. A new high-water-mark-only maturation checkpoint would miss late data/failed symbols; a correct durable pending-work queue is a follow-up, not silently approximated here. Full-history audits also keep their integrity scope rather than dropping older outcomes to reduce bytes.

## Telemetry: meaning and operation

`db/traffic.py` installs driver-native cursor subclasses through connection factories. Query behavior, rows, transactions, generator-based `executemany` and default fetch sizes are preserved. Context-local, lock-protected counters propagate into API synchronous worker threads and isolate concurrent requests. Aggregate counts have no data values. With no active scope, cursor calls bypass measurement work.

| Field | Meaning |
| --- | --- |
| `calls` | Logical executions attempted; `executemany` counts each consumed parameter set, not one network round trip. |
| `rows_returned` | Buffered result rowcount reported after `execute`, where the driver exposes it; not affected-row counts for ordinary writes. |
| `rows_fetched` | Rows actually consumed through fetch/iteration. |
| `db_to_client_payload_bytes` | Approximate consumed row payload, UTF-8 scalar text and compact JSONB serialization. |
| `client_to_db_payload_bytes` | Approximate query representation plus parameter payload. Repeated logical executions may overestimate SQL bytes sent with prepared/pipelined queries. |
| `execute_ms`, `fetch_ms` | Client-observed execution/fetch time, including waits, not pure PostgreSQL CPU time. |
| `connections_opened`, `connections_reused` | Successful physical opens / warm checkouts in the instrumented paths; not total live server connections. |
| `cache_activity` | Separate hit/miss/stale counts for API lookups, local market frames and Neon price-cache symbols. Hit rate = hits / (hits + misses); show stale hits separately. |
| `elapsed_ms` | End-to-end scope/job elapsed time. |

API access logs attach metrics to existing request IDs and route templates. Slow-query warnings contain duration only. Background cache refreshes and scan jobs use separate scopes, so their traffic is not wrongly attributed to the request that queued them. A CLI can be measured with:

```bash
python -m scripts.db_traffic_job scripts.observation_health --limit 1000 --out artifacts/automation
```

This example is a production-capable read; it was **not** run during this audit. Offline benchmark:

```bash
python -m scripts.benchmark_db_transfer
```

Workflow reports are in `artifacts/db_traffic/*.json`; sum both directional fields separately across every report/process in a workflow, including failed attempts. API logs and workflow artifacts are separate sources. Retry-attempt/workflow reruns must be included in monthly totals. Counters with no scope do not accumulate globally.

| Configuration | Default | Effect |
| --- | --- | --- |
| `DB_TRAFFIC_WARN_BYTES` | 100,000,000 | Warn on estimated DB → client payload per scope. |
| `DB_TRAFFIC_WARN_UPLOAD_BYTES` | 100,000,000 | Warn separately on client → DB payload. |
| `DB_TRAFFIC_WARN_CALLS` | 10,000 | Warn on logical executions per scope. |
| `DB_QUERY_WARN_MS` | 2,000 | Warn on slow executions, no SQL text. |
| `DB_JOB_WARN_MS` | 900,000 | Warn on scope duration. |
| `DB_TRAFFIC_ENABLED` | 1 | Set 0 before connection creation to disable cursor instrumentation. Cache/scope summaries still work. |
| `API_SCAN_WORKERS` | 1, clamped 1–4 | Existing per-process API scan concurrency setting. |
| `AI_SCANNER_DB_POOL` | 1 | Existing warm per-thread connection reuse toggle. |
| `DB_CONNECT_TIMEOUT` | 10 s | Existing primary connection timeout. |

The modified workflows read warning thresholds from matching GitHub repository Variables. Set equivalent environment values on the API service. Warnings are non-fatal: they do not abort data collection or shorten the universe. Default thresholds are starter budgets, not empirically tuned baselines. Set per-job baselines after a representative week; infrastructure log alerting can route `db_traffic.budget_exceeded=true` and `db_query_slow` events to operators. No messages to operators were sent during this audit.

### Coverage limits and provider-only measurements

Application estimates omit TLS/authentication, PostgreSQL framing/metadata/command acknowledgements, result data buffered but never fetched, replication, exports, other clients and Neon accounting. JSONB formatting and binary encodings vary. They must not be used as invoice numbers. `rows_returned` is incomplete for pipelined multi-result/streaming operations. COPY/named server cursors/custom cursor factories are outside this instrumentation; none was found in the inspected runtime paths. Direct diagnostic connections and inline ML workflow programs are not automatically scoped by the CLI wrapper. Standalone `billing_service/` deployments do not have the scanner telemetry module and keep the existing lightweight path. Scope coverage should be expanded if those workloads become significant. Global live connections come from provider/server statistics, not these open counters.

Only Neon can establish billed public egress, project/branch/compute attribution, consumption history, traffic from other clients/replication and real before/after savings. Use project Overview/organization Projects usage and the paid Consumption API (`public_network_transfer_bytes`, `private_network_transfer_bytes`). The Billing page may omit transfer until overage begins. Hourly API history is available for seven days, daily for sixty days, monthly for a year according to the current documentation. Capture a baseline promptly.

Neon monitoring / `pg_stat_activity` can show live connections; `pg_stat_statements` shows calls/rows/execution times and resets, not network bytes. Do not export normalized SQL text to public logs as a substitute for telemetry: normalized statements can still contain identifying identifiers or some utility-statement literals. The existing manual query-stats workflow was not enabled or executed.

## Validation and measurements

See [the checked-in synthetic benchmark](audits/neon-transfer-benchmark.json). Its baseline reproduces the retry structure from audited `dev`; its optimized path calls the actual new readiness loader. No production query was executed.

| Workload | Before | After | Interpretation |
| --- | --- | --- | --- |
| 10,000 synthetic readiness rows; first scan-index query fails after rows downloaded | 4 calls; 30,000 fetched rows; 2,947,780 estimated inbound bytes | 3 calls; 20,000 fetched rows; 1,628,890 estimated inbound bytes | Equal loader output; one redundant observation read removed. About 44.7% less consumed payload **in this failure scenario**. Normal successful runs still need both reads. Query/parameter bytes and timing deliberately excluded. |
| 8,000 synthetic fresh cache-hit symbols × 250 bars; existing JSON serializer | 8,000 unnecessary upsert rows; 117,000,000 frame/symbol upload bytes | 0 unnecessary upsert rows; 0 upload bytes | Serialization-based upload model; cached read traffic remains. Not a billed-egress saving. Mixed-cache and all-hit engine tests preserve frames/results. |
| Recent observation horizon read | One query over whole-history aggregate | One query; aggregate restricted to selected IDs | Same result fields and byte volume for the same selected IDs. No network-saving claim; reduced historical aggregation scope. |
| Alert event reads, current schema | All current event columns | Same explicit current columns | Current result equivalence; prevents future accidental payload expansion. |

Real PostgreSQL 17 checks cover psycopg3/psycopg2 execute/fetch/default fetchmany/iteration/generator bulk calls, actual JSONB projection, context filters, missing contexts, stable timestamp ties, one-query horizon retrieval and unchanged observation/outcome counts. SQLite provides an independent old/new SQL comparison. Existing maturation report/outcome equivalence tests, point-in-time tests, ML evaluation tests and full-universe tests remain in the suite. No retention or historical rewrite was added.

Workflow validation parses all YAML, resolves wrapped modules, preserves arguments/status on failure and checks scheduling/concurrency conditions. Existing tests that asserted literal old CLI commands were updated for the wrapper. These checks do not claim that a post-change production workflow has run; executing a live workflow would be a deployment/production job action beyond this audit.

Final validation: **2,945 tests passed, 24 skipped, and 324 subtests passed** in 147.37 seconds with `HSF_TEST_POSTGRES=1`, including the disposable PostgreSQL checks. Repository-wide `ruff check .` and `git diff --check` passed. The final focused workflow/telemetry/maturation/BTC check passed all 53 tests. The environment uses Python 3.12; CI uses Python 3.13 and repository dependency constraints. Local installed dependency versions were not a fully locked production environment. CI remains the authoritative parity check.

## Illustrative monthly model, not a production forecast

Use decimal GB. Replace these assumed row counts/widths with telemetry and measured dispatch frequencies:

`estimated_monthly_egress = sum(runs_per_month × inbound_payload_per_run) + API cache misses + stamps + other clients`

| Assumed workload | Explicit sizing assumption | GB/month |
| --- | --- | ---: |
| Maturation | 22 trading days × 20 runs/day × 5,000 rows × 350 bytes | 0.770 |
| Once-daily Stair-stepper | 22 × (20,000 observations + 80,000 outcomes) × average 2,000 bytes | 4.400 |
| Metadata audit | 22 × 100,000 observations × 2,000 bytes | 4.400 |
| Forward readiness / full-history integrity | 22 × (100,000 observations × 2,000 + 400,000 outcomes × 350 bytes) | 7.480 |
| System health | 44 × (20,000 observations × 2,000 + 20,000 recent outcomes × 350 bytes) | 2.068 |
| ML readiness | 44 × (100,000 slim rows × 250 + 200,000 scan-index rows × 80 bytes) | 1.804 |
| Saved-run API | 100 changed/cold cache misses/day × 30 days × 500,000 bytes + 100,000 stamps × 64 bytes | 1.506 |
| OHLCV cache read, if enabled/fresh on every scan | 22 × 6 scans/day × 8,000 symbols × fixture frame 14,619 bytes | 0–15.438 |

Subtotal **22.43–37.87 GB/month** under these invented workload sizes. Adding a 20% planning reserve gives **26.91–45.44 GB/month**. The reserve is not a validated protocol multiplier; unmeasured clients or larger history can exceed it. Some audit limits may be much larger than actual production row counts, so this model is for capacity sizing, not incident attribution. Extra manual audits/training, API replicas, missed cache invalidations and reruns add traffic. Price uploads are tracked separately and are not added to public-egress estimates.

For scale: a 500 KB result downloaded once a minute for eight hours is 240 MB/day for one active cache stream. Twenty cycles downloading 100,000 historical rows averaging 2 KB is 4 GB/day. These mechanisms can exhaust 5 GB without a large table. The once-daily/cache baseline fixes materially change this model, but their actual savings need Neon before/after consumption history.

## Remaining risks and follow-ups

1. Full-history audit outcomes remain unbounded and grow monthly. Do not filter them by only the selected observation IDs: that could suppress orphan/conflict evidence. A server-side integrity aggregate or validated partitioned/keyset audit is the next optimization.
2. Latest-5,000 maturation selection can keep older pending observations outside the active window. Stable ordering improves reproducibility but is not a backlog queue. Use durable pending pairs with bounded pages and revisit failed/late data; preserve all historical outcomes.
3. Large `attach_outcomes="full"` IN lists can approach parameter limits; scheduled Stair-stepper's 20,000-row limit is below PostgreSQL's limit, but larger callers should use a validated batched/array path. This PR does not silently change research scope.
4. Per-thread warm pools do not impose a cross-process connection ceiling. Keep API scan concurrency at 1 initially; account for API replicas/sync threads/parallel workflows. Use Neon's pooled connection endpoint where transaction pooling is compatible; pooling reduces connection overhead, not result-payload size. `db/core` remains an independent connection path to preserve tuple/context-manager expectations.
5. Workflow concurrency prevents overlaps, not sequential duplicate work. GitHub can replace an older pending run; monitor completion/backlog. Keep one authoritative dispatch configuration. External cron configuration is not versioned here; verify the still-scheduled forward-readiness workflow is not also dispatched daily externally before removing a trigger. Different workflow families can still overlap.
6. Readiness rows/outcomes are mutable during maturation. Persisting an incremental cache needs updates/deletes/late-arrival invalidation, fixed time windows and reconciliation, not only `id > checkpoint`. Training joins and integrity comparisons must be proven equivalent first.
7. Run cache uses an existing timestamp/count stamp; multi-replica cold starts and same-stamp rewrites remain risks. Avoid longer price freshness windows as a cost shortcut. Snapshot inputs and market data remain point-in-time sensitive.
8. Estimates currently do not cover every standalone/inline/manual workload. Confirm telemetry coverage with one representative week before treating monthly extrapolation as operationally reliable.

## Neon Launch cost controls

Official pricing checked on 2026-10-09 lists **500 GB/project/month public egress included, then $0.10/GB**, versus Free's 5 GB. Public transfer counts outbound data; private transfer is a different, bidirectional Scale-plan product. Reconfirm the project's actual contract/region/invoice. Launch is headroom, not a substitute for query control.

- Capture hourly/daily consumption for comparable workload windows before and after release; normalize by universe size, observation count, API cache misses and workflow attempts. Keep this baseline with release SHAs. Do not claim savings from local estimates alone.
- Set organization spending notifications (current docs describe 80%/100% thresholds). Add an operational project-egress alert well below 500 GB, and alert on an unusual daily slope relative to the measured week. No account settings were changed here.
- Keep scale-to-zero on for non-production computes; monitor recurring polling that keeps compute awake. Set appropriate autoscaling bounds after observing latency and queue lengths, rather than cutting the market universe.
- Track compute, data storage, restore history, branches and egress separately. Launch currently lists $0.106/CU-hour, $0.35/GB-month database storage and $0.20/GB-month restore history. Keep required observation retention; shortening database restore history is a separate owner decision and not deletion of observation records.
- Prefer pool reuse, same-job reusable data, correct result invalidation and bounded pending-work retrieval. Do not create replicas or frequent database exports as a cost workaround: they can also contribute outbound traffic.
- A 600 GB public-egress month would imply approximately $10 of transfer overage under the listed rate, excluding compute/storage/history and other charges. Confirm allowance aggregation before budgeting across projects.

Sources: [Neon pricing](https://neon.com/pricing), [network transfer definition/monitoring](https://neon.com/docs/introduction/network-transfer), [Monitoring dashboard](https://neon.com/docs/introduction/monitoring-page), [spending notifications](https://neon.com/docs/introduction/spending-notifications). Pricing/retention facts can change; this report is dated.
