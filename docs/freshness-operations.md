# Data freshness and processing health

The public Today, Scanner and Brief views reuse saved full-US-market scans. A
shared calendar rule evaluates the last due in-session scan, with the existing
45-minute grace and 10-minute early tolerance. Weekends and market holidays do
not age a successfully completed last session into stale data. Early closes
exclude later scheduled slots. Outside the bundled calendar's coverage the
state is partial. API timestamps carry UTC offsets; public displays use Eastern
time (EDT/EST in Streamlit, ET in web).

`freshness` is additive on Market, LatestScan and Brief; existing `scan_at`,
`latest_scan_at`, `stale`, pagination and entitlements retain their contracts.
The generated OpenAPI and TypeScript clients include the new schema. Older API
responses without this field remain supported by the web UI.

## Public example

An illustrative Friday scan viewed on Saturday:

```json
{
  "state": "partial",
  "scan_state": "fresh",
  "market_data_state": "unavailable",
  "last_successful_scan_at": "2026-10-09T20:25:00+00:00",
  "scan_completed_at": null,
  "market_data_at": null,
  "expected_scan_at": "2026-10-09T19:35:00+00:00",
  "calendar_covered": true,
  "checked_at": "2026-10-10T17:00:00+00:00",
  "timestamp_basis": "saved_scan",
  "market_data_note": "Underlying market-data timestamp unavailable"
}
```

The saved scan timestamp is the latest successful persisted cron/US_MARKET
snapshot, not an independently recorded job completion time. Historical
snapshots do not contain a reliable underlying quote/bar watermark or completion
time. These fields remain null rather than inferring freshness from a newer
scan. The Brief now selects the matching market scan instead of an unrelated
latest daily snapshot and no longer calls it Live merely because the market is
open. A known old market-data timestamp makes the result stale even when the
scan itself is fresh.

## Admin API and dashboard

`GET /v1/admin/operations` requires the existing bearer authentication and
`is_admin` authorization: unauthenticated calls receive 401; non-admins 403.
The existing Streamlit Admin Analytics → System Health page renders the same
allowlisted summary before any product/research-history loading.

Example response excerpts (synthetic values; omitted fields are documented by
the implementation):

```json
{
  "available": true,
  "generated_at": "2026-10-09T21:00:00+00:00",
  "stale": false,
  "refresh_failed": false,
  "workflows": [{
    "workflow": "scheduled-scans.yml",
    "label": "Scheduled market scan",
    "status": "HEALTHY",
    "last_run": null,
    "last_success": "2026-10-09T21:00:00+00:00",
    "last_failure": null,
    "consecutive_failures": 0
  }],
  "maturation": {
    "last_success_at": "2026-10-09T21:00:00+00:00",
    "pending_observations": 12,
    "oldest_pending_age_minutes": 35,
    "scope": "ready observations in latest maturation report selection; not all pending or global backlog"
  },
  "neon_usage": {
    "available": false,
    "public_network_transfer_bytes": null,
    "compute_cu_hours": null,
    "storage_bytes": null,
    "note": "Provider usage unavailable; application estimates are not billed usage"
  }
}
```

`database_traffic.measurement` is `application_payload_estimate`. Its
`sample_totals` and per-workflow `metrics` expose calls, errors, fetched rows,
database-to-client and client-to-database payload bytes, connection/cache counts
and durations. Missing values are null, never fabricated zero. Totals sum the
available latest completed instrumented run per workflow, **not** all runs in a
day or month. The existing System Health job retrieves at most four existing
traffic artifacts, sanitizes their numeric fields and stores them in its usual
health report. No new scheduler or production job dispatch is introduced.

Operational responses whitelist known workflow names, status enums, timestamps
and numeric aggregates. They exclude credentials, SQL, user identifiers, raw
research records and detailed exception strings.

## Bounded access and recovery

Public freshness is derived from already loaded run metadata; it adds no database
query. The operations summary reads only the newest health record (`LIMIT 1`)
with a shared, single-flight, 120-second application cache. The public system
status and admin views reuse that snapshot. Streamlit also retains its existing
300-second presentation cache. No request calls GitHub or Neon, performs a
full-history download, or aggregates outcome history. `api.today.clear_cache()`
invalidates the snapshot and retained recovery state; ordinary new reports
appear after the TTL. Caches are per process, not shared across replicas.

Failures are cached for the same 120-second retry interval. A last good health
snapshot remains usable for up to one hour, explicitly marked stale and
refresh-failed; after that it becomes unavailable. Public system status does
not call a retained failed refresh Operational. Existing web `useApi` preserves
loaded results when refreshes fail; this behavior now has a regression test.
No new polling is added.

## Limitations and validation

- A global pending-outcome count and globally oldest pending outcome cannot be
  computed safely from current bounded report sources. The existing maturation
  selection count/age are shown with their scope; a full-history aggregate is
  deliberately avoided.
- Older health reports lack traffic fields. Until the normal System Health job
  collects new artifacts, estimates are unavailable. Missing or expired artifacts
  remain unavailable. Coverage is the four instrumented health-plane workflows.
- Monthly transfer and actual Neon billed transfer, compute and storage require
  provider metrics. No provider credentials are consumed by this request path;
  provider usage is separately unavailable, and no actual savings are claimed.
- Recording trustworthy market-data and completion timestamps requires a future
  ingestion provenance change; this change does not backfill history.
- No scoring, model evaluation, prediction provenance, market universe,
  observation storage or outcome processing semantics are modified.

Full Python suite: 2,979 passed, 28 skipped, 325 subtests passed (one existing
Starlette/httpx deprecation warning). Clean development-dependency environment:
9 passed, 2 skipped for the new test module. Web: typecheck, lint, 175 tests
passed; the additional refresh-failure test passed in the focused four-test run.
Final API/model-focused run: 68 passed, 9 skipped, 35 subtests passed.
Production build passed and production dependency audit found zero
vulnerabilities. Ruff and whitespace checks passed. Integration tests requiring
opt-in PostgreSQL were not run against production.

The [HTML preview](previews/freshness-operations.html) and
[screenshot](previews/freshness-operations.png) use synthetic fixtures rendered
by the actual Streamlit banner/admin HTML helpers. They are not production
measurements. Regenerate with `python -m scripts.preview_operations_ui`.
