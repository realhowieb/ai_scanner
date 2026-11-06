# Label, diagnostics and billing review — November 6, 2026

Scope: the production-source repository after promotion #370. This is a source
and isolated-test review. No live charges, webhook replays, training, registration,
production model updates or dataset finalization were performed.

## Findings and changes

| Priority | Result | Evidence |
|---|---|---|
| Economic label semantics | Fixed a silent switch: merely having `Return_5D` selected terminal-return labels instead of the stated first-barrier objective. The default now always requires a complete finite OHLC window. Missing data stays unlabeled. | `tests/test_economic_label_contract.py`: cached +6% after a prior -2% stop remains negative, with and without a cached-return column. |
| Cross-symbol helper | Corrected `ml_prebreakout.add_future_breakout_label` to look ahead inside the normalized symbol group. | Regression proves BBB's breakout cannot label AAA, while AAA's own later breakout can. All repository call sites inspected: active economic training uses future-quality-setup/path labels; the audit keeps an explicitly named historical bug reproduction. |
| Diagnostics | Public `/readyz` retains only `ok`, `database`, `stale`, `build`; strict failures also use this allowlist. Memory, cache, timing and dependency detail moved to admin-only `/v1/admin/readiness`. Billing `/health` returns only `ok`; `/debug/status` requires its enable flag plus a single-use authenticated billing token and an active admin account. | Tests for anonymous 401, ordinary-user 403, active/admin role checks, authenticated detail and public failure redaction. Build identity remains public for deployment verification. |
| Tutorial | Replaced the retired UI guide with the current web navigation, custom scan/RSI/MACD selector, stock checks, watchlists, alerts and billing return flow. | `APP_TUTORIAL.md`, checked against current `AppShell`, `ScannerView`, `CustomScanView` and stock-page components. |
| Model and billing behavior | Inspected current callers and exercised training/evaluation, signatures, replay, reconciliation, ownership and entitlements using isolated tests. | Suites listed below. Production state was not exercised with fake events or changed. |

## Economic target contract

`path_first_hit_v1`: entry close; future bars only; target high >= +4% before
stop low <= -2%; when both touch in one daily bar, stop wins conservatively.
Every close/high/low in the required horizon must be finite and positive;
missing closes are not dropped to extend the horizon. Unsorted/duplicate bar
indexes are refused. Extrema cover the full horizon, even after the first barrier.
Daily bars cannot establish intrabar ordering; the same-bar policy is an explicit
approximation, not a claim about the actual sequence.

A terminal-close-return experiment must explicitly request
`target_kind="terminal_return"`. Its recorded definition differs from path labels;
column presence cannot select the target. `force_path=True` cannot be combined
with a terminal-return objective. The standard AI Confidence and PreBreakout
builders continue to request the path target. Future artifacts record updated
rules; existing artifacts and historical reports are not retroactively relabeled.

The research v4 primary `F_clean_path_5d` remains distinct: MFE >= +4% and
MAE > -2% over the full window, independent of barrier order. It is intentionally
not rewritten as the legacy first-hit target.

## Training and evaluation paths

- `scripts/train_ai_confidence_model.py` uses economic labels, chronological
  splitting, writes local artifacts, and uploads only with `upload_db`/
  `--upload-db`. Its legacy calibration fits on validation predictions, so its
  calibration buckets are fit evidence, not an independent calibrated test.
- `train_prebreakout_model_run17`, used by the current manual PreBreakout
  workflow, builds a diagnostic target/ranking report and explicitly returns
  `NO_PROMOTION`; it does not replace a production model.
- The older `train_prebreakout_model` remains a compatibility path with rollback
  and conditional active-model writes. It is not the current workflow entry
  point and was not executed. Do not invoke it as a supposedly read-only audit.
- The v4 workflow separates preflight, development evaluation, final-test claims,
  registration and lifecycle transitions. Fingerprint verification, purged folds
  and final-report reuse remain enforced. Development evaluation is the chosen
  next comparison; this review does not approve production training or a switch.
- Stricter path requirements may reduce eligible legacy training rows and change
  results versus historical mixed-label runs. Do not compare their AUCs as though
  their targets were identical. Existing production predictions are unchanged.

## Billing and account boundaries

`BillingEngine.webhook` verifies the untouched request bytes with the Stripe SDK
and configured signing secret before account resolution/writes. Recognized event
mode must match the key mode. Replay checks happen before work and again under
the account transaction; the processed-event marker commits with the plan update.
The handler reconciles current Stripe state under the account lock rather than
trusting a late event snapshot to restore or remove access. Errors remain
retryable and do not mark a failed account update as processed.

Checkout and portal handlers resolve the authenticated account server-side.
Customer-link mismatch is refused; plan and price mapping are server-controlled.
Unknown prices, incomplete payments, stale cancellation events and repeated
checkout retries have dedicated tests. API scan ownership, watchlist ownership,
row limits, premium features and admin overrides are enforced server-side, not
by hiding controls alone.

Tests use fake Stripe calls: they validate handler behavior and SDK verification
arguments, not a real charge or external Stripe configuration. Transaction rollback
and concurrent replay checks run in the CI `billing-postgres` job against its
throwaway PostgreSQL service. Local tests requiring an explicit test database
remain skipped; no production database was substituted.

## Monitoring configuration

System Health now authenticates to `/v1/admin/readiness`. Repository secret-name
inspection found neither `ACC_ADMIN_EMAIL` nor `ACC_ADMIN_PASSWORD` configured.
Until an active admin test account's credentials are added in GitHub Actions
secrets, the report emits `API_DIAGNOSTICS_UNAVAILABLE`; it must not silently
report healthy memory or expose details publicly. Other health collection
continues. This is a deployment configuration dependency, not a reason to leave
diagnostics public. No password values were retrieved or written by this review.

## Verification

Targeted model suites include economic contracts, persistence, legacy audit,
AI Confidence training, calibration, v4 preflight and evaluation. Billing/access
suites include billing API, reliability, service, preflight, token authentication,
account isolation, entitlement matrix, custom scans, system health and deployment
build checks. Generated OpenAPI/client types, frontend typecheck, Ruff and diff
whitespace checks accompany the change. CI supplies isolated dependency and real
PostgreSQL checks before merge.
