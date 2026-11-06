# Historical-performance audit

Source base: `1fb35f5ebad097fe3ea5d4faad95a264a40a84b8`. These are verified code
findings, not claims of deployed production impact. No production configuration,
data, paid training or live billing action was changed or invoked.

| Finding | Severity/status | Source and resolution |
|---|---|---|
| Availability timestamp discarded; earlier price can precede signal | High, confirmed | `analytics/track_record.py::_eligible_snapshots`, `_forward_return`, `compute_track_record`. Preserve timestamp, select exact session open after availability and persist provenance. |
| Silent Open-to-Close fallback | High, confirmed; open mode is used | `_forward_return`; `scheduler/cron_runner.py::_refresh_signal_leaderboard` computes both modes. Refuse missing/invalid field; align both legs. |
| Ambiguous benchmark-relative aliases | Medium, partly already addressed | Compute/storage aliases mean excess/outperformance; `api/history.py` and `TrackRecordView.tsx` already label vs-SPY/Beat SPY. Add absolute and explicit rates without reinterpreting aliases. |
| Missing-data exclusions vanish | High, confirmed | Track compute previously skipped picks or returned no summary without SPY. Shared evaluator reports selected/evaluated/excluded, reasons and coverage even with zero evaluated picks. |
| Observations mistaken for independent trades | Medium, confirmed | Daily snapshot dedup does not remove adjacent ticker selections or overlapping horizons. Expose ticker/date counts and limitations, not a simulator or naive intervals. |
| Historical prediction protection | Already present; retained | `_ranked_symbols` uses saved provenance/probabilities and never current scoring. Keep regression; sibling leaderboard now shares PreBreakout selection. |
| Post-close snapshots in scheduled production | Uncertain | Cron snapshot promotion has regular-session guards; forced runs, save latency and retained history may differ. Source review alone cannot establish actual historical exposure. |

`db/runs.py::save_daily_snapshot` updates payload and `created_at` together. Its
write instant is conservative availability; overwritten originals cannot be
reconstructed. Latest snapshot per exchange date, score epoch and weekend
protections remain; explicit exchange holidays improve the prior weekday-only
filter. Timestamp ties now use ID. Pro checks in `api/routes/history.py` and
account/billing boundaries are untouched.

Implementation: shared pure `analytics/track_method.py`, complete timestamps,
aligned structured outcomes/exclusions, versioned storage/caches, API types,
ranking table/CSV, sibling leaderboard, morning digest and scheduler persistence.
The study explicitly restricts source SQL to scheduler-owned public snapshots;
private saved scans cannot appear in per-pick provenance. Overall Outcome
Intelligence is separate and unchanged. See [methodology](TRACK_RECORD_METHODOLOGY.md)
for selection details, versioning, interpretation and operational limitations.

## Actual verification

- `.venv/bin/python -m pytest tests/test_track_method.py tests/test_track_record.py tests/test_signal_leaderboard.py tests/test_api_paid_features.py tests/test_train_ai_confidence_model.py tests/test_prebreakout_model_persistence.py tests/test_morning_digest.py tests/test_scheduler_compat.py -q`: **201 passed, 1 skipped**, plus four subtests. The skip requires `HSF_TEST_PG_URL` pointing to a throwaway PostgreSQL database; none was configured. No production database was substituted.
- In `web/`: `npm run api:types`, `npm run typecheck`, and `npm test -- tests/account.test.tsx tests/premium-value.test.tsx tests/research-polish.test.tsx`: **49 passed**. The test DOM prints its existing navigation-not-implemented notice; it did not fail a test.
- `npx eslint src/features/TrackRecordView.tsx tests/account.test.tsx tests/premium-value.test.tsx tests/research-polish.test.tsx tests/fixtures.ts`: passed.
- Ruff on changed Python implementation/tests and `git diff --check`: passed.
- OpenAPI exported locally with `scripts/export_openapi.py`; no live endpoint was queried.

No production deployment, schema migration, production-data recomputation, paid
job or live billing action was run. The initial local run covered SQL persistence with mocks and namespace/payload
regressions. The merge follow-up added `tests/test_track_record_postgres.py`: both
legacy-schema upgrade/summary/daily isolation and leaderboard JSONB/provenance
round-trip tests passed against the throwaway CI PostgreSQL service. The complete
isolated PostgreSQL job passed 30 tests in [run 42313090065](https://github.com/realhowieb/ai_scanner/actions/runs/42313090065). Provider calls in return
regressions are mocked. Test numbers are validation counts, not financial results.

Users should see **Avg stock**, **Median stock**, **Avg/Median vs SPY**,
**Positive stock return**, **Beat SPY**, **Evaluated / selected**, **Coverage**,
**Unique tickers**, and **Signal dates**, with expandable exclusions. A -2% stock
return alongside +3 percentage points vs SPY is displayed as a loss and benchmark
outperformance together. Entry policy and methodological limitations remain
visible, and old-method data cannot populate the new ranking study.
