# ML v3 Post-Deployment Provenance Verification

## Initial findings (before fixes)

On inspection, latest main is `e62b229500ccb4873fab1c26d4f08d2bd2e328c1`.
Latest successful Scheduled Market Scans execution is `37960183886`, created
2026-10-09 16:35:13 UTC, running prior SHA
`4866e642ff4f06fba731ff5ce71c3652cc1c54b1`.
No post-merge scheduled execution was available at the first check.
Execution status: **WAITING_FOR_PROSPECTIVE_RUN**, not deployment verified.

Source findings:
1. `ui.results.render_results` permits ModelProvenance into customer tables,
   raw previews and CSV exports; it contains internal AI probabilities and inputs.
2. `cron_runner._results_to_json` uses pandas default numeric precision.
   Nested calibration snapshots round while their hashes retain original
   precision. Final feature vectors can also lose precision.
3. Scheduled execution calls `scan.engine.run_breakout_scan` directly, not
   `scan.execution.run_manual_scan_execution`. The engine does not invoke
   AI-confidence inference. PreBreakout is explicitly invoked afterward.
   Therefore scheduled AI provenance is unavailable by architecture, not a
   serialization failure. Adding AI inference would change scheduled behavior
   and is intentionally outside this verification.

Narrow proposed fixes: remove internal columns from the rendering copy only;
serialize provenance as a lossless JSON string inside the DataFrame (existing
models_from_row already accepts strings), then decode at observation/freezing
boundaries. No inference or scoring change is needed.

## Production evidence

Read-only Diagnostics execution **37962355033** succeeded at audit SHA
`a87851c7727150119faf373a1be2dfc0d99ea2ae`. It enforced a repeatable-read,
read-only transaction and 30-second statement timeout, with bounded reads and
no model loading, provider calls, schema setup, inference or mutations.
At **2026-10-09 16:54:17 UTC**, the production inventory from merge time
(16:48:29 UTC) contained:

| Stage | Records | Exact prediction links |
| --- | ---: | --- |
| Saved scheduled runs | 0 | Not measurable |
| Saved candidate rows | 0 | Not measurable |
| Canonical observations | 0 | Not measurable |
| Frozen opportunities | 0 | Not measurable |

Every per-role/per-field count is therefore empty, not successful coverage.
No deployed app SHA or post-merge scan execution is verified. The latest scan
remained 37960183886 on the previous commit at the subsequent check.
The machine-readable evidence is
`ml/reports/provenance_post_deployment_verification.json`.

## Contract validation and fixes

The final input matrix and calibration map are now JSON-encoded inside the
DataFrame's ModelProvenance string, preserving full Python float round trips.
Existing string/object readers decode before canonical JSONB persistence and
opportunity freezing. Previously persisted rounded objects remain readable but
cannot be claimed to have valid original hashes. No historical rows are edited.
The regression reproduces the old default-pandas hash failure and verifies
exact feature values, raw probabilities and calibration identity after the fix.

Customer tables, CSVs and raw row previews consume a display copy without
ModelProvenance/SourceScanId. Storage is unchanged. Scanner and stock API responses
use explicit field allowlists; automation candidate export also uses an explicit
schema and does not copy internal model inputs. Its existing intentional model
probability fields are unchanged. Watchlist CSVs contain watchlist fields only.
Admin research storage remains available for separate diagnostic tools.

## Call path and preservation

`scheduled-scans.yml` → `scheduler.cron_runner` →
`scan.engine.run_breakout_scan` → SourceScanId → `_score_prebreakout` →
`_results_to_json` → saved runs/daily snapshots; the same result records enter
`observation_capture` → canonical `models.prebreakout`.

AI-confidence's capture is in `scan.execution`, which the scheduled engine path
does not call. We do not introduce AI predictions into scheduled scanning here.
The existing workflow comment claiming both models are scored is not evidence
of actual AI execution. Both-role prospective scheduled coverage remains a gap.

Brief → `_prebreakout_picks` recalculates under existing behavior →
`build_opportunities` → `freeze_opportunity` → `raw_signal.models`.
Tests retain actual later inference timestamps separately from earlier snapshot
times. Frozen rows with a recalculation are not the same inference as saved rows;
source scan identity alone does not establish identical predictions.

The audit keys exact links by source scan ID, symbol, role and inference time,
then compares complete provenance hashes. No ticker/day heuristic joins are used.
Existing first-write/idempotency keys are unchanged; fixture regressions pass.
No production linkage or idempotency outcome is claimed with zero new rows.

## Safety and cost

No scoring, model inference count, ranking, threshold, calibration computation,
training, target definition, research capture policy or scheduled cadence changed.
Existing inference fixtures check unchanged displayed outputs and row ordering.
No production inference was invoked for comparison.

Production payload size/runtime is not measurable without a new run. A local
synthetic 100-row/75-feature fixture measured 3,301 bytes without provenance and
429,501 bytes with one role: 4,262 added bytes per row. This is a fixture, not
production capacity or latency evidence. No DB reads are added to inference;
capture is linear in rows/features and cached model hashing remains unchanged.

## Target and readiness

Original-target outcomes remain separate. Top-N absence is never a negative
target. Final training/calibration cutoffs, target versions and source market
timestamps can still be unavailable; incomplete future scanner/OHLC coverage
blocks original-target-compatible evaluation. No proxy-target performance metric
is computed.

**Result: WAITING_FOR_PROSPECTIVE_RUN.** Source verification also identified
real defects on current main; branch fixes are tested but NOT merged/deployed.
Prospective PreBreakout capture requires review/merge of these fixes and a real
scheduled-run check. Both-role scheduled collection and evaluation are NOT READY.
Re-run the read-only Diagnostics inventory after the next normal scheduled run;
do not force a production scan without separate approval.

## Tests

Focused regression: 288 passed, one existing NumPy deprecation warning.
Cleanup/verification rerun after correcting the UI-module line-limit failure:
38 passed. Final full-suite results are recorded in the delivery report.
Ruff and syntax checks pass. No tests were weakened or deleted.

Final full `pytest tests -q`: **2,938 passed, 1 failed, 24 skipped,
10 warnings, 324 subtests passed**. The remaining failure is the existing
caller-scope audit scanning the untouched nested `ai_scanner/tests` checkout
as application code. The UI-module size-limit failure is resolved and its test
passes without changing the limit.

## Files and follow-up

Added: this report, `scripts/verify_prediction_provenance.py`,
`tests/test_provenance_verification.py`, and
`ml/reports/provenance_post_deployment_verification.json`.
Modified: `analytics/prediction_provenance.py`, `ui/results.py`,
`tests/test_prediction_provenance.py` (representation-aware assertions retain
the same content checks), and the provenance-capture contract document.

The verification script also includes hour-bucket-aware canonical filtering,
old-snapshot/new-inference frozen filtering, strict exact-link comparison for
both stages, payload caps, and sanitized failure output. No migration or
production scan is introduced.

After a normal post-merge scheduled scan, repeat read-only verification with:

```sh
gh workflow run diagnostics.yml --ref codex/provenance-post-deployment-verification \
  -f script=verify_prediction_provenance.py
```

Review and authorize any merge separately. Do not treat the current zero-row
inventory as production PASS or as evidence sufficient to evaluate model AUC.
