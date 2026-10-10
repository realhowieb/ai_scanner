# PR #60 Post-Deployment Provenance Verification

## New-bucket production verification: canonical PASS, overall PARTIAL

Scanner **37976348510** successfully executed production SHA
`61dc0526ca8d66803e8d8d55dd5f114688bb9243`, started 2026-10-09T18:52:54Z,
completed by 18:59:11Z. This scan occupies the distinct **18:00 UTC bucket**.
Read-only Diagnostics **37977289730**, verifier commit `bd63ab6`, audited at
19:01:31Z. Evidence: `ml/reports/pr60_new_bucket_production_audit.json`.

- Capture logs: **100 attempted, 100 inserted, 0 duplicates, 0 failures**.
- Saved run 3230 and updated snapshot 3227: 200 representations, **100 unique
  inference identities**, no conflicting duplicates; 1,010,406 bytes total.
- Canonical CANDIDATE: **100/100 captured, exactly linked and hash-identical**
  to saved provenance using source scan identity, ticker, role and inference time.
- Newly frozen opportunities 11282-11286: **5/5 captured, exactly linked and
  hash-identical**. Displayed PreBreakout values remain absent for these rows.
- CONTROL 100 and NEAR_MISS 50: expected non-invocation, not lost predictions.
  AI-confidence is absent on this scheduled path as expected.
- Artifact/schema/calibration hashes, feature/mask lengths, finite inputs,
  probability ranges and mask types pass at saved, canonical and frozen stages.
- Actual input timestamps, target versions, training and calibration boundaries
  remain unavailable. No scan/render timestamp substituted for market-data time.

The scoped window returned one inference source represented in two saved records;
no mixed-run denominator was observed. Earlier frozen records were excluded by
the window; this does not constitute a new before/after immutability comparison.
Brief recalculations and authorized live tier/API/table/CSV privacy remain
UNVERIFIED for the previously documented reasons. Existing artifact privacy
verification is not generalized to those surfaces.

**Canonical preservation: PASS. Frozen preservation: PASS. Overall: PARTIAL.**
Scheduled PreBreakout storage collection is demonstrated; complete collection
verification remains limited by the outstanding live checks. Evaluation remains
NOT READY due to original-target evidence and metadata/boundary gaps. No model
performance was calculated and no production data was changed by the audit.
Scoped tests: **14 passed** in 0.97 seconds; verifier Ruff check passes.
Next: close authorized live privacy/recalculation checks and address evaluation
evidence separately, without changing scoring or certified capture identities.

## New-bucket checkpoint: 2026-10-09 18:50 UTC

**WAITING_FOR_NEW_BUCKET.** Latest main and verification branch were fetched;
PR #61 remains open. The latest five scanner workflow executions still end
with successful 37969482226 (executed SHA `61dc0526ca8d66803e8d8d55dd5f114688bb9243`).
No later successful execution exists in the reviewed list. Production inspection
stopped at this gate: no scan, diagnostic, inference or mutation was triggered.

`analytics/observation_capture.py:35` floors the supplied scan timestamp to its
UTC hour. Both prior scans fall in the 17:00 UTC bucket; workflow completion
at 18:00 does not establish a new capture bucket. The earlier run's logs already
confirmed 100 duplicates, zero new candidate inserts and zero failures.
New-bucket insertion counts, linkage coverage and hashes remain **unknown**.

Focused tests rerun: **42 passed, 0 failed/skipped/warnings**, 1.98 seconds;
Ruff and syntax checks pass. No production privacy or Brief recalculation
claim has been upgraded from fixture coverage. Overall collection status
remains PARTIAL; this checkpoint is WAITING. Evaluation remains NOT READY.
Next: inspect an actual successful scan with a distinct supplied UTC scan-hour
and then scope the read-only audit to its identity. Do not treat elapsed time
or completion time as evidence that a suitable scan ran.

## Remaining-gap review

Latest five scanner executions still end with 37969482226 at 17:53 UTC.
**Canonical capture: WAITING_FOR_NEW_BUCKET**; no scan was triggered.
No additional production database inspection was necessary in the absence
of a suitable new execution. Earlier first-write evidence is not presented
as new per-inference canonical linkage.

The actual `scanner-automation-snapshot` artifact from 37969482226 was downloaded
read-only and parsed with strict JSON handling. `latest_scan.json` and its
historical copy are 72,571 bytes each, with 100 candidates; `status.json` is
363 bytes. Recursive field inspection found no internal `ModelProvenance`,
`SourceScanId`, feature names/values, default masks, calibration snapshots or
per-inference source IDs. The public top-level `models` object is intentional:
artifact, version, training time, feature count/schema version and availability
warning only. It is not the internal per-candidate provenance object.

| Surface | Live verification | Supporting evidence |
| --- | --- | --- |
| Automation latest/history/status | PASS | Actual published artifact inspection |
| Brief picks/opportunities, every tier | UNVERIFIED | Controlled redaction fixtures pass |
| Scanner/Stock Intelligence | UNVERIFIED | Source/controlled serialization coverage only |
| Customer tables and CSV | UNVERIFIED | Controlled serialization coverage only |

No authorized deployed account-tier session was established for this review.
No account was impersonated, subscription modified or inference-capable page
opened. Fixture results do not certify live customer responses.

Brief recalculation production linkage remains **UNVERIFIED**. The inspected
five freezes contain saved-source predictions, not independently identified
recalculation events. The existing call path computes picks in memory and
persists evidence only when applicable opportunities freeze; there is no
independent archive of every Brief recalculation. Controlled tests verify
actual inference timestamps, role-specific replacement and ambiguity behavior,
but cannot prove production recalculation linkage for unobserved events.

Historical row immutability is supported by previous duplicate logs and
first-write fixtures, not a new before/after production digest comparison.
Production provenance-specific overhead is not measured. No inference,
mutations, backfills, target changes or model performance calculations occurred.

Final focused suite: **42 collected, 42 passed, 0 failed, 0 skipped, 0 warnings**
in 2.02 seconds. Ruff and syntax checks pass for verifier and test modules.
Overall status remains **PARTIAL**; collection readiness is pending remaining
live checks, evaluation readiness is **NOT READY**. Next: a new-bucket scan
followed by read-only linkage verification and authorized customer-surface
inspection. No observation identity or certified behavior should change just
to eliminate expected deduplication.

## Post-scan update: PARTIAL

Successful production scanner run **37969482226** executed merge SHA
`61dc0526ca8d66803e8d8d55dd5f114688bb9243`, from 17:53:47Z to last update
18:00:05Z on 2026-10-09. Read-only Diagnostics **37970363522** completed
successfully; audit timestamp 18:01:59Z. Raw sanitized aggregates are in
`ml/reports/pr60_post_scan_production_audit.json`. The earlier waiting result
below is retained as the initial checkpoint, superseded by this update.

- Saved run 3229 plus updated daily snapshot 3227: 200 representations,
  **100 unique inference events**, zero conflicting identities, 1,010,408 bytes
  total stored JSON. All 200 captured representations passed probability,
  artifact/schema/calibration hash, numeric and mask checks.
- Five newly inserted freezes (11277-11281): **5/5 exact identity matches and
  5/5 identical provenance hashes**. All have captured PreBreakout evidence;
  all keep displayed PreBreakout probability absent, as before. This verifies
  forwarding beyond selected displayed PreBreakout picks without score changes.
- Canonical capture logs: 100 attempts, **0 inserted, 100 duplicates, 0 write
  failures**. Hour-bucketed identities (`analytics/observation_capture.py:35`)
  and first-write DO NOTHING (`db/hsf_observations.py:139`) preserve the earlier
  run's records. No exact canonical link to this new inference can be claimed.
- 109 new cohort records: 100 CONTROL plus 9 NEAR_MISS. Both model roles absent
  as expected; scheduled AI-confidence is not invoked. No extra inference added.
- Input timestamp, target version and training/calibration boundaries remain
  unavailable in all captured records. No substitute timestamps were used.
- Scan logs report 261.8 seconds scanner runtime, 100 candidates, zero provider
  errors/timeouts and successful automation publication with zero warnings.
  This is total scanner runtime, not measured provenance overhead.

**Forwarding/freezing: PASS for this run. Overall: PARTIAL.** Production
customer-facing API/table/CSV inspection and Brief recalculation linkage remain
unverified; controlled fixtures cover them. Canonical first-write deduplication
limits per-inference preservation within one hourly bucket and was not changed.
Evaluation remains NOT READY due to original-target/boundary evidence gaps.
Follow-up scoped verifier tests: **14 passed** in 1.01 seconds. No production
mutations, backfills or model/scoring changes were made by verification.

## Result

**WAITING_FOR_PROSPECTIVE_RUN**, checked 2026-10-09 17:50 UTC.
Latest main is `61dc0526ca8d66803e8d8d55dd5f114688bb9243`, the PR #60 merge.
No successful Scheduled Market Scans execution containing that merge exists
in the inspected latest ten workflow runs. A merge and green CI do not prove
scanner execution or production persistence.

Latest successful scanner run: **37964124080**, executed SHA
`62b7d09a439f5ee313e80e3d46079f50a9778966`, started 2026-10-09T17:08:17Z,
last updated 17:18:47Z. It predates the merge at 17:44:12Z.
Post-merge Smoke Checks run 37968371818 succeeded, but is not a scanner run.

## Evidence scope

No post-merge production records were inspected: counts and exact linkage
coverage are **unknown**, not zero or passed. No database audit was launched
against old records because it cannot establish deployment of this fix.
Prior evidence remains in the coverage/freezing report: 100 unique candidate
inferences in two storage representations, 100 exact canonical links, 150
expected unscored CONTROL/NEAR_MISS records and five freezes without evidence.
Those are historical findings, not post-fix results.

## Merged call path and controlled verification

Scheduled inference -> saved JSON -> canonical observations -> Brief
`source_models` -> build_opportunities -> freeze_opportunities ->
freeze_signal -> `raw_signal.models`.

- `ui/market_brief.py:68` forwards the saved source map.
- `ui/opportunities.py:97` starts each opportunity with available evidence;
  recalculated picks replace only the roles actually recalculated.
- `ui/results_intelligence.py:134` preserves unambiguous source evidence.
- `api/market.py:99` strips internal evidence from public Brief picks.
- `ui/entitlement_view.py:20` strips it for all opportunity entitlement tiers.
- `scripts/verify_prediction_provenance.py:65` enforces read-only transactions.

Controlled tests cover artifact identity, matrix/mask order, probability units,
schema/calibration hashes, legacy serialization, exact-link reconciliation,
ambiguity, actual Brief composition, saved forwarding into freezes,
recalculation timestamps, unchanged score/probability outputs, public Brief
redaction, immutable first-write behavior and automation export compatibility.
These are fixture results, not claims about deployed customer responses.

Scheduled AI-confidence remains not invoked. CONTROL/NEAR_MISS must not be
scored solely to increase coverage. Ticker-only matching is not exact linkage.
Future verification must separate newly inserted freezes from older first-write
records and verify inference identities/hashes through each applicable stage.

## Tests

`.venv/bin/python -m pytest tests/test_provenance_coverage_gaps.py
tests/test_prediction_provenance.py tests/test_provenance_verification.py
tests/test_automation_export.py -q`: **42 passed**, zero failures/skips/warnings,
2.02 seconds. Ruff and syntax checks on the verifier and these tests pass.
The new report reloads as strict JSON. No application code changed in this run.

## Safety, cost and readiness

No production scans, database mutations, migrations, backfills, provider calls,
model changes or inference were triggered. Unrelated local files are preserved.
Production payload size and runtime after the fix: **not measured**.
Customer-facing live API/table/CSV privacy: **unverified in production**;
controlled privacy/serialization tests pass.

Collection implementation is ready for prospective verification; end-to-end
production collection readiness remains unverified. Evaluation is **NOT READY**:
actual input timestamps, target versions, training/calibration boundaries and
complete original-target evidence remain gaps documented in prior audits.
Top-N absence is never a negative label. No proxy-target metrics were computed.

## Next step

Wait for a successful Scheduled Market Scans run including PR #60, or obtain
explicit permission for a manual scan. Then perform the bounded read-only audit
for that run's records, including newly created freezes and public exposure
checks. Do not merge or deploy this verification branch automatically.
