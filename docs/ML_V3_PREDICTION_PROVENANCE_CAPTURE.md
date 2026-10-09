# ML v3 Prediction Provenance Capture

## Contract and scope

Prospective capture only. No historical reconstruction, model tuning, migration,
production mutation, ranking change, or automatic promotion is included.
See the frozen-prediction audit at commit
`6bed4f509fd996231b989db45e2a09fddc0b1ef3` for historical evidence limitations.

`ModelProvenance` is a serializable per-row object with `prebreakout` and
`ai_confidence` roles. Existing displayed percentages remain unchanged.
Each role includes schema version, UTC inference time, supplied input timestamp
and scan ID, optional build SHA, verified loaded artifact identity, ordered final
input names/values, schema hash, default mask, preprocessing identity/snapshot,
raw/calibrated probabilities in **0–1** units, calibration snapshot/hash, target
identity/version and available training/calibration dates. Unknown values are
null, not guessed from constants, current registry state, or training time.

Database identities use the selected registry ID/version and SHA-256 of the
actual deserialized bytes. Local identities hash the same byte buffer passed to
joblib; filename is retained without an absolute path. Artifact identification
happens at cached load, not per candidate. Calibration snapshots are embedded
and resolvable directly; canonical sorted JSON SHA-256 verifies their identity.
Missing calibration means identity mapping, not an invented calibration model.

PreBreakout availability reports disabled/completed-best-effort/failed enrichment
operations and feature-generation fallback. Completed operations do not imply
complete provider coverage. Feature masks describe values defaulted to zero at
the final matrix boundary; preprocessing details remain in the captured plan.

## Call path

- Existing AI inference in `scan.execution` captures before confidence sorting.
- Scheduled results receive `SourceScanId` before existing PreBreakout inference.
- Both inference functions capture the actual matrix used in their single
  `predict_proba` call. No additional provider calls are made for capture.
- `cron_runner._results_to_json` serializes the object into saved run JSON;
  canonical capture copies it into `hsf_observations.models`.
- Brief's existing PreBreakout recalculation replaces that role with its actual
  new inference time and carries it through picks, opportunity composition and
  `signal_outcomes.raw_signal.models`.
- Existing freeze uniqueness/first-write behavior is unchanged. Older rows
  without provenance remain readable and are not upgraded in place.

AI source scan identity is unavailable if the invoking caller has not supplied
it before inference. Input timestamps are unavailable where scanner rows do not
carry an actual timestamp; feature-generation wall time is not substituted for
market-data time. Code version tags remain separate from loaded artifact identity.

## Original targets and evidence gaps

`FutureQualitySetupHit` requires eligible non-breakout candidates (score below 8,
price below the 20-day high), then a quality setup within 1–3 later weekday
observations, with economic `ForwardReturnHit`. The latter normally requires a
4% upside hit before a -2% stop in the following five bars, stop first when both
hit in one bar. Existing `Return_5D` training rows instead use the >=4% return
branch; absent high/low can use close fallback. These branches remain unchanged.

Do not substitute positive 5-day returns, benchmark excess returns or top-N
absence for the original target. The audit found incomplete future scanner-state
coverage, absent historical full OHLC and unknown final training cutoffs.
Original-target maturation is therefore **not implemented**. Captured target
evidence is explicitly unavailable pending original-target-compatible evidence.
Existing future outcomes remain in their separate outcome infrastructure.

## Deployment and performance

No schema migration is required: existing JSON/JSONB fields hold the additions.
Deploy this branch only after review/authorization; no production migration or
collection was executed during implementation. Existing scheduled jobs then
capture prospectively. Verify saved JSON, canonical models and frozen raw models
on a newly completed run, retaining inference times distinct from snapshot time.

Storage grows linearly with final feature count and candidate count. Calibration
and feature names repeat per row for independent resolution; no new DB queries
or second inference are introduced. Local artifact deserialization temporarily
holds a byte buffer. No live production throughput measurement was performed.

## Verification

Focused regression: 239 passed, one existing NumPy deprecation warning. Additional final results are recorded in the
 delivery report. Tests cover strict serialization, both roles, exact probability
units, feature ordering/default masks, unavailable legacy evidence, and unchanged
single-call inference/sorting. Existing persistence, capture, research-cohort,
calibration, opportunity and freeze tests protect compatibility/idempotency.

Remaining gaps: unknown historical identities cannot be repaired; existing source
timestamps are not universally supplied; unavailable training/calibration cutoff
metadata remains null. No claim of improved model performance or retrospective
evaluation readiness is made.

Full `pytest tests -q`: 2,933 passed, 24 skipped, 324 subtests passed,
10 warnings, one failure. The failure is the existing caller-scope source audit
reading the untouched untracked nested `ai_scanner/tests` checkout as production
code. Running pytest without a tests-directory scope also collects that nested
checkout and causes duplicate-module collection errors. Neither checkout nor
existing tests were weakened or removed. Ruff and syntax compilation passed.
A synthetic sanitized sample was written and strictly reloaded at
`/tmp/ml_v3_prediction_provenance_sample.json`; no production data was used.
