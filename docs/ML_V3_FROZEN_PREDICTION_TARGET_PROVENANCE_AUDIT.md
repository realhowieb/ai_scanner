# HSF ML v3: Frozen Prediction and Target Provenance Audit

## Executive decision

**A trustworthy production-model evaluation cannot be performed on the audited
1,041 frozen opportunities today.** The recovered returns are valid evidence
for outcome analysis, but they do not supply either model's original target or
the model, calibration, inputs and actual inference time behind each prediction.

Classification of this cohort:

| Classification | Observations |
|---|---:|
| VERIFIED_EVALUABLE | 0 |
| RECONSTRUCTABLE_WITH_LIMITATIONS, established by this audit | 0 |
| NOT_EVALUABLE as a production-model prediction/target pair | 1,041 |

This does not mean price outcomes cannot be reconstructed or the models lack
signal. It means a historical production prediction/target pairing has not been
established. No AUC, calibration, leakage-safety certification, or model-quality
claim follows from an empty evaluable cohort.

## Sources, scope and reproducibility

- Latest main fetched and inspected: `4866e642ff4f06fba731ff5ce71c3652cc1c54b1`.
- Prior read-only audit/report: `codex/post-rescore-audit`, report commit
  `3d136b2`; its 613-row repair and idempotency findings remain intact.
- Audit branch: `codex/frozen-prediction-audit`, based on that report branch.
  Main was not merged or promoted.
- Production Diagnostics runs: `37890991616` (observation coverage),
  `37891497595` (scheduled history), `37891653258` (floor diversity),
  `37892172438` (canonical research store), and `37892504220` (final read-only
  enforcement check). All completed successfully. Final production-read code:
  `0591559`.
- Frozen observation scope: source `opportunity`, fired September 1 through
  October 9 05:12 UTC, the same fixed population as the prior audit.
- Scheduled source scope: `runs.username='cron'`, `label='US_MARKET'`, same date
  bounds. This excludes personal scans; results expose aggregate counts only.
- New machine-readable report:
  `ml/reports/frozen_prediction_provenance.json`.
- Direct SQL starts a database-enforced repeatable-read, read-only transaction,
  using an explicit transaction on an autocommit connection and checking
  `SHOW transaction_read_only` before reading any observations,
  with 15-second connection and 30-second statement timeouts. Limits are 10,000
  observations, 2,000 runs, 64 MB total payload and 2 MB per run. A total cap
  exceedance fails; oversized individual payloads are explicitly counted.
  The observed scope reached none of these limits.
- No schema initialization, provider calls, artifact deserialization, inference,
  fitting, reconstruction, or production writes were performed.

Run through the existing Diagnostics workflow with
`script=frozen_prediction_audit.py` on this audit branch. The JSON bundle is
gzip/base64 encoded in logs for transport and contains no credentials or users.

## Prediction lineage

| Stage | PreBreakout | AI confidence |
|---|---|---|
| Registry observed | `prebreakout_models` id 13, `prebreakout-xgb-v9`, trained Sep 10 16:46 UTC | Prior production audit: `ai_confidence_models` id 3, `ai-confidence-xgb-v1`, trained Sep 9 20:55 UTC |
| Feature order | 75 ordered registry feature names | 6 ordered registry feature names |
| Load | Active database bundle; 15-minute process cache | Active database bundle; 15-minute process cache |
| Fallback | Local artifact on database load exception; no bundle produces zero displayed probability | Local joblib/metadata when database bundle unavailable; missing model/columns attaches a warning |
| Inputs | Live enrichment, historical features, bundle preprocessing, missing feature/NaN zero fill | Named numeric columns, NaN zero fill; absent columns skip inference |
| Raw output | `predict_proba(X)[:,1]`, unit 0-1 | `predict_proba(X)[:,1]`, unit 0-1 |
| Calibration | Bundle map via `np.interp`, clipped 0-1; unusable map passes raw output through | Metadata map via interpolation; unusable map passes through |
| Display | `PreBreakoutProb%`, calibrated x100, rounded 1 decimal | `AI Confidence`, calibrated x100, rounded 1 decimal |
| Freeze | `opp.prob` stored as `prebreakout_prob` | `freeze_opportunity` supplies `ai_confidence=None` |
| Original target | `FutureQualitySetupHit` | `ForwardReturnHit` |

References:
- PreBreakout registry loading: `db/prebreakout_models.py:232`.
- PreBreakout cache/fallback: `ml_prebreakout.py:3691`.
- Live inputs: `ml_prebreakout.py:3743`; scoring/defaults: `:3780`.
- Calibration interpolation: `ml_prebreakout.py:3293`.
- AI registry loading: `db/ai_confidence_models.py:152`.
- AI database/local loader: `scan/ai_confidence.py:98`; scoring: `:225`.
- AI training features and matrix: `scripts/train_ai_confidence_model.py:49`,
  `:68`; validation/calibration: `:103`, `:125`.
- Frozen opportunity fields: `db/signal_outcomes.py:155`, `:182`, `:198`.

The registry describes what is active when read. It cannot establish which
artifact was served for any prior observation. Model versions are labels, not
immutable artifact hashes; historical registry versions repeat, and metadata
can be updated. The code constant is v16 while the observed champion is v9.

### Observation time versus inference time

`scheduler/morning_digest.py:126` calls `score_prebreakout(df)` to build picks.
`ui/market_brief.py:76` calls that helper after loading a saved scan.
The brief independently gets `snapshot_time` through `_snapshot_time`
(`ui/market_brief.py:26`, `:120`). `analytics/opportunity_freeze.py:28` uses
that timestamp when freezing the composed opportunities.

Consequently the code permits a displayed probability to be recalculated when
the brief is built, then frozen under a separately chosen snapshot timestamp.
The actual inference time and source run ID are not persisted. This is a
verified lineage risk, not proof that every historical row was rescored late.
It blocks treating `fired_at` as a verified model-inference timestamp.

### Deployment changes

Git history verifies these code changes, but not deployment/serving times:
- `4350e92` (Oct 7): live-model calibration reporting.
- `00d92bd` (Oct 8): `_live_feature_frame` adds scan timestamp, OHLCV and
  benchmark enrichment before scoring.
- `da304fb` (Oct 8): fixes blank live PreBreakout scores.

Before `00d92bd`, inference called `add_prebreakout_features(df.copy())`
directly and zero-filled unavailable model fields. An inference pipeline hash
is required in addition to model identity to distinguish these inputs.

## Provenance coverage matrix

| Evidence on the frozen observation | Count | Percent |
|---|---:|---:|
| ID, ticker, observation timestamp | 1,041 | 100.00% |
| Stored displayed PreBreakout percentage | 592 | 56.87% |
| Stored AI-confidence prediction | 0 | 0.00% |
| Five-day benchmark return | 882 | 84.73% |
| Served model version / artifact hash | 0 / 0 | 0.00% |
| Feature schema / final feature vector | 0 / 0 | 0.00% |
| Calibration identity | 0 | 0.00% |
| Training-data cutoff | 0 | 0.00% |
| Original target name / version / value | 0 / 0 / 0 | 0.00% |
| Raw model probability | 0 | 0.00% |
| Actual inference timestamp | 0 | 0.00% |

These are field-presence counts, not verification of values. All 1,041 rows
are missing the artifact/input/target chain; 449 also lack a displayed
PreBreakout probability. AI prediction evaluation is blocked for all 1,041.

Actual raw-signal keys are only `hsf_score`, `score_version`, `score_components`,
`primary_setup`, `status`. Indicator keys are `primary_setup`, `signals`,
`n_signals`, `gap_pct`, `chg_pct`, `fading`, `status`. These are HSF composition
inputs, not either model's final feature vector or target identity.

### Alternate canonical research store

The existing `hsf_observations` archive was also inspected, rather than assuming
`signal_outcomes` was the only possible source. In the same date bounds it holds
**10,953** observations in `scheduled:us_market`, across Sep 22-Oct 8:

| Canonical archive evidence | Count |
|---|---:|
| Code model-version tags | 10,953 |
| Observation schema / scanner commit tags | 6,720 / 6,720 |
| Stored PreBreakout / AI-confidence prediction | 0 / 0 |
| Served artifact hash / calibration identity / target identity | 0 / 0 / 0 |
| Archived resistance-position source | 3,920 |
| Archived high and low together | 0 |

`analytics/hsf_observation.py:45` resolves model versions by importing
`MODEL_VERSION` constants. It does not ask the loaded bundle which artifact
served the row. The tag can therefore say the code's v16 while the active
registry champion is v9. This tag is not verified served-model provenance.

`analytics/observation_capture.py:112` calls `build_observation` without its
`models` argument, even after the scheduled result was scored. The constructor
stores an empty models object (`analytics/hsf_observation.py:149`). Run 57
research metadata archives scanner commit and a small set of row fields
(`analytics/research_metadata.py:43`, `:110`, `:143`), not the model's ordered
75-feature matrix. Its feature-schema tag identifies the canonical observation
schema, not the inference schema. None of these 10,953 rows supplies a stored
production-model prediction/target pair for evaluation.

### Coverage by entry date

All dates are in 2026. Source is opportunity throughout. Full field counts and
percentages by date/source are in the JSON; unavailable provenance is zero in
every date group.

| Entry date | Rows | Stored PreBreakout | Percent |
|---|---:|---:|---:|
| Sep 14 | 635 | 586 | 92.28% |
| Sep 15 | 25 | 2 | 8.00% |
| Sep 16 | 30 | 3 | 10.00% |
| Sep 17 | 35 | 0 | 0% |
| Sep 18 | 35 | 0 | 0% |
| Sep 21 | 20 | 0 | 0% |
| Sep 22 | 45 | 0 | 0% |
| Sep 23 | 40 | 0 | 0% |
| Sep 24 | 20 | 0 | 0% |
| Sep 25 | 15 | 0 | 0% |
| Sep 28 | 32 | 0 | 0% |
| Sep 29 | 16 | 0 | 0% |
| Sep 30 | 18 | 0 | 0% |
| Oct 1 | 15 | 0 | 0% |
| Oct 2 | 10 | 0 | 0% |
| Oct 5 | 10 | 0 | 0% |
| Oct 6 | 24 | 1 | 4.17% |
| Oct 7 | 10 | 0 | 0% |
| Oct 8 | 6 | 0 | 0% |

### Scheduled history and candidate linkage

51 canonical scheduled runs contain 5,100 result rows. Their dates begin Sep 22;
no canonical scheduled run in this query covers the dominant Sep 14 cohort or
its immediate 1-3-day future setup window.
The scheduler defaults to `CRON_TOP_N=100` (`scheduler/cron_runner.py:460`).
These retained results are a selected set, not proof of complete same-symbol
scanner-state coverage across the full market.

- 5,100 rows have candidate/setup source fields (`IsBreakout`, `BreakoutScore`,
  `BreakoutPos20D`, `Last`). Presence does not prove candidate eligibility.
- 400 have both raw and displayed PreBreakout values: 100 on Oct 7, 300 on Oct 8.
- No AI-confidence outputs or original target labels were present in these
  scheduled payloads under the canonical field names.
- Zero rows contain all 75 current-schema fields. Every one of 69 engineered
  fields is absent from all 5,100 stored rows. Some could be recalculated from
  source history, but that would not prove the actual served vector.
- There are 4,880 distinct signatures of the available current-schema source
  fields, largest cluster 9. These are **not** signatures of final model inputs.
- 255 frozen observations have at least one same-ticker/entry-day scheduled
  candidate; zero have an exact scheduled-run timestamp candidate.

Ticker/day matches are insufficient: multiple runs can share a date, snapshot
selection is separate, and model/calibration identity is absent. No match was
silently selected or used for evaluation. Personal scans, older universes,
unversioned external bars, local model files, and platform deployment records
were not claimed as verified alternatives.

## Exclusion funnel and evaluation policy

This reproduces the prior diagnostic funnel, without treating its cutoff as a
verified production training boundary:

| Stage | Remaining | Removed at stage |
|---|---:|---:|
| Frozen population | 1,041 | - |
| First observation per ticker/entry day | 371 | 670 repeated observations |
| Entry after Sep 22 conservative gap | 104 | 267 |
| Finite stored five-day return | 56 | 48 |
| Stored PreBreakout prediction | 0 | 56 |

Selection happens before inspecting outcomes. All 592 finite displayed outputs
occur Sep 14-16 except one Oct 6 observation; that recent observation has no
usable five-day outcome in this audit. This explains the empty prior model
diagnostic without changing exclusions to recover rows.

The eight-session buffer is motivated by a three-day setup lead plus five-day
economic path, but it is only a conservative diagnostic using the current
registry's training timestamp. Neither a training-data maximum nor label-window
maximum is recorded in the checked registry fields (`training_data_end`,
`train_end`, `training_end`). Validation fold end dates are not the final fit
cutoff: final PreBreakout fitting uses the full selected matrix
(`ml_prebreakout.py:4643`). Actual served model is unknown, so the buffer does
not certify independence. Weekday-count setup leads and exchange-session paths
also need explicit calendar semantics.

Defensible future cohort policy:
1. Require immutable observation-to-inference-to-artifact linkage and actual
   prediction/data-availability timestamps.
2. Verify the original target and its full window; retain censored/missing cases
   explicitly rather than turning them into negatives.
3. Exclude windows overlapping the served artifact's training and calibration
   data, using actual recorded boundaries and trading-calendar rules.
4. Choose deterministic ticker/time observations before labels, then report the
   exclusion funnel and independent temporal support.
5. Calculate metrics only for the resulting verified cohort.

No cohort in this audit satisfies step 1, so performance calculation stops.

## Exact targets and reconstructability

### ForwardReturnHit

Constants: `ml_prebreakout.py:93`: +4% upside, -2% downside, five days.
`_forward_path_hit` (`:570`) chooses the first non-null close bar on/after the
normalized scan date, enters at its close, and examines the following five
non-null close-indexed bars. High/low columns fall back to close when absent.
Stop is checked before upside on each bar; a bar crossing both thresholds is
a failure. Returns are thresholded relative to that entry close.
NaN closes are removed before choosing positions, so missing sessions can make
five observed bars span more than five trading sessions. High/low NaNs are not
explicitly rejected by the path helper; their comparisons do not trigger either
threshold. Full calendar coverage therefore needs separate verification.

`add_forward_return_labels` (`:693`) has a separate branch: if `Return_5D`
already exists, it drops missing returns and labels `Return_5D >= .04` without
path checks. Otherwise it obtains a complete close-return path and uses the
path hit when available, falling back to the same close-return threshold when
the path helper returns None. No bars/complete forward return removes the row.

The historical target therefore depends on the branch, data source, adjustment
policy and exact bar set. The repaired `return_5d` field has a different name
and cannot establish which training branch was used. MFE/MAE aggregates cannot
recover hit/stop ordering. One cannot replace this target with `return_5d > 0`.

### FutureQualitySetupHit

`prebreakout_candidate_mask` (`ml_prebreakout.py:560`) requires no breakout,
BreakoutScore below 8, and price below resistance (`BreakoutPos20D < .999` or
price `< .999 * High20`, `:535`). Missing resistance fails eligibility; missing
score is defaulted to zero; absent breakout flag defaults false.

`add_prebreakout_target_label` (`:759`) first applies the forward-return labeling
and its row filtering. On the retained same-symbol history, a future quality
setup means `IsBreakout` or BreakoutScore >=8, **and** ForwardReturnHit=1.
Candidate positives require such a later setup with 1-3 weekdays counted by
`np.busday_count`; same-day later scans do not qualify. This is weekday counting,
not exchange-holiday-aware session counting. The economic entry is the future
setup's close, not the original candidate's close. Labels can depend on roughly
eight sessions ahead with complete daily data. Eight sessions is not a strict
upper bound if dropped/missing bars stretch the observed-bar path; the actual
target end timestamp is needed for overlap checks.

Git blame verifies the principal labeling implementation predates the September
models (Sep 9 changes); it does not prove the per-row historical target variant.

Faithful reconstruction needs the contemporaneous candidate fields, a complete
future same-symbol scanner-state history, the label-generation branch/version,
and the corresponding full price paths. Incomplete future setup capture cannot
prove that a qualifying setup did not happen. Missing future rows must not be
manufactured as negative labels.

**Reconstruction decision:** conditional local research reconstruction is
possible in principle with sufficient independent archives, but neither model's
historical served prediction/target pairing is defensible from the audited
records. In particular the dominant recovered cohort predates the canonical
scheduled history in scope. No original targets were reconstructed or backfilled.

## Temporal-safety findings

- Sorting is stable Symbol/Timestamp (`ml_prebreakout.py:965`); feature generation
  preserves row order (`:1360`). Group shifts operate on sorted symbol history.
  Their windows can count observations rather than unique trading days, so
  repeated snapshots affect reconstruction of temporal features.
- OHLCV and benchmark enrichment restore row identities and use backward
  as-of merges (`:1209`, `:1277`). Context bar availability timestamps are shifted
  to the following UTC day (`:1144` and OHLC context construction).
  This inspected code is relevant evidence, but no historical provider revisions
  or per-row availability times are frozen.
- Repeated stored rows, mutable daily snapshots (`db/runs.py:129`, `:161`) and
  distinct timestamp sources prevent assuming run history is an immutable
  point-in-time inference archive.
- Original target windows need full overlap checks. The earlier audit identified
  a five-session PreBreakout purge versus the longer setup-plus-path target;
  no validation rule was modified here.
- AI training uses an 80/20 chronological split (`ml_prebreakout.py:1734`), without
  purge in that function, and fits calibration on the same validation predictions
  (`scripts/train_ai_confidence_model.py:125`). Calibration dataset identity and
  temporal boundaries are not frozen on observations.

Result: **historical temporal safety is UNVERIFIED**. No PASS is inferred from
the empty evaluation population or the existence of backward joins alone.

## Probability clustering: evidence and limits

The frozen cohort has 592 finite displayed PreBreakout values; 529 equal 13.1
(89.36% of present probabilities, 50.82% of all observations). Distinct displayed
values: 21. None of those observations freezes a raw probability or final input.

The active calibration map has first knot x=0.03425020499116203 and
y=0.13126323218066338. `np.interp` clamps raw values below that x to the first y;
displaying y*100 to one decimal gives **13.1**. Equal y knots further along the
map can also create ties. The implementation interpolates between knots; the
docstring's "step-map" wording is not an exact description of that operation.

Recent scheduled payloads provide direct numeric evidence:
- All 400 stored raw/display pairs match this current map after rounding.
- The 84 displayed 13.1 rows have **84 distinct raw values**, min 0.0030624762,
  max 0.0330848135; all are below the first x knot.
- The endpoint clamp and rounding therefore explain these recent 84 ties.
  Equal displayed probabilities do not establish identical feature vectors.

For the older September cluster, this mechanism is consistent with the values
but is not a proved historical cause: the served artifact/map is unrecorded,
and no corresponding raw predictions exist in the audited canonical history.

Other code mechanisms: missing model yields zero; missing PreBreakout input
columns/NaNs are zero-filled. A 13.1 value is not itself the no-model zero
fallback, but a weak raw prediction caused by incomplete inputs could be mapped
to the calibration floor. The 69 missing engineered fields in persisted scan
payloads are evidence of absent archived vectors, **not proof those fields were
all zero during inference**. Live enrichment can produce fields only in memory.
No missing-feature/default causality or exact served-vector collapse is claimed.

## Minimum remediation proposal

**Proposal only; no persistence/schema/scoring change implemented.** Reuse the
existing model registry, scored result persistence and observation freeze path.

At inference, attach a compact versioned provenance object for each model role:
- immutable registry/artifact hash, model version and inference-code SHA;
- actual inference timestamp, input observation timestamp and source run ID;
- ordered feature schema/hash, preprocessing identity, final numeric feature
  vector and missing/default mask (explicit NaNs/nulls);
- raw probability and calibrated probability in documented 0-1 units;
- calibration hash and resolvable immutable map revision;
- actual training-data and training-label-window boundaries, plus calibration
  dataset boundaries;
- model target name/version and target-definition code identity;
- enrichment/provider revision and latest input-bar availability timestamp,
  fallback status/reason.

Populate the existing canonical `models.<model_role>` namespace with this
object, and copy it into frozen opportunity `raw_signal.models.<model_role>`
when an opportunity is frozen. Link the actual source scan/inference instead of
inferring it from another snapshot timestamp. Keep existing display percentages
and observation columns backward compatible. Existing schema and scanner-commit
tags remain useful but separate from model, inference-feature and calibration
identity. Extend these existing stores; do not introduce a parallel archive.

The model registry should retain immutable calibration/preprocessing revisions
or a resolvable compact revision record alongside its metadata. Hashing a map
without retaining the corresponding revision cannot make it reproducible.
Store identifiers/maps and numeric inputs, never model bytes or secrets in each
observation. A 75-number vector is modest, but measure storage cost before rollout;
do not persist unrelated provider payloads or user data.

For targets, archive the necessary same-symbol scanner-state sequence during
the future setup window and completed price paths through the economic window.
Use the existing maturation infrastructure to record target identity, branch,
target value, window end, underlying source IDs and missing/censoring reasons.
Do not classify missing future scanner capture as absence of a setup.

Ownership:
- Existing inference functions produce the actual model/input provenance.
- Existing scheduled scan persistence retains it and the source run identity.
- Existing opportunity freezing copies it without new inference.
- Existing maturation computes versioned targets after verified window completion.

Before rollout, tests should cover provenance survival through persistence,
both model roles, raw/calibrated units, source-run linkage, immutable calibration
revision resolution, default masks, original target branches and same-bar ordering,
missing future setup capture, overlapping windows, and old-row compatibility.

Exactly one next action: **HSF ML v3 - Freeze Prediction Provenance and Target-Path
Evidence**. Implement and verify the above minimum contract, then collect a
prospective cohort with enough independent entry dates. Tuning cannot be judged
until predictions can be paired with their actual targets and model identity.

## Files and tests

Added audit-only files:
- `scripts/frozen_prediction_audit.py`
- `tests/test_frozen_prediction_audit.py`
- `ml/reports/frozen_prediction_provenance.json`
- This report.

Ten focused tests cover missing/NULL provenance, exact training-boundary
exclusion, target identity, zero versus NaN probabilities, deterministic selection
before outcomes, database-enforced read-only entry and query bounds, distinct
raw outputs at a calibrated floor, missing source fields without filling, and
the inability of a code-version tag to certify a served artifact.
The read-only guard also rejects the audit before queries if the server reports
that the transaction is not read-only.

Relevant regression command:
`python -m pytest tests/test_frozen_prediction_audit.py tests/test_post_rescore_audit.py tests/test_ml_readiness.py tests/test_ml_v3_audit.py tests/test_research_dataset.py -q`

Result: **111 passed, 0 failed, 10 subtests passed, nine existing deprecation
warnings** (Starlette/httpx and NumPy timedelta). New-file lint and syntax checks
pass. The NULL-provenance test was also rerun after its final hardening edit.

Production data, model artifacts, scoring, feature calculations, validation,
thresholds, calibration, scanner scheduling and previous audit artifacts remain
unchanged. No main promotion. The unrelated nested local checkout and earlier
unfinished Today work remain preserved.
