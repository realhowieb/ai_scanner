# PreBreakout scanner-history inputs

Contract: `public_snapshot_observations_v1`.

The active diagnostic training path (Run 17) and live inference share
`snapshot_feature_frame`. The legacy D suffixes on delta/slope/rolling features
mean **scanner observations**, not elapsed trading days. This change supplies
that observation history; it does not reinterpret the legacy feature names as
daily bar offsets.

## Data and ordering

- Use canonical public snapshots only: `username=cron`, `label=daily_snapshot`,
  `is_snapshot=true`. Private/custom scans never enter this feature history.
- Live inference makes one read-only query for all requested tickers, considering
  at most 128 snapshots in the preceding 90 calendar days and returning at most
  64 observations per ticker. Only input fields are returned, without model
  vectors, account fields or economic outcomes. Current production snapshots
  have 100 rows each, bounding the source expansion to 12,800 rows.
- Stored observations must be strictly earlier than the latest requested input.
  The shared builder sorts within each ticker, drops stored duplicates in favor
  of current inputs, and uses causal shifts/rolling windows. Later observations
  cannot enter an earlier row's features. Duplicate current ticker/timestamp
  pairs and invalid timestamps are rejected.
- One price-provider batch enriches historical and current observations with the
  prior available OHLCV values; benchmark context retains its existing cache.
  This does not manufacture scanner scores for days on which a ticker was not
  captured in a public snapshot.
- Run 17 selects the same public snapshot population, enriches the complete
  stream, and builds features **before** labels discard non-candidates or
  immature rows. `build_ml_dataset` recognizes the prepared contract marker and
  does not recalculate rolling features on the filtered candidate subset.

## Missingness and evidence

Insufficient observations remain missing. The scorer retains its existing
zero-fill behavior and records the default mask. Enrichment provenance includes
contract identity, scanner-history availability and loaded history-row count;
successful downloads do not imply that every required input was available.

The serving-skew audit now distinguishes the old scan-only comparisons from a
`current_contract` history replay, with per-feature missing percentages. The
replay is reconstructed evidence, not the original served matrix. Preserve
captured production vectors and original predictions.

## Deployment and evaluation

Populating previously defaulted features can change scores from the existing
champion. No model weights, calibration map or active registry row are changed
by this patch. After deployment, inspect recent captured default masks and
contract identity. Do not claim that all 75 inputs are complete: newly seen or
sparsely captured tickers may still have insufficient history.

Run a reviewed out-of-time comparison before retraining or promoting a model.
The old champion was trained under the previous mixed history pipeline; this
contract improves reproducibility but does not establish its current accuracy
or calibration. Legacy Run 16 and older training entry points remain legacy;
the active workflow uses Run 17. Historical mutable snapshots cannot recover
versions overwritten before capture, so replay cannot certify original
point-in-time training data.

Validation covers shared-builder parity, filtering after feature construction,
ticker isolation, future-row exclusion, duplicate rejection, one batched history
query/provider enrichment, read-only public scope, and missing-history fallback.
