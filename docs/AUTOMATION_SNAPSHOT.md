# Scanner Automation Snapshot

HSF scheduled scanner runs publish a vendor-neutral JSON snapshot for external
automation systems. The scanner still owns market-data collection, feature
calculation, scoring, model inference when present in the existing scan path,
and scheduled execution. Consumers such as ChatGPT should only read the
snapshot and perform their own interpretation.

## Producer

The scheduled scanner writes:

- `artifacts/automation/latest_scan.json`
- `artifacts/automation/status.json`
- `artifacts/automation/history/YYYY-MM-DD/<run_id>.json`

The scheduled GitHub Actions workflow uploads the latest and status files as an
artifact named `scanner-automation-snapshot`.

Publishing is atomic: the exporter writes a temporary file and replaces the
target only after JSON serialization and validation succeed. If publishing
fails, the previous `latest_scan.json` remains intact whenever the filesystem
supports atomic replacement.

Historical retention defaults to 30 days and can be configured with
`AUTOMATION_HISTORY_DAYS`.

## Schema

Top-level fields:

- `schema_version`
- `generated_at_utc`
- `scan`
- `models`
- `summary`
- `diagnostics`
- `warnings`
- `errors`
- `candidates`

Candidate fields use normalized names for external consumers:

- `symbol`
- `price`
- `percent_change`
- `ema9`
- `ema21`
- `ema_spread_pct`
- `ema_signal`
- `rsi14`
- `rvol`
- `volume`
- `avg_volume`
- `prebreakout_score`
- `breakout_score`
- `prebreakout_ml_probability`
- `breakout_ml_probability`
- `scanner_signal`
- `rank`
- `data_quality`

Unavailable optional values are serialized as `null`. Non-finite values such as
NaN and Infinity are never emitted as invalid JSON.

## Model Provenance

The exporter records available model metadata without retraining:

- model version
- training timestamp
- source/artifact indicator
- feature-schema version when available
- feature count
- git SHA / GitHub Actions run ID when available

The current repository names the breakout probability model `AI Confidence`, so
that model appears as `models.ai_confidence`.

## Data-Quality Diagnostics

Each snapshot includes warnings for:

- missing symbols
- duplicate symbols
- invalid RSI values
- probabilities outside 0-1
- negative volume
- non-finite source values
- score/probability clustering, including suspiciously identical PreBreakout
  values across many candidates

Diagnostics do not change scanner results or scoring formulas. They only make
questionable output visible to downstream consumers.

## Consumer Boundary

External automation should treat the snapshot as read-only input. This
repository does not call ChatGPT/OpenAI APIs, embed external AI credentials, or
make Streamlit responsible for the integration.

To let ChatGPT consume the latest snapshot securely, provide it with an
authenticated retrieval path to the GitHub Actions artifact or copy the JSON to
a controlled storage location with access scoped to this artifact only.
