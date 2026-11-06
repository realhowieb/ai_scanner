# Production PreBreakout status

Admin → Models → What is trained and running? displays PreBreakout separately
from AI Confidence and the ML v4 challengers. Its read-only admin endpoint is
`GET /v1/admin/ml/production/prebreakout`.

The configured model comes from the latest active `prebreakout_models` record,
ordered exactly as the serving loader. The response includes model version,
training time and dataset version when recorded. It never downloads model bytes
or loads/deserializes the artifact.

Latest recorded inference is independent evidence from `signal_outcomes`:
`raw_signal.models.prebreakout.loaded_artifact` supplies version and checksum,
with `inferred_at` and signal time. Captured predictions may legitimately be
zero. A legacy positive prediction is included only when it has a recorded
model version and no explicit provenance status. Explicit unavailable events
and legacy display zeros are not counted as evidence of successful inference.
Missing tables or records are unknown; database/query failures return 503.

Neither an active database record nor a previous inference proves what every
running process currently has cached. Local fallback artifacts remain possible.
The card does not train, promote, replace, or probe models and makes no new claim
about model quality. No migration or change to production scoring is required.
