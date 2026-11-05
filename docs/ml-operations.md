# Operating ML readiness and model evidence

Admin → Research → ML v4 readiness separates the required rolling 20-trading-day
coverage from the last five trading days and the successful record streak after
the latest missing record in that window. These diagnostics never relax the
gates. A new failure resets the streak. An empty population is unknown, not
100%. Counts are observation records, not independent signal-days.

Missing scan-feature matches retain their canonical join reason. New confidence
failures retain the inference reason (for example model_unavailable,
required_feature_columns_missing or inference_failed). Older rows without a
recorded cause say historical_reason_not_recorded. The readiness database query
reads the confidence reason only; it does not transfer the full inference payload.

The opportunity freeze logs attempted scoring, scored and missing counts plus
reason counts. These are scoring attempts, not confirmed inserts: retries still
use the existing immutable ON CONFLICT DO NOTHING behavior. Scoring exceptions
leave confidence null and freeze an explicit unavailable record. Historical
backfills do not calculate confidence with today's model.

Frozen signals now keep the AI confidence prediction evidence produced by the
actual inference: artifact checksum, registry/version identity when available,
inference time, inputs and calibration identity. The freeze does not reload a
model to infer which artifact was used. This adds evidence to new rows only;
existing frozen records and scores are not rewritten.

Admin → Models distinguishes the configured database confidence model from the
latest saved confidence inference. A configured active model is not proof of
which artifact a worker loaded: local fallback exists. Missing historical hashes
and times stay unknown. The production summary uses SELECT-only metadata queries
and never loads model bytes or ML libraries. Challenger training records list
training time, dataset, artifact registration, evaluation evidence and registry
stage. No registered record does not prove that an unregistered training run
never happened. Shadow eligibility does not establish successful shadow scoring;
consult the shadow results and monitoring below the summary.

## Once live readiness passes

1. Finalize and review a fresh immutable dataset using the existing research
   process. An older finalized dataset retains the gaps it had when frozen.
2. Open the manual ML v4 Train & Evaluate workflow, select that exact dataset
   version, and run preflight. The selected snapshot and actual target must pass.
3. Train the providers on the same examples. Retain the generated artifacts.
4. Review development evaluation first. Request the reserved final test explicitly;
   its existing durable ledger and opt-in protections remain in force.
5. Register reviewed candidates and move them into shadow through the existing
   lifecycle controls. Review new predictions and matured results.
6. Consider promotion only after the forward gates pass. Registry champion status
   does not connect the model to production AI Confidence; serving is a separate
   reviewed change.

The UI checklist and links do not dispatch workflows, finalize datasets, promote
models, or change customer-visible scoring.
