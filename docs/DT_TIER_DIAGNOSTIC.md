# DT quality-tier diagnostic

Production DT Score remains v1. The rejected v2 candidate (`a6820a9`) is
replayed only inside `analytics/day_trade_tier_diagnostic.py`. The diagnostic
uses the same historical observation inputs as the existing DT validation job;
future returns are excluded from gate and score calculations.

Run the held-out window through the existing `Validate DT Score` GitHub Action
with `start=2026-07-13`, `end=2026-08-01`, the original held-out symbol list,
and the original `sample_every` value. The job uploads
`dt_tier_diagnostic.json` and `dt_tier_diagnostic.md` in the existing
`dt-score-validation` artifact. Compare the directional count to 9,248 before
interpreting its gate counts. A different universe, provider feed, bar coverage,
or sampling interval is a different dataset.

For locally available credentials and the same inputs:

```sh
DTV_SYMBOLS='...' DTV_START=2026-07-13 DTV_END=2026-08-01 \
  python -m scripts.validate_day_trade_score --out artifacts
```

For a saved JSON array of historical observations containing `diagnostic_inputs`:

```sh
python -m scripts.diagnose_dt_tiers --observations observations.json \
  --profile rejected_v2 --out artifacts
```

The output includes per-observation scores, evidence, conflicts, and gate
results, plus aggregate funnel, conflict, agreement, confirmation, sensitivity,
and 77.1-score pileup tables. A `NO_OBSERVATIONS` or
`INCOMPLETE_OR_MISMATCHED_INPUTS` status blocks causal conclusions. Aggregate
validation reports alone cannot reconstruct the missing indicator values.
