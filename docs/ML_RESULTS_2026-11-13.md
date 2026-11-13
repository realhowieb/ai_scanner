# November 13 ML experiment: completed, production unchanged

The reviewed Scanner cohort was frozen, all six challengers trained, and development and final evaluations completed. The development-selected LightGBM production-input candidate does not provide convincing evidence of improved ranking or probability quality. Do not advance it to shadow promotion on these results.

## Reproducibility

- Dataset: `hsf-ml-2026-11-13-v1`, 3,204 observations; cohort starts September 14. Trainer reload reproduced its fingerprint.
- Fingerprint: `sha256:9e0d5c83aee4512c5080be858b4d0a54f4de4e5156f50fccc5477afca8d073fa`.
- Code: `c9768f54e077cfb3e9150e23d599cc762fcf67a9`, pinned at `codex/ml-freeze-2026-11-13`.
- Target: reach +4% and never touch −2% across five trading days. This is an extrema-based clean-path proxy, not a first-hit ordering label.
- Preflight: 1,452 eligible signal-days, 366 hits and 1,086 misses; 1,003 training examples, 218 purged and 45 embargoed; 14 development folds. Reserved final test: 186 rows from October 30.
- Six artifacts: XGBoost, LightGBM and CatBoost, each with production and leakage-safe feature sets. All downloaded SHA-256 checksums verified. No model registered or production model changed.

## Selection before final access

The November 6 predeclared rule ranked the six single-model recipes by pooled out-of-fold raw AUC, then raw Brier, provider and feature-set tie breakers. LightGBM / production won with development AUC 0.5660 (95% interval 0.4940–0.6259). Its 14 folds were valid. Development top-third mean net five-day return was −1.09% after 20 bps round trip; this is not evidence of a profitable strategy.

Candidate, parameters, data binding and artifact identity were committed and pushed in `625aee8e` **before** the final workflow was dispatched. That lock is preserved in `ml_experiment_lock_2026-11-13.json`. No alternative is selected from final-test results.

## Final result for the locked candidate

| Measure | Result |
|---|---:|
| Final test rows | 186 |
| Target hits / misses | 35 / 151 |
| Raw AUC | 0.5323 |
| AUC 95% interval (entry-day block bootstrap) | 0.4603–0.6098 |
| Raw Brier (lower is better) | 0.2463 |
| Calibrated Brier | 0.1813 |
| Simple baseline raw Brier | 0.1581 |
| Calibrated ECE | 0.1117 |

The AUC interval includes chance. Calibration improves the candidate's Brier but does not beat the baseline. This does not justify replacing a production scorer or claiming improved probabilities.

The saved AI Confidence comparison has no usable development folds. Its final head-to-head challenger is fitted on only 147 recent matched training rows, rather than this selected artifact's 1,003 rows; do not conflate those two fitted models. That comparison also does not establish a challenger improvement.

## Decision and next research

No registration, shadow enrollment, champion change or model replacement was performed. Preserve these artifacts and this negative result. The final test was consumed once under its durable ledger claim; do not reset it, tune against it, or claim a later run on the same rows is unseen evidence.

Next: audit feature availability/defaults and temporal drift on development data, investigate why broader features degraded out-of-fold results, and collect a genuinely new forward test cohort before another model-selection cycle. Keep production behavior unchanged while that research proceeds.

## Workflow evidence

- [Freeze and fingerprint reload](https://github.com/realhowieb/ai_scanner/actions/runs/43460876949)
- [Preflight](https://github.com/realhowieb/ai_scanner/actions/runs/43461299881)
- [Six model artifacts](https://github.com/realhowieb/ai_scanner/actions/runs/43461579653)
- [Development evaluation](https://github.com/realhowieb/ai_scanner/actions/runs/43461865127)
- [Single final evaluation](https://github.com/realhowieb/ai_scanner/actions/runs/43462622239)

Artifacts are also retained locally under `ml/artifacts/2026-11-13/` (ignored by Git). GitHub workflow artifacts expire after 30 days. Aggregate final results and the durable ledger claim are recorded in `ml_experiment_result_2026-11-13.json`.
