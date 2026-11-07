# PreBreakout audit — November 6, 2026

Scope: active repository training/evaluation entry point and recent GitHub workflow evidence. No training workflow was dispatched and no production model was replaced.

## What the model predicts

The standard target is `FutureQualitySetupHit`: eligible candidates have not broken out, have BreakoutScore below 8, and sit below their 20-day high. A positive requires a high-quality setup in the next 1–3 business days, followed by a +4% price-path hit before a −2% stop within five bars. Same-bar ties favor the stop. This is a compound setup-and-outcome target, not simply tomorrow's positive return.

## Evidence and fixes

- The latest listed workflow, [37710066771](https://github.com/realhowieb/ai_scanner/actions/runs/37710066771), ran October 7 Pacific (October 8 UTC). Its output is a **serving-skew audit**, not evidence of new training. It identifies `prebreakout-xgb-v9`, trained `2026-09-10T16:46:21.140042Z`.
- The current workflow calls `train_prebreakout_model_run17`. Run 17 produces diagnostic classifier/ranker comparisons and explicitly returns `NO_PROMOTION`; it does not save a new champion. Before this change, it could nevertheless invoke the Run 16 rollback helper, which changes the active production row. That implicit rollback is removed from Run 17.
- Run 17 used a five-day purge, or the maximum of setup lead and outcome horizon. These are sequential windows: 3 + 5 for the standard target, up to 10 + 5 for the longest experiment. Run 17 now sums them for classifier and ranking evaluations, and records the standard eight-day gap in its report. Historical evaluation scores are not retroactively corrected.
- Removed stale Run 17 messages claiming cached close returns can supply path labels. Complete OHLC is required by the existing labeling code.

## Next evidence to collect

1. Rerun the serving-skew audit against current main after review. The October audit reported stored-score AUC 0.5227 versus reconstructed training-feature AUC 0.7039, with some live delta/slope features entirely zero. These are old diagnostics under old code/labels, **not current performance estimates**. Check whether recent captured inference features resolve the mismatch before fitting another model.
2. Inspect candidate label maturity and observation completeness. Candidates require a complete setup-search window as well as outcome windows; a missing future scan must not silently mean a negative target. The present patch does not establish that invariant.
3. Review scan-time versus daily-close availability in historical OHLC enrichment and label entry selection. Business-day counts also omit exchange holidays. The summed purge is consistent with current counting, not an exchange-calendar migration or a full point-in-time certification.
4. Run the corrected report only after those inputs are trustworthy. Require all five valid folds; compare ranking, calibration and economic results on the same cohort. Cross-target AUC improvements alone do not justify changing the target.
5. Promote only through a separately reviewed production-model change. The ML v4 confidence-capture gate is a separate readiness mechanism; reaching it does not train or promote PreBreakout.

Validation: all 98 `test_prebreakout*.py` tests pass locally, including new tests for every target's full purge window and report dispatch without production rollback/save calls. These tests use synthetic data and mocked external dependencies; no production fitting occurred.
