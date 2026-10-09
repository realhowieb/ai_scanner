# ML v3 label audit

## Current production labels

| SYSTEM | LABEL | SOURCE |
|---|---|---|
| HSF Score | none: a fixed heuristic (signals + max(BreakoutScore, PreBreakout %) + momentum - fading), never trained | ui/opportunities.py score_breakdown |
| PreBreakout (prebreakout-xgb-v9, active) | FutureQualitySetupHit: a high-quality setup (setup score >= 8) appears within 3 sessions for a below-20d-high candidate | ml_prebreakout.add_prebreakout_target_label |
| AI Confidence (ai-confidence-xgb-v1, active since 2026-09-09) | ForwardReturnHit: +4% before -2% within 5 trading days (fallback return_5d >= 4%) | ml_prebreakout.add_forward_return_labels |
| Outcome Intelligence (what users see) | win = return_h > 0; benchmark beat = excess_h > 0 | analytics/outcome_intelligence.metrics |

None of the production model labels is what Outcome Intelligence reports to users. PreBreakout predicts a future scanner state, not a price outcome.

## Candidate targets on the frozen research dataset (walk-forward, purged)

| LABEL | H | N | POS RATE | FOLDS | HSF SCORE AUC (mean / pooled) | LOGISTIC AUC (mean / pooled) | XGB SAFE AUC (mean / pooled) | DEFINITION |
|---|---|---|---|---|---|---|---|---|
| A_return_gt_0 | 1 | 134 | 55.2% | 2 | 0.390 / 0.454 | 0.301 / 0.302 | 0.418 / 0.426 | return_h > 0 (Outcome Intelligence win) |
| A_return_gt_0 | 3 | 134 | 53.0% | 1 | 0.414 / 0.414 | 0.470 / 0.470 | 0.500 / 0.500 | return_h > 0 (Outcome Intelligence win) |
| A_return_gt_0 | 5 | 134 | 55.2% | 1 | 0.507 / 0.507 | 0.493 / 0.493 | 0.500 / 0.500 | return_h > 0 (Outcome Intelligence win) |
| B_return_ge_4pct | 1 | 134 | 14.9% | 1 | 0.484 / 0.484 | 0.441 / 0.441 | 0.500 / 0.500 | return_h >= +4% (production models' upside threshold) |
| B_return_ge_4pct | 3 | 134 | 28.4% | 1 | 0.250 / 0.250 | 0.375 / 0.375 | 0.500 / 0.500 | return_h >= +4% (production models' upside threshold) |
| B_return_ge_4pct | 5 | 134 | 30.6% | 1 | 0.510 / 0.510 | 0.407 / 0.407 | 0.500 / 0.500 | return_h >= +4% (production models' upside threshold) |
| C_excess_gt_0 | 1 | 81 | 56.8% | 1 | 0.556 / 0.556 | 0.276 / 0.276 | 0.500 / 0.500 | excess_return_h > 0 (beats SPY; OI benchmark beat) |
| C_excess_gt_0 | 3 | 81 | 49.4% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | excess_return_h > 0 (beats SPY; OI benchmark beat) |
| C_excess_gt_0 | 5 | 81 | 45.7% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | excess_return_h > 0 (beats SPY; OI benchmark beat) |
| D_excess_ge_2pct | 1 | 81 | 28.4% | 1 | 0.512 / 0.512 | 0.125 / 0.125 | 0.500 / 0.500 | excess_return_h >= +2% vs SPY |
| D_excess_ge_2pct | 3 | 81 | 34.6% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | excess_return_h >= +2% vs SPY |
| D_excess_ge_2pct | 5 | 81 | 30.9% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | excess_return_h >= +2% vs SPY |
| E_abs_and_rel_win | 1 | 81 | 49.4% | 1 | 0.394 / 0.394 | 0.259 / 0.259 | 0.500 / 0.500 | return_h > 0 AND excess_return_h > 0 |
| E_abs_and_rel_win | 3 | 81 | 46.9% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | return_h > 0 AND excess_return_h > 0 |
| E_abs_and_rel_win | 5 | 81 | 43.2% | 0 | n/a / n/a | n/a / n/a | n/a / n/a | return_h > 0 AND excess_return_h > 0 |
| F_clean_path_5d | 5 | 134 | 23.9% | 1 | 0.424 / 0.424 | 0.319 / 0.319 | 0.500 / 0.500 | mfe_5d >= +4% AND mae_5d > -2% (risk-adjusted; order-free proxy of the production '+4% before -2%' rule, stricter because it also needs no -2% touch after the +4%) |

Labels C, D and E need SPY benchmark returns, so they are evaluated only on rows that have one (smaller N). No label was chosen using the holdout.

