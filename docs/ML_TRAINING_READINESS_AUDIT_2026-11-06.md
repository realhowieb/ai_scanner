# ML training-readiness audit — November 6, 2026

Audited at approximately 11:24 AM Pacific (19:24 UTC). Production code:
`64c78684d1e8689ffa22f2f985585b2ce9d33926`. This was a read-only production audit: no production-data training,
model registration, dataset finalization, production final-test evaluation, or production
model replacement occurred.

**Recommendation: make ML the next workstream now. Target November 13 for the
fresh-dataset preflight and first development comparison. November 12 after the
late regular-session capture is the earliest conditional window for the current
capture gate; it is not an unconditional training date.**

## Verified findings

| Item | Evidence | Implication |
|---|---|---|
| Live readiness | 16 of 17 gates pass; `ai_confidence_frozen` = 80.0%, requires 95.0% | Current full-cohort training remains blocked. |
| Capture window | 1,100 of 1,375 observations have frozen confidence; 275 missing | All 275 missing values belong to October 12–16. |
| Recent collection | Every observation from October 19 through the audit has confidence; 75 per completed trading day, 50 so far November 6 | Current capture is healthy; future collection must remain complete. |
| Finalized datasets | Only `hsf-ml-2026-10-08-v1`, 1,041 members, frozen October 8 at 21:04:21 UTC | The old dataset cannot acquire newer healthy observations. |
| Real check-only preflight | The October 8 version fails `Frozen dataset fingerprint mismatch` before fitting | Preserve the old version. A reviewed fresh version is required; never override the check or reuse its name. |
| ML v4 registry/shadow tables | `ml_models`, `ml_model_events`, `ml_shadow_predictions` absent | No registered v4 candidate/shadow evidence is present. Unregistered workflow artifacts may still exist. |
| Production model configuration | AI Confidence `ai-confidence-xgb-v1`; PreBreakout `prebreakout-xgb-v9` | These are separate configured models. Database configuration does not prove every process's loaded artifact. |

The readiness audit's generic Diagnostics runner does not install ML libraries;
its `libraries=false` values are not a training dependency failure. The dedicated
ML v4 workflow installed its dependencies and reached dataset verification.
Its job is named `train`, but this invocation used `mode=preflight`,
`register=false`, and `final_holdout=false`; it did not fit a model.

## Conditional capture forecast

This uses the exact application's rolling cutoff: market close 20 trading days
before the check date. It does not substitute calendar days or a 20-row window.
Observed daily counts came from the same Scanner-supersedes-Brief population as
the readiness reader. October 12/13/14/15/16 contribute 15/35/75/75/75 missing rows.
All those observations occurred before their session close, so daily aggregates
are sufficient for this particular cutoff calculation.

Assume 75 new observations per trading day, every new confidence value captured,
and completion of the remaining 25 observations on November 6. Counts below are
projected after the day's last regular-session capture, not observed results.

| Check date | Captured | Window observations | Capture | Gate |
|---|---:|---:|---:|---|
| Fri Nov 6 | 1,125 | 1,400 | 80.36% | Fail |
| Mon Nov 9 | 1,200 | 1,460 | 82.19% | Fail |
| Tue Nov 10 | 1,275 | 1,500 | 85.00% | Fail |
| Wed Nov 11 | 1,350 | 1,500 | 90.00% | Fail |
| Thu Nov 12 | 1,425 | 1,500 | 95.00% | Conditional pass |
| Fri Nov 13 | 1,500 | 1,500 | 100.00% | Conditional pass |

The late regular-session scan is scheduled for 12:35 PM Pacific (3:35 PM ET),
but qualification depends on its actual completed capture, not its scheduled
start. November 12 has no margin under this scenario: fewer observations or a
new missing value can delay passage. November 13 removes the October 16 gaps
from the rolling cutoff. Recheck all gates then; passage is not guaranteed.

## Fresh dataset preparation

Proposed cohort: September 14, 2026 through the actual freeze day. This excludes
the September 12–13 bootstrap burst under the existing readiness policy.
The live audit found 608 bootstrap observations and **zero** earlier opportunity
observations, so this date boundary excludes exactly the burst in today's store,
not an additional historical training population. The proposed population currently
contains 1,834 opportunity observations; this is not the eligible training count.

Readiness already excludes the bootstrap burst, but the generic dataset finalizer
and trainer do not automatically apply that exclusion. Set the start date
explicitly to September 14 when preparing the new version; do not finalize with
the all-history default and assume the readiness exclusion carries into training.

As a coarse completeness check, 1,368 non-bootstrap observations store five-day
returns and MFE/MAE; 369 satisfy the raw clean-path condition. These counts are
observation-level, **not** the certified signal-day training population or class
balance. They cannot replace the actual-target preflight. Final eligible rows,
class balance, training dates and holdout composition remain unverified until a
fresh dataset is built and checked.

Before finalization:

1. Re-run the live readiness audit and require every existing gate to pass.
2. Build the research coverage report for September 14 through the freeze date.
   Inspect point-in-time joins, confidence capture and matured/certified outcomes.
   Keep features separate from labels; do not regenerate historical confidence
   with today's model or lower a threshold to obtain a passing result.
3. Rebuild twice with unchanged inputs/code and require identical membership and
   fingerprint. Review any change before creating a new immutable dataset version.
4. Finalize one new version only after reviewing the concrete report. This step
   writes a dataset registry row; it was not performed in this audit. Record the
   returned version, fingerprint, freeze instant and code SHA in the run manifest.
5. Run the dedicated workflow with that exact dataset version and `mode=preflight`.
   Never leave the dataset blank during a reviewed comparison; `newest` can move.

The current October 8 dataset remains untouched. Its observed fingerprint mismatch
has not been attributed to a single schema/code/data change by this audit; changing
its stored fingerprint would defeat the integrity check.

## Pre-declared development comparison

- Providers: XGBoost, LightGBM, CatBoost.
- Feature sets: `production` and `leakage_safe` (six fixed recipes).
- Primary label: `F_clean_path_5d`: MFE >= +4% and MAE > -2% over five trading days.
  This means reaching +4% while never touching -2% in that window. It is an
  order-free proxy; it does **not** establish which threshold was reached first.
- Sampling unit: certified, matured signal-days, not duplicate intraday readings.
- Identical time splits, rows and fixed provider defaults across the comparison.
- Walk-forward: at least 60 training rows, 30 validation rows, five examples per
  validation class, purging overlapping outcomes, and a one-day embargo.
- Actual v4 target minority-class share >=20%, both classes in training, a valid
  reserved final block of >=100 examples across >=5 entry days, and >=3 usable
  pre-holdout folds. These target-specific checks are additional to the headline
  readiness gates and could still block training.
- Baselines: majority class and frozen AI Confidence as served, on comparable rows.
  Do not fill missing historical served predictions by rescoring them today.
- Development report: ROC-AUC, PR-AUC, Brier/calibration, top-ranked precision/lift,
  returns after the fixed 20-basis-point round-trip cost, and fold stability. Inspect
  existing risk metrics and coverage rather than claiming a trading edge from AUC.

First run `mode=evaluate`, `register=false`, `final_holdout=false` on the reviewed
version and pinned code commit. This fits development models on pre-holdout folds
and produces reports; it does not promote or register production models. Select
one recipe using development evidence and stability before requesting a final test.
If preparation refuses the dataset, resolve its stated blocker without bypasses.

Only after that review: explicitly approve the one-time final holdout evaluation.
It uses a durable claim/report-reuse ledger. Do not peek repeatedly or choose the
winner based on final-test results. Candidate registration is another explicit
step; it requires the appropriate final-test evidence. Shadow scoring follows
candidate review, and any production switch remains a separate decision. Neither
training nor readiness automatically replaces AI Confidence or PreBreakout.

## Work to prioritize

1. Keep collection and maturation healthy through the target window; verify the
   capture trend and all gates rather than waiting for a date alone.
2. Review the fresh dataset's membership and reproducibility, with the bootstrap
   exclusion explicit. This is the immediate next dataset task.
3. Run the fixed development comparison, then decide from the evidence whether
   any candidate deserves a final test and shadow run. More scanner indicators
   are not needed before completing this ML workstream.

The date forecast is conditional, not an automatic training trigger. The next
production step remains blocked until the live gates and fresh-dataset preflight
pass; no dataset or model writes were performed in this audit.

## Evidence and artifacts

- [Read-only live readiness audit](https://github.com/realhowieb/ai_scanner/actions/runs/42279306067)
- [Blocked check-only frozen-dataset preflight](https://github.com/realhowieb/ai_scanner/actions/runs/42279425462)
- Local machine-readable audit and daily aggregates:
  `artifacts/research/ml_training_audit_2026-11-06/audit.json` and `capture_daily.json`.
- Downloaded preparation failure report:
  `artifacts/research/ml_training_audit_2026-11-06/ml_v4_preflight.md`.
- [Neon Postgres skill used for the read-only database audit](/Users/realhowieb/.agents/skills/neon-postgres/SKILL.md).

## Preparation validation

59 isolated tests passed covering preparation guards, all three provider adapters,
checksummed artifact round trips, development evaluation, and the durable holdout
ledger. Test-fixture model fits and temporary test ledgers are not production
training, production holdout consumption, or model-quality evidence.

A draft run manifest is saved at
`artifacts/research/ml_training_audit_2026-11-06/proposed_run_manifest.json`.
It pins the inspected code and current fixed provider parameters, features, target,
folds and cost assumptions. Dataset version/fingerprint remain null until an actual
reviewed finalization; the manifest explicitly remains blocked and undispatched.
