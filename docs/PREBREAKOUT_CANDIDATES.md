# PreBreakout candidates (November 2026)

The PreBreakout model learns from stocks that are **not yet setups**: no breakout,
BreakoutScore below 8, and below the 20-day high (`prebreakout_candidate_mask`).
The live audit on 2026-11-12 found that every stock the scan saved already scored
12 or more, so the PreBreakout tab ranked names outside the model's population
(AUC 0.51; top 10 hit 23% against a 25% average). On real candidates the same
model ranks with AUC about 0.7.

## What runs now

- **Scheduled scans** (recorded scans only) rank every liquid row, keep the usual
  top 100 as results, and pass the 300 best-scored near-setup names to
  PreBreakout. Scores are saved in `prebreakout_candidates` (one row per scan,
  180-day retention). Kill switch `HSF_PREBREAKOUT_CANDIDATES=0`; count
  `HSF_PREBREAKOUT_CANDIDATES_N`.
- **PreBreakout tab, Brief picks and PreBreakout scan alerts** read the latest
  candidates (top 10 by raw model score). Scans without stored candidates fall
  back to the old behavior.
- **The % is hidden on names that are already setups** (BreakoutScore of 8 or
  more) in every display copy. The saved scan and the HSF score are unchanged.
- **PreBreakout Weekly Check** (Mondays 14:30 UTC, read only) reports top-10 hit
  rate, base rate and AUC on matured picks, for stored candidates and for the
  scan's top 100. Script: `scripts/diag_prebreakout_live_audit.py`.
- **Retrain**: `Train PreBreakout Model` with `mode=challenger` trains a fresh
  model on matured candidates and compares it with the live champion on a later
  holdout, under per-scan and full-contract inputs. Report only; promotion is a
  separate reviewed step. On 2026-11-12 only 282 matured candidates existed
  (Sep 14 to early Oct), too few to train; stored candidates fill this in a few
  weeks.
