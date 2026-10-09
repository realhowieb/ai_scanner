# Prospective Provenance Readiness Hardening

## Changes

Brief picks now mark their actual inference as `brief_pick_recalculation` and
carry a parent prediction's inference timestamp, source scan identity and exact
provenance digest where unambiguous captured source evidence exists. This is
attached after the existing single inference call; probability/ranking and
actual inference timestamps are unchanged. Source scan identity explicitly
means input lineage, not the time of recalculation. No full recalculation archive
or database migration is introduced. Existing applicable freezing carries this
metadata; unselected/unfrozen recalculations still have no independent archive.

Training cutoff, calibration cutoff and target version each carry an explicit
availability status/reason from loaded metadata. Values already supplied are
preserved. Missing fields remain null and are never inferred from trained_at,
the current registry, scan time or model filenames.

## Verification

Five new tests cover actual pick-path single inference, unchanged probability,
parent digest/time linkage, absent parent, missing boundary handling and supplied
boundary preservation. Focused regression suite: **59 passed**, no failures,
skips or warnings. Ruff and syntax checks pass for modified files.

## Remaining gates (not claimed fixed)

- Live customer-tier/API/table/CSV privacy requires authorized deployed access;
  no such access was used. Existing source redaction and fixture tests remain
  distinct from live verification.
- These prospective annotations require review/deployment and inspection of a
  subsequent applicable freeze to demonstrate production recalculation linkage.
- Current loaded artifacts do not supply verified training/calibration cutoffs
  or target versions. This change reports that gap, not repairs historical
  metadata. Any recovery must bind authoritative training evidence to the exact
  loaded artifact. Retraining and registry mutations are not authorized here.
- Original-target evidence remains incomplete. No target maturation, inferred
  negatives from top-N absence, historical backfill or proxy performance metrics
  are introduced. Capturing provenance cannot create missing future evidence.

No production scan, inference or mutation was triggered. Models, predictions,
scoring, ranking, target definitions and certified capture semantics remain
unchanged. Collection verification remains PARTIAL and evaluation NOT READY.
Implementation is isolated on `codex/provenance-readiness-hardening`; do not merge
or deploy automatically. The branch includes the existing verification history.
