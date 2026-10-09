# ML v3 Coverage and Freezing Gaps

## Initial diagnosis (before changes)

Continue on `codex/provenance-post-deployment-verification`, existing PR #60.
Evidence: live scanner run 37964124080 at 62b7d09; read-only audit 37965515898.
The 200 saved rows are storage representations, not yet proven unique inference
events. Canonical cohorts must be reconciled before treating 150 absent model
records as defects.

Source inspection: scheduled candidates are scored before canonical capture;
NEAR_MISS and CONTROL observations are created separately without ML inference.
AI confidence is not invoked on the scheduled engine path. No extra inference
will be added to increase those denominators.

Freezing path: cron → freeze_latest_opportunities → _compute_brief →
build_opportunities → freeze_opportunities → raw_signal.models. Current Brief
only forwards models for the three PreBreakout picks. Other high HSF-ranked
opportunities may have an existing saved prediction but receive no provenance
when they enter from top-setups/gainers/gappers. This is a prospective forwarding
gap, not permission to rescore or overwrite prior frozen history.

Production reconciliation and test results will be appended after bounded
read-only inspection. No scoring, selection, model, cadence or target change
is proposed.

Additional privacy finding before forwarding changes: `api.market.brief` copies
all pick fields except symbol and returns redacted opportunities without removing
their models object. The scanner/stock APIs are allowlisted, but Brief is not.
Internal models must be removed at this customer boundary for every tier.

## Verified production reconciliation

Bounded read-only Diagnostics run **37966480000**, script commit `3cfc1b5`,
inspected records from successful scanner run **37964124080** (executed SHA
`62b7d09a439f5ee313e80e3d46079f50a9778966`). Sanitized aggregate evidence is
in `ml/reports/provenance_coverage_freezing_gaps.json`.

- Saved run 3228 and daily snapshot 3227 contain 100 rows each: **100 unique
  inference events**, zero conflicting duplicate identities.
- CANDIDATE: 100/100 PreBreakout records captured and exactly preserved.
- NEAR_MISS: 50 and CONTROL: 100 have no inference by design. Their absence
  is expected, not 150 lost predictions. Scheduled AI-confidence is not invoked.
- Five frozen records (11272 through 11276) were newly created during the scan,
  not older first-write records. All lacked model evidence. Their tickers exist
  in saved candidates, but ticker presence alone is not exact inference linkage.
- Captured feature/mask lengths, probability ranges and hashes passed the
  original audit. Actual input timestamps, target versions and training/
  calibration boundaries remain unavailable; no substitute dates are invented.

## Prospective implementation

`analytics.prediction_provenance.models_by_ticker` forwards existing evidence
from one source result set. Identical duplicates collapse; multiple distinct
events for one ticker/role explicitly become unavailable rather than guessed.
`ui.market_brief._compute_brief` carries that map into
`ui.opportunities.build_opportunities`. Recalculated picks replace only roles
actually inferred, preserving their real inference timestamps and other roles.
`ui.results_intelligence` also preserves the map during result consolidation.

Production call path remains scheduled cron -> freeze_latest_opportunities ->
_compute_brief -> build_opportunities -> freeze_opportunities -> freeze_signal
-> raw_signal.models. First-write uniqueness and DO NOTHING remain unchanged.
Saved evidence does not create a displayed probability or alter HSF components.

Public Brief picks and all entitlement tiers now strip internal model evidence
through `customer_opportunity`. This corrects the previous overly broad claim
that every public API response was already allowlisted.

No inference, model artifact hashing, provider call or database query is added.
The forwarding map requires linear traversal plus evidence digesting and stores
references to existing records. Payload growth is limited to opportunities that
already have evidence; no runtime timing is claimed.

## Tests and deployment limits

Nine focused tests cover deduplication/cohorts, ambiguous events, actual Brief
assembly, freezing forwarding, recalculation timestamps, consolidation, public
redaction, legacy inputs and immutable first-write behavior. The focused
regression selection passed **355 tests**, with one existing NumPy warning.
Modified Python files pass Ruff, syntax compilation and diff whitespace checks.
Final full-suite results are recorded below.

Final `.venv/bin/python -m pytest tests -q`: **2947 passed, 1 failed,
24 skipped, 10 warnings, 324 subtests passed** in 129.40 seconds.
The existing `ListRunsScopeTests.test_every_caller_scopes_its_runs` failure
comes from scanning the unrelated untracked nested `ai_scanner/tests` checkout
as application source. Neither that checkout nor the assertion was changed.
Warnings are existing Starlette/httpx and NumPy timedelta deprecations.
The sanitized machine-readable report reloads as strict JSON successfully.

This branch has not been merged or deployed. Prospective forwarding is ready
for review, but production preservation after this fix is **UNVERIFIED** until
a normal approved post-deployment run. Evaluation readiness remains **NOT READY**:
original target evidence and training/calibration boundaries are still incomplete.
No historical backfill, production migration, scan trigger, target maturation,
model tuning, score/rank change or extra cohort/AI inference was performed.
