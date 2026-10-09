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
