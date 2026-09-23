# Intelligence consolidation (Run 49)

One canonical, explainable, consistent presentation model over HSF's **existing**
intelligence — **without changing how HSF trades or ranks**. Run 49 is an
adapter/view layer; it creates no new score and recalibrates nothing.

> Result target: **same intelligence, better structure, better explanations,
> better consistency, better frontend readiness.**

## Intelligence inventory

| Concept | Source | Range/Type | Affects ranking? | Affects alerts? |
| --- | --- | --- | --- | --- |
| Opportunity Score | `ui.opportunities.build_opportunity_score` (`HSF_SCORE_VERSION`) | 0–100 | opportunity feed ordering | via priority | 
| PreBreakout | `ml_prebreakout` (`prebreakout-xgb-v16`) | 0–100% (calibrated) | scan/brief | picks | 
| Day Trader | `analytics.day_trade_intel` (coherence, research CLOSED) | 0–100 + direction/quality | Day Trader view | no | 
| ML probability (AI Confidence) | `scan.ai_confidence` (`ai-confidence-xgb-v1`) | 0–1 | ranking column | no | 
| Alert Priority | `analytics.opportunity_view` | HIGH/MEDIUM/LOW | feed order | yes (attention only) | 
| Direction | scanner votes (`opportunity_view`) | LONG/SHORT/NEUTRAL/MIXED | no | context | 
| Lifecycle | `opportunity_view.lifecycle_state` | NEW/ACTIVE/STRENGTHENING/WEAKENING/RESOLVED | no | transitions | 
| Tier | opportunity status (STRONG/WATCH/CAUTION) | enum | no | no | 
| Agreement | scanner_count (`opportunity_view`) | int | context | context | 
| Confirmations/Conflicts | `opportunity_view` reasons/risks | lists | no | context | 
| Market regime | `ui.opportunities.classify_market_regime` | enum | no | context | 
| Scanner rank | breakout `BreakoutScore` order | int | yes | no | 
| Replay context | `analytics.replay` (point-in-time) | timeline | no | no | 

**Already shared:** Market Brief (Run 41), Watchlist (Run 42), and Historical
Replay (Run 43) all derive direction/lifecycle/priority/reasons from the same
`opportunity_view` engine — so those surfaces are already semantically consistent.
Run 49 formalizes this into one `IntelligenceView`.

## Canonical view model (`analytics/intelligence_view.py`, `hsf-intelligence-view-1.0`)

`build_intelligence_view(obs, *, prior_obs, opportunity_score, tier,
research_evidence)` → symbol/timestamp; **direction/lifecycle/tier**; **scores**
(opportunity_score, prebreakout, ml_probability, alert_priority — surfaced
side-by-side, **never combined**); **agreement**; **confirmations/conflicts**
(structured factors); **supporting/caution_factors**; primary_setup;
changes_since_prior; next_confirmation; **scan_health** (with an explicit "not
prediction confidence" note); **research_evidence** (level + promotion gate);
freshness; **staleness** (intelligence_as_of / market_data_as_of / scan_run_id);
**provenance** (versions). `to_dict()` is deterministic + null-safe. It reuses
`opportunity_view` — no reinterpretation that changes production semantics.

## Terminology map

`bullish→LONG`, `bearish→SHORT`, `strong/high→STRONG`, `watching→DEVELOPING`,
`confirming→CONFIRMING`, `warning→CONFLICT`, `supporting factor→CONFIRMATION`.
Distinct concepts are **not** merged (e.g. CONFIRMING ≠ CONFIRMED).

## Explainability & provenance

Every factor is `{code, label, source, value, severity}` and is emitted **only**
when a backing value exists (e.g. `RVOL_EXPANSION` ← `indicators.rvol`,
`PREBREAKOUT_MODEL` ← `models.prebreakout.probability`). No hard-coded prose
disconnected from state; tests verify factors trace to real values and that
absent inputs emit nothing.

## Score clarity

`SCORE_METADATA` documents each number (label, range, higher-means, source,
version key, calibrated/predictive flags). Alert Priority is explicitly labeled an
**attention signal, not a return/probability claim**. **No composite "HSF Score"
is created.**

## Lifecycle & tier

Lifecycle states (NEW/ACTIVE/STRENGTHENING/WEAKENING/RESOLVED) and their
transitions are documented (unchanged from Run 40). Tier is surfaced from existing
status only; where earlier work flagged weak tier separation, Run 49 **preserves**
behavior and exposes it as **UNVALIDATED** rather than presenting it as validated.
No thresholds changed.

## Confirmations vs conflicts / agreement

Confirmations = structured supporting factors; conflicts = structured caution
factors — same underlying `opportunity_view` conditions, deduplicated and
consistently formatted. Agreement = scanner_count; documented as **derived from
existing scanner votes**, not an independent predictive model.

## Scan health vs prediction confidence

`scan_health.coverage_health` (Run 45) is carried with an explicit note: it
describes the **scan's** market coverage, **not** the individual stock's signal
reliability. A DEGRADED-scan candidate is distinguishable from a HEALTHY-scan one.

## Research evidence status

`research_evidence.level ∈ {UNVALIDATED, PRELIMINARY, MODERATE, STRONG}` with the
**evidence promotion gate** (Run 48 framework): INSUFFICIENT/UNVALIDATED → no
production recommendation; PRELIMINARY → hypothesis; MODERATE → controlled
experiment; STRONG → production-change proposal. Fixture/synthetic results are
**never** shown as live evidence; default is UNVALIDATED.

## Run 48 readiness check

`run48_readiness(observations)` (reuses the Run 48 evidence framework, no second
statistics): trading days, cohort counts, matured paired outcomes per horizon,
healthy records, evidence level, promotion gate, and a recommendation
(CONTINUE_ACCUMULATING vs RERUN_RUN48). Current live state → INSUFFICIENT →
CONTINUE_ACCUMULATING.

## Surface consistency (audit)

Scanner, Watchlist, Alerts, Market Brief, and Historical Replay already share the
`opportunity_view` engine for direction/lifecycle/priority, so their canonical
state agrees for the same observation. Run 49 does **not** rewrite these surfaces
(no risky mass migration); it provides the single view they can consume
incrementally. Replay remains strictly point-in-time (Run 43) — the view is built
from the historical observation, never recomputed against today.

## Unknown/empty states

Unknown is `None` (→ "Unknown"/"Unavailable" in UI), never 0/False/Neutral, so the
product never implies information it lacks.

## Known limitations

- Surfaces are not yet wired to consume `IntelligenceView` (documented incremental
  path; they already share the engine, so semantics are consistent today).
- Tier validation and signal effectiveness remain UNVALIDATED pending Run 48 data.
- Day Trader coherence is surfaced but excluded from ranking (research CLOSED).

## Run 50 readiness

HSF now has one canonical, serializable, provenance-carrying presentation model
with a documented API contract (`INTELLIGENCE_API_CONTRACT.md`) — ready for a
production-readiness audit / future API/Next.js frontend without changing product
semantics.
