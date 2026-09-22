# Historical Replay & Signal Timeline (Run 43)

Reconstructs how HSF's view of a symbol **developed** over a session from the
canonical observation history (Run 36), giving HSF memory rather than only a
current snapshot. It answers: what did HSF know at time T, when/why the setup
changed, how signals evolved, and — strictly separated — what happened afterward.

**Reuses Run 40** (`analytics.opportunity_view`) for state/reasons/lifecycle — no
new signal engine, no Opportunity Score, no scanner/model/DT/Alert-Priority
change. Not a backtest, simulator, or P&L tool.

## Architecture

Pure engine `analytics/replay.py`; thin surface `ui/historical_replay.py`;
bounded loader `db.hsf_observations.load_observations_for_symbol(symbol, limit)`.
Replay consumes **persisted observations only** — no market scan, no model
inference, no scanner recomputation (Task 25). Reachable from Watchlist
Intelligence ("📽️ Signal Timeline").

## Data source & coverage

Scheduled production observations (`source="scheduled"`) captured since Run 38A.
`available_dates(observations, symbol)` lists the sessions that actually exist;
the UI states the real range ("replay available for N session(s): X → Y") and
never implies coverage before capture began. Early history is expected to be thin
and the UI degrades gracefully.

## ReplaySession / ReplayEvent (`hsf-replay-1.0`)

`build_replay_session(symbol, date, observations, *, outcomes=None,
source="scheduled", min_completeness, allow_stale, allow_fallback)` returns:
```
symbol, date, session,
observations[]  — per-timestamp OpportunityView + _feature_snapshot + market_regime,
events[]        — {timestamp, event_type, importance, title, details},
timeline[]      — collapsed stable spans + event entries,
outcomes_by_timestamp, outcomes_available,   # SEPARATE from state
coverage, summary
```

## Event extraction & importance (Tasks 4–5)

Events are derived from Run 40 change-detection between **consecutive**
observations plus lifecycle/priority transitions:
- **MAJOR:** SETUP_APPEARED, SETUP_RESOLVED, PRIORITY_INCREASED→HIGH,
  DIRECTION_CHANGED.
- **NOTABLE:** SCANNER_ADDED, VWAP_CROSS_UP/DOWN, RVOL_CHANGE,
  LIFECYCLE_STRENGTHENING/WEAKENING, PRIMARY_SETUP_CHANGED, priority changes.
- **INFO:** PREBREAKOUT_CHANGE, SCANNER_AGREEMENT_CHANGE.

Importance is **presentation** importance, not predictive significance. Only
meaningful changes (Run 40 thresholds) become events — tiny wiggles do not.

## Stable-period collapsing (Task 6)

`collapse_stable` folds consecutive observations with no events into a single
span ("10:05–10:30 ACTIVE — No meaningful setup change"), keeping the raw
observations available underneath.

## Point-in-time integrity (Task 3 — the cardinal rule)

Each observation's view is built with `prior_obs =` the **immediately preceding**
observation only, so it can contain nothing from the future. `state_at(session,
T)` returns the last view with timestamp ≤ T. Outcomes are never consulted for
state. A dedicated regression test
(`test_replay.PointInTimeIntegrityTests.test_no_future_leakage_regression`) proves
that replaying 10:15 exposes nothing from 10:30/10:45 in primary setup, scanner
agreement, priority, lifecycle, reasons, feature snapshot, direction, or market
context.

## Outcome separation (Tasks 15–16)

Matured outcomes live only in `outcomes_by_timestamp` and are rendered in a
clearly separate "What happened afterward" section (`outcomes_at`). Pending
horizons show **Pending**, never an estimate. Outcomes never feed back into
state/priority/lifecycle/explanations.

## Timeline UX & inspector (Tasks 10–14)

Readable timeline (times, lifecycle, primary setup, top changes) — no raw JSON.
A point-in-time inspector uses the **same** Run 40 explanation logic (Why Showing
/ Caution), shows market regime if it was captured, and offers an optional
feature snapshot (only fields actually captured). A time slider steps through
observations. (Price-chart event markers reuse existing chart infra where
available; not required for the timeline and deferred to avoid rebuilding charts.)

## Session summary (Task 18)

Descriptive facts only: observations, first setup, peak scanner agreement,
priority changes, resolved time, major-event count. Never "best entry", "profit",
or "buy".

## Navigation

Entry point mounted in Watchlist Intelligence (per-symbol "Signal Timeline"
expander). `render_historical_replay(symbol)` also accepts a
`st.session_state["_replay_symbol"]` hook for Market Brief / Opportunity Feed
"Signal History" links (documented mount, kept off primary cards to avoid
clutter).

## Failure states (Task 24)

Guarded throughout: no observations / no dates → honest info message; malformed
observation (bad timestamp, None) → skipped, replay continues; DB unavailable →
empty; stale/degraded excluded by default (source + quality filter). One bad
observation never destroys the replay.

## Limitations

- Only scheduled production sessions since capture began are replayable.
- Scheduled breakout observations are partial-quality (no ADX/VWAP/EWO in that
  path), so early feature snapshots are sparse.
- Outcomes appear only after the Run 38B maturation worker has run.
- Price-chart overlay is deferred.
