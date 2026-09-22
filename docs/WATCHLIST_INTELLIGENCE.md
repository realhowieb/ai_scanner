# Watchlist Intelligence 2.0 (Run 42)

Turns the watchlist from a saved ticker list into a personal market-monitoring
dashboard: which symbols need attention, what changed, why, which setups
strengthened/weakened, and which symbols have nothing important happening.

**Reuses Run 40** (`analytics.opportunity_view`) — no parallel scoring engine, no
Opportunity Score, no DT/model/scanner change. Alert Priority is an attention
signal, never a return prediction. Consistent with the Run 41 Market Brief
watchlist section (both feed the same Run 40 engine).

## Architecture

`USER → watchlist (db.watchlists) → tickers → one batched build_day_trader_metrics
fetch → analytics.watchlist_view → WatchlistSymbolView → ui.watchlist_intelligence_feed`.

Opening a watchlist triggers **one batched metrics fetch** (not per-symbol, not a
full-market scan). The existing storage/management (`db.watchlists`,
`analytics.watchlist_intelligence`, `ui.personal_watchlist`) is untouched; the
Run 40 intelligence is mounted **additively** at the top of the watchlist page.

## WatchlistSymbolView (`hsf-watchlist-view-1.0`)

A Run 40 OpportunityView plus watchlist identity:
```
… all OpportunityView fields (symbol, price, change_pct, primary_setup,
   secondary_setups, direction, alert_priority, priority_reason, lifecycle_state,
   scanner_count, scanner_names, positive_reasons[], risk_reasons[],
   changes_since_prior[], freshness, data_quality, is_watchlist) …
+ schema_version, watchlist_id, watchlist_name, rvol,
+ no_active_setup, no_setup_facts[]
```
`observation_from_market_row` adapts a metrics row → canonical observation;
`derive_row_scanners` maps indicators to triggers with meaningful thresholds
(RVOL ≥2 → unusual_vol, chg ≥1% & not below VWAP → momentum, |gap| ≥0.5% → gap)
so a quiet symbol is a valid **no active setup** state, not a false trigger.

## Every symbol stays visible (Task 3)

Unlike the Opportunity Feed ("what deserves attention?"), the watchlist answers
"what is happening to everything I care about?" — no-setup symbols render with
`no_setup_facts` (e.g. "+0.4% today · Above VWAP · RVOL 0.9x · No meaningful
change since prior scan").

## Attention sorting (Task 4)

`attention_sort`: active setups before no-setup; within active, by Alert Priority
then lifecycle (NEW/STRENGTHENING highest, WEAKENING surfaced above plain ACTIVE)
then scanner count. User sorts: Attention, Ticker, % Change, RVOL, Priority,
Setup.

## What Changed & lifecycle (Tasks 5–6)

Change detection and NEW/ACTIVE/STRENGTHENING/WEAKENING/RESOLVED come straight
from Run 40 (`detect_changes`, `lifecycle_state`) using the prior render's rows
(kept in `st.session_state["_watchlist_prior_rows"]`). No prior → graceful
current-state presentation (NEW/ACTIVE, no fabricated history).

## Why Showing / Caution (Task 7)

Reuses Run 40 `positive_reasons` / `risk_reasons` — evidence-based, present-data
only. No generic AI prose.

## Summary & filters (Tasks 8, 16)

`watchlist_summary`: total, needs_attention, new_setups, strengthening, weakening,
no_active_setup. `filter_watchlist`: All, Needs Attention, High Priority, New,
Strengthening, Weakening, Active Setups, No Setup, Bullish, Bearish.

## Activity feed & Since last visit (Tasks 9–10)

`activity_feed` lists detected `changes_since_prior` events per symbol (never
manufactured). There is **no reliable per-user last-visit timestamp store**, so
this run does not build one; the feed is "recent changes since the previous
observation," documented as such.

## Multiple & default watchlists (Tasks 12–13)

Intelligence is symbol-based; membership stays per-watchlist via the existing
`db.watchlists`. The same symbol in multiple lists is observed once (no duplicate
market observation). Default-watchlist behavior is unchanged (existing
`db.watchlists` logic); the intelligence layer reads all of a user's tickers.

## Market Brief consistency (Task 11)

Market Brief's "★ Your watchlist" (Run 41) and this page both derive lifecycle /
priority / reasons from the **same Run 40 engine**, so classifications agree.
Market Brief shows the summary; the Watchlist page shows the full personal view.

## Session & freshness (Tasks 20–21)

`session` is passed through to the view; card freshness (Fresh / Partial / Stale)
comes from Run 37/36 data-quality metadata and shows only when not Fresh. Closed/
after-hours data is presented as the latest observation, not implied live.

## Failure isolation (Task 26)

Every layer is guarded: DB/metrics unavailable → empty message; one bad row →
skipped (`build_watchlist_views` try/except per row); AI not used here; the whole
mounted block is wrapped so the basic watchlist below always renders.

## Analytics separation (Task 28)

Views carry only market fields + watchlist id/name — no PII. User-product events
(add/remove/lifecycle) are intentionally kept separate from the canonical market
research dataset.

## Non-goals honored

No Opportunity Score, no DT change, no scanner-threshold/trigger change, no ML
retrain, no buy/sell language, no predictive claim from Alert Priority or scanner
agreement, no Streamlit rewrite.

**Historical Replay (Run 43):** each watchlist symbol has a "📽️ Signal Timeline" replay of its session history (`analytics/replay.py`, `ui/historical_replay.py`) — reuses the same Run 40 engine, with strict point-in-time integrity and outcomes shown separately. See [HISTORICAL_REPLAY.md](HISTORICAL_REPLAY.md).
