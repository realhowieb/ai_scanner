# Market Brief 2.0 (Run 41)

Market Brief is HSF's primary market-intelligence experience — the layer *above*
the scanner tools. It answers, in order: what is the market doing → where is the
action → what changed → which stocks deserve attention → why → what to watch next.

Run 41 mounts the **Run 40 Intelligent Opportunity Feed** into the live page and
adds a deterministic "What Changed" / watchlist-intelligence layer. No scanner,
model, or DT behavior changed; **no Opportunity Score**; Alert Priority remains an
attention signal, never a prediction.

## Page hierarchy (`ui/market_brief.py::render_market_brief`)

| # | Section | Source | Notes |
| --- | --- | --- | --- |
| A | **Market State** | `render_market_header` + `ui.opportunities.classify_market_regime` | regime + SPY/QQQ/IWM + breadth + freshness (pre-existing, reused) |
| A1 | Grounded AI narrative | `_render_claude_narrative` (`_brief_narrative_facts`) | AI summarizes structured facts; page works if AI fails |
| **A2** | **🔔 Intelligent Alerts (Run 40)** | `render_intelligent_alerts` → `analytics.market_brief_view` + `analytics.opportunity_view` | **new this run** — why/changed/risk/priority cards + What Changed + watchlist events |
| B | Since last scan | `render_since_last_scan` | HSF-score movement (pre-existing) |
| C/D | Top Opportunities (movement) | `render_top_opportunities` | pre-existing, complementary |
| E | Watch next | `render_watch_next` | |
| F | Signal scorecard | `render_signal_scorecard` | outcome evidence |
| G | Sector leadership | `render_sector_leadership` | |
| H+ | Secondary detail (toggleable) | gappers/movers/setups/picks/… | demoted below the fold |

## Intelligent Alerts integration (A2)

`render_intelligent_alerts(data, user)`:
1. Reuses `ui.opportunities.build_opportunities(data)` (the same builder the freeze
   pipeline uses) — **no new scan is triggered**; it reads the latest snapshot the
   brief already computed.
2. Adapts each opportunity to a canonical observation
   (`market_brief_view.observation_from_opportunity`) and builds Run 40
   OpportunityViews (`build_top_opportunity_views`, ranked by `rank_feed`, Top 6).
3. Renders compact cards via `ui.opportunity_feed._render_card` (symbol, primary
   setup, priority, price move, top reasons, top risks, changes, lifecycle).
4. Shows **What Changed** (`diff_brief_state`) and **★ Your watchlist**
   (`build_watchlist_events`) when history/watchlist exist.

All logic is in the pure, tested `analytics/market_brief_view.py`; the page is a
thin renderer. The whole block is wrapped so any failure leaves the rest of the
brief intact.

## Market State semantics

Reuses the existing deterministic `classify_market_regime` (SPY/QQQ change,
breadth, sectors) — no new classifier invented. Regime is cached in
`st.session_state["_last_market_regime"]` and fed into What Changed.

## What Changed semantics

`summarize_brief_state` snapshots {high_priority, total, regime, breadth_pct,
by_setup, tickers}; `diff_brief_state` reports **state transitions** vs the prior
render's snapshot (stored in `st.session_state["_brief_prior_state"]`): High
Priority count change, regime change, breadth ±5%+, and up to 3 new opportunities.
Small numeric wiggles are ignored. Empty when no prior exists.

## Session behavior

Market State/freshness come from `_market_phase()` and the snapshot timestamp; the
existing brief already orders secondary sections by phase (afterhours/closed lead
with movers). The Intelligent Alerts feed reads the latest snapshot regardless of
phase; stale snapshots are flagged via the card freshness badge.

## AI grounding

Unchanged and reused: `_brief_narrative_facts` builds a deterministic fact string
(regime, breadth, counts) that the AI narrative summarizes. If AI is unavailable,
the deterministic brief — including Intelligent Alerts — still renders fully.

## Data health / freshness / coverage

Freshness stamp is pre-existing (`_freshness_label`). Run 40 cards surface
per-opportunity freshness (Fresh / Partial / Stale) only when not Fresh. (Subtle
whole-market coverage display from Run 37 artifacts is a documented follow-up —
not wired here to avoid fragile in-session file reads.)

## Failure / empty states

Every optional layer is guarded: no opportunities → real-count empty message
(`opportunity_view.empty_state_message`); no prior state → What Changed omitted;
no watchlist → watchlist section omitted; AI failure → narrative skipped, rest
works; DB/snapshot unavailable → the brief's existing "no snapshot yet" info
message. The Intelligent Alerts block never raises into the page.

## Performance

No new fetch/scan/inference: `build_opportunities` operates on the already-cached
snapshot (`_brief_cached`). Change detection uses in-session prior state, not a DB
round trip. Reuses Run 40's pure functions (no duplicated logic).

## Non-goals honored

No Opportunity Score; DT untouched; scanner thresholds/triggers unchanged;
PreBreakout/ML unchanged; no scanner tables copied wholesale (Market Brief
summarizes and routes; deep-dive stays in the scanner pages); no predictive claim
from scanner agreement.
