# Intelligent Alerts & Opportunity Feed (Run 40)

Presentation layer that turns HSF's existing scanner/model outputs into a clear,
deterministic feed answering: **what deserves attention now, why it matters, what
changed, and how urgently to surface it.**

> **Alert Priority is NOT a prediction of future return.** It is an attention/
> urgency signal. Nothing here claims expected return, win probability, or edge.
> Scanner agreement is shown as **confirmation/context**, never proven edge. No
> scanner, model, or DT behavior is changed or re-run; no Opportunity Score is
> created.

Engine: `analytics/opportunity_view.py` (pure, deterministic, tested). Thin
Streamlit surface: `ui/opportunity_feed.py`.

**Mounted (Run 41):** the feed is live in **Market Brief** as the "🔔 Intelligent
Alerts" section (`ui/market_brief.py::render_intelligent_alerts`), reusing this
engine via `analytics/market_brief_view.py` — no duplicated logic. See
[MARKET_BRIEF.md](MARKET_BRIEF.md). Both read a canonical Run 36
observation (as produced by Run 38A production capture), so the feed integrates
naturally with the accumulating research dataset.

## Opportunity View Model (`hsf-opportunity-view-1.0`)

`build_opportunity_view(obs, *, prior_obs, prior_alert, watchlist, present)` →
```
symbol, timestamp, session, price, change_pct,
primary_setup, primary_setup_name, secondary_setups,
scanner_count, scanner_names, direction,
scores: {prebreakout_probability, ai_confidence},
positive_reasons[], risk_reasons[], changes_since_prior[],
data_quality, freshness, is_watchlist,
alert_priority, priority_reason, lifecycle_state,
should_alert, dedup_reason
```
Presentation only — it aggregates existing evidence and computes no new score.

## Primary setup

Deterministic priority by **product specificity** (not historical performance —
Run 40 must not use unvalidated performance weighting):
`prebreakout → golden_cross → breakout → momentum → unusual_vol → gap_up →
gap_down → most_active`. The first triggered scanner in that order is the primary
setup; the rest become `secondary_setups`.

## Multi-scanner agreement

`scanner_count` + `scanner_names`, rendered as "N scanners agree — also: …" with
the explicit caption "confirmation/context, not a predictive edge."

## Alert Priority rules (deterministic, transparent)

- **HIGH** — ≥3 scanners agree, OR PreBreakout probability ≥ 70%, OR a
  significant change (direction flip, moved above VWAP, new scanner).
- **MEDIUM** — exactly 2 scanners, OR PreBreakout ≥ 40%, OR RVOL ≥ 2×, OR any
  material change.
- **LOW** — otherwise (single setup, limited confirmation).
- **Data-quality cap:** stale or fallback data caps priority at MEDIUM ("strong
  signals but data quality is limited") so untrustworthy setups are never HIGH.

`priority_reason` states exactly which condition(s) fired. Priority never
references return/probability-of-profit/edge.

## Explanation rules (positive reasons)

Evidence-based, from present fields only: PreBreakout probability, AI confidence,
RVOL ≥ 1.5×, Above VWAP, ADX ≥ 20 (trending), positive change %, gap ≥ 1%, and
"N scanners agree" (≥2). Missing values produce no line — nothing invented.

## Risk / caution rules

Only data-supported: Below VWAP / Extended above VWAP (≥4%), low participation
(RVOL <1×), weak trend (ADX <15), large opening gap (≥5%), high volatility
(ATR% ≥6), conflicting scanner directions, late-session, and Run 37/36 data-
quality flags (Stale / Incomplete-fallback / Partial). No fabricated
earnings/news/catalyst risk (no reliable repo data → not shown).

## Change detection (vs prior observation)

Meaningful transitions only: PreBreakout probability ±10 pts, RVOL ±0.5×, VWAP
crossing ("Moved above VWAP"/"Lost VWAP"), new scanner, scanner-agreement count
change, direction change. Small wiggles below thresholds are ignored.

## Dedup / cooldown

`should_alert(view, prior_alert)` fires a NEW alert only on: first trigger,
priority increase, setup reappearance (prior RESOLVED), or a material change (new
scanner, direction, VWAP transition, probability/RVOL move). Otherwise the
existing alert is updated, not re-fired — six consecutive identical PreBreakout
scans yield one alert, not six.

## Lifecycle

`NEW` (no prior) → `STRENGTHENING` (new scanner / moved above VWAP / rising
prob/RVOL) / `WEAKENING` (lost VWAP / falling metric / direction→neutral) /
`ACTIVE` (little change) → `RESOLVED` (setup gone, `present=False`). States are
only assigned when history supports them.

## Feed UX

`rank_feed` orders by priority → scanner agreement → PreBreakout probability →
symbol (deterministic, not predictive). `filter_feed` supports All / High
Priority / PreBreakout / Momentum / Unusual Volume / Gappers / Bullish / Bearish
/ Watchlist / New / Strengthening. Cards are compact (symbol, primary setup,
priority, price move, top 4 reasons, top 2 risks, top 3 changes) — scannable, not
a report. Watchlist symbols show a ★.

## Data quality / freshness

`freshness` → Fresh / Partial Data / Stale from Run 37/36 metadata; only surfaced
when not Fresh, so healthy cards stay clean.

## Empty states

`empty_state_message` uses real counts only: "No high-priority opportunities
right now. HSF scanned N symbols. M setups were detected, but none met the
current High Priority criteria." Unknown counts are omitted (never "None").

## Analytics instrumentation

Each view carries `observation_id`, `alert_priority`, `scanner_names`,
`lifecycle_state`, `timestamp` — market-only, no user PII — so surfacing/lifecycle
can later be logged into the canonical dataset without new personal data.

## Existing score labeling

The view keeps scores namespaced and named: `prebreakout_probability`,
`ai_confidence`, and `alert_priority` (attention, not probability). The renderer
labels agreement as confirmation/context and priority as an attention signal.

**Watchlist (Run 42):** the same Run 40 engine powers the full Watchlist Intelligence page via `analytics/watchlist_view.py` — see [WATCHLIST_INTELLIGENCE.md](WATCHLIST_INTELLIGENCE.md). Market Brief shows the summary; the Watchlist page shows the detailed personal view, with consistent classifications.
