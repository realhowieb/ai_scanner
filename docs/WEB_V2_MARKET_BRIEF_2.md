# Web v2 Market Brief 2.0

## Architecture and inspection

The /brief route mounts web/src/features/BriefView.tsx. Previously an index strip
sat above a wrapping main/sidebar split; the main column contained another
two-column grid. Unequal supporting cards drove the layout rather than current
intelligence. The replacement uses independent supporting stacks and a wider
center stack. At tablet widths the center spans both columns; phones use center,
left, right DOM order. No masonry or cross-column row-height coupling.

Data remains GET /v1/brief, with entitlement-gated GET /v1/earnings?days=7 and
on-demand GET /v1/ai/brief-narrative. Earnings loading/failure is independent;
AI generation is never automatic. Brief loading reserves two named skeleton
panels. Existing unavailable/retry states remain, and stale refresh failures are
shown without discarding the previous response.

## Data availability and implementation decisions

| Metric | Availability | Treatment |
| --- | --- | --- |
| SPY / QQQ | AVAILABLE: market list, from evening_wrap._market_close_context | Display supplied labels, values, signed daily changes |
| IWM / VIX | MISSING in current producer | Not fabricated; renderer accepts them if supplied later |
| Market session | AVAILABLE: Brief.phase, backend _market_phase | Existing header label; no client holiday/calendar calculation |
| Regime | MISSING from Brief contract | No risk-on/off badge |
| Breadth | AVAILABLE: snapshot PctChange counts | HSF snapshot breadth, not whole market; denominator excludes unchanged |
| Unchanged breadth | MISSING | Omitted; never assumed zero |
| HSF 80+ | DERIVABLE within returned five opportunities only | Explicitly labeled HSF 80+ in Radar; not a universe count |
| PreBreakout | AVAILABLE, Premium gated, returned picks | Count labeled picks shown; original details retained |
| Stair-Stepper totals | MISSING in Brief; separate user-requested computation | Omitted; no new scanner execution |
| Golden crosses | AVAILABLE: returned list | Count labeled crosses shown; ticker links retained |
| Prior counts | MISSING | No invented deltas |
| Sectors | AVAILABLE: sector ETF changes | Deterministic min/max among supplied sectors, no new score |
| HSF scores / setups | AVAILABLE: opportunities.score / primary_setup | Existing canonical Brief order unchanged |
| Score movement | AVAILABLE: score_delta from compare_opportunities | Signed movement; absent baseline stays absent |
| Global rank / previous rank | MISSING from current Brief | Brief # is position in canonical list, NOT global Scanner rank; rank movement only if both fields supplied |
| Snapshot time | AVAILABLE | Existing Freshness and ET formatting; render time never replaces source timestamp |
| Per-index/sector quote timestamps | MISSING | Explicit disclosure; snapshot time is not claimed as quote time |
| Historical signal quality | Separate Pro /v1/outcomes/* exists | Omitted: no verified live matched matured evidence in this environment; no selected historical winners |

api/market.py builds five opportunities via ui.opportunities.build_opportunities
and compares persisted prior snapshots. Radar reuses this exact response rather
than fetching an independently timed scanner population. /v1/scans/latest has
canonical setup rows but does not supply prior ranks either. Watchlist intelligence
has prior ranks in a different population; those are not mixed into Brief ranks.

## Presentation

Market Pulse leads with market values, clearly scoped breadth and counts, then
sector context. Radar shows ticker, HSF Score, real score movement, Brief position,
setup and canonical status. Ticker links reuse /stocks/{ticker}; View Scanner uses
/scanner without introducing filter conventions. Supporting model outputs are
retained under native keyboard-accessible Model details. Plan redaction remains
server-owned and unchanged.

Desktop column weights are 1 : 2 : 1.15. At 768-1199px, intelligence spans the
page above two supporting stacks. Below 768px, compact Radar rows wrap setup text
and status beneath ticker/score. Signed numbers and text supplement color. Native
links, details, headings and existing focus treatment remain keyboard accessible.

No API calls added or removed, no new cache, no DB calls per card, and no frontend
business scoring. Counts and sector extrema are bounded presentation operations
on already-loaded arrays. Backend, API contracts, ranking, ML, PreBreakout,
Day Trader, outcomes, authentication and billing are unchanged.

## Verification

- Frontend full suite: 172 tests passed, 18 files (no live providers).
- Typecheck, ESLint and production build passed.
- Existing jsdom navigation-not-implemented notice remains non-failing.
- Next warns about an unrelated parent-directory package-lock.json being ignored.
- Headless Chrome checked 1440, 1280, 1024, 768 and 390px: no horizontal page overflow.
- Screenshots: /tmp/brief-1440.png, /tmp/brief-1280.png, /tmp/brief-1024.png,
  /tmp/brief-768.png, /tmp/brief-390.png.
- Screenshots use test-only representative data rendered by the real BriefView
  with production CSS, not live market values. This verifies component layout,
  not authenticated production shell or live provider operation.
- Long setup text wraps, center is twice the left-column width on desktop, and
  compact rows replace desktop-table scrolling on phones. Supporting sections
  cannot push Radar down based on their heights.

Tests cover positive/negative indexes and scores, missing movement/breadth, rank
direction and absent prior rank, freshness, four backend sessions, empty Radar,
loading panels, earnings failure isolation, ticker/Scanner links, plan gates and
center-first responsive DOM order. The optional BRIEF_QA_HTML environment variable
on the discover test exports a test-only HTML fixture for reproducible browser QA.

## Files

- web/src/features/BriefIntelligence.tsx: Pulse, Radar and rank movement renderer.
- web/src/features/BriefView.tsx: new composition, Model details, loading/error states.
- web/src/app/globals.css: scoped dashboard grid and compact component treatment.
- web/tests/brief-intelligence.test.tsx: deterministic component tests.
- web/tests/discover.test.tsx: integration regressions and responsive QA fixture.
- docs/WEB_V2_MARKET_BRIEF_2.md: this audit and implementation record.

## Remaining opportunities

Full-universe counts, canonical prior ranks, independently timestamped indexes,
and matched historical evidence require explicit API support/verification before
display. Current API sometimes converts missing sources to empty lists; the UI
cannot distinguish those failures from genuinely empty data. This change does
not invent an error taxonomy or alter that contract. Production authenticated
QA remains necessary. No automatic main promotion.
