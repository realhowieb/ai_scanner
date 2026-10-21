# HSF research workflow improvements — October 18, 2026

Implemented against the current root checkout. The older nested `ai_scanner/`
checkout was not changed. No scoring, ranking, outcome computation, entitlement,
authentication, or billing code was changed.

## Changes

- Stock history now separates grouped ticker/day signals, saved ticker
  observations, score-range price outcomes, and HSF-state persistence. Each
  measure describes its sample unit and horizon. Unknown horizons remain
  explicitly undefined rather than being converted to trading days.
- The fallback signal-outcome display uses the returned horizon for both SPY
  comparisons and median returns, instead of labeling every fallback as 5 days.
- Scanner saved views, filters and table display share a compact toolbar. Existing
  URL filters, pagination, comparison controls, saved settings and mobile cards
  retain their behavior.
- Weekend freshness explanations use the API's checked timestamp and confirmed
  on-schedule state. They do not introduce elapsed-age rules or guess holidays.
- Day Trader headings distinguish open, pre-market, after-hours and closed states.
  Request-check times are labeled as checks. Weak best-ranked candidates retain
  their score and ranking, with quality clearly labeled. “No strong setup” is
  shown only when no unflagged strong row for that direction exists.
- Alert creation starts with stock, watchlist or market scope. Basic price, HSF
  Score and setup choices are grouped separately from advanced technical options.
  The existing server catalogs determine conditions, ranges and availability.
  Each condition still submits to its original API. HSF rules require exactly
  one stock or watchlist for ticker conditions. After the October 21 main update,
  market-wide HSF Score, setup and earnings conditions are created from the Scanner;
  market-only types are excluded from the stock/watchlist composer.
- HSF rules appear in the Alerts Rules tab with evaluation/trigger timestamps
  when supplied, enable/disable controls and confirmed deletion. No rules were
  changed in a real account during verification.
- Activity empty states distinguish active rules, paused rules, unavailable rule
  status and no rules. Email availability is not presented as verified delivery.
  Price/technical saved limits and the HSF active allowance remain distinct.
- Watchlists keep Add tickers accessible and collapse administration on all
  screen sizes. Empty change summaries use one line rather than a large card.
- Root and web READMEs describe the current web routes and setup accurately.

## History investigation

The reviewed counts are not equivalent denominators:

1. `GET /v1/outcomes/symbols/{ticker}` defaults to `signal_day`: one record per
   ticker per entry day, with 1-, 3- and 5-trading-day outcome windows. The stock
   page prefers the declared 5-day window.
2. `ui.stock_intelligence` builds `history_summary` from individual frozen
   observations returned by `fetch_ticker_opportunity_history`. Its default
   retrieval window is 45 days and 300 rows. Multiple observations can belong
   to a single day. These totals therefore need not match the signal-day totals.
3. Score-range price context uses matured observations across tickers. Its
   positive outcome is a rise of at least 4% within 5 trading days, using a
   5-day return fallback when maximum favorable excursion is absent.
4. Setup persistence uses the same initial HSF status and score band, with
   comparable HSF-state outcomes. H24 selects the first valid comparable
   observation at least 24 elapsed hours later; weekends and holidays count.
   It measures state persistence/strengthening, not price gains.

The frontend now explains these differences. No production rows were retrieved
or recomputed, and the original 2-vs-3 example is not evidence of a calculation
error by itself.

## Separate follow-up: positive-outcome definition mismatch

Source inspection found a definition difference worth investigating separately:
`analytics.hsf_calibration._positive` uses `return_5d` when `mfe_5d` is missing,
whereas `ui.stock_intelligence`'s ticker `history_summary.positive` counts only
records with `mfe_5d >= 0.04`. The ticker-history fetch does not return
`return_5d`. Equal samples could therefore produce different positive counts
when maximum-excursion data is missing. This may be intentional, but needs a
decision about the canonical definition and a data audit before changing it.
No calculation or data was modified in this pass.

## Remaining limitations

- Day Trader's current response exposes request `as_of`, but no session date or
  per-symbol last-trade timestamp. The UI discloses this. Supplying verified
  timestamps requires a separate API/provider change; dates were not inferred.
- Alerts Activity describes the recent event feed, not an all-time event count.
- Historical cohort date ranges are not supplied in the stock-detail dictionaries.
  The UI does not invent dates or claim the different cohorts cover equal periods.
- Screenshots use synthetic local fixtures, not live quote validation. The fixture
  service rejects data mutations. Physical phones, real delivery, provider data,
  checkout, and production deployment were not tested.

## Verification

- Frontend regression: 38 test files, 314 tests passed.
- Type checking and lint passed.
- Production build passed.
- Browser checks at 1440 × 1000 and 390 × 844: Scanner, stock history, alert
  composer, Watchlists and Day Trader. No horizontal document overflow was
  observed at the checked phone width.
- Keyboard checks: native Scanner disclosures, watchlist management disclosure,
  alert-dialog forward/reverse focus wrapping, Escape, focus return to the opener,
  and focus transfer from View rules to the Rules tab.
- Existing test warnings about React `act` and simulated navigation remain.
  These checks are not a full accessibility certification.

Locally captured screenshots and a gallery are under
`artifacts/review-2026-10-18/implemented/`. This folder is ignored by Git; the
images are workspace review artifacts, not part of the deployed application.

## Main integration — October 21, 2026

Updated the working branch to main `fc82314` and reconciled the local UI work
with earnings scans, similar setups, smart alerts, starter actions, and watchlist
price/performance features. Main already retires Streamlit and removes its UI
packages from the API dependency path. Updated local setup instructions to start
FastAPI and Next.js. The original local UI checkpoint remains in Git stash.

Verification after integration: 44 frontend test files, 350 tests passed; type
checking, lint and production build passed. API/cache/dependency checks: 107
passed, 9 skipped, plus 35 subtests passed. A fresh-process regression blocks
Streamlit, Plotly, Altair, Pillow and PyArrow imports and confirms API creation,
OpenAPI generation, and shared scanner/research imports succeed. These checks
do not measure production latency or memory savings. No deployment was performed.
