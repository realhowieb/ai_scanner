# Run 79: HSF Information Hierarchy and Trust

## Executive Summary

Run 79 makes the existing product hierarchy explicit without changing any
score, model, ranking, scan, or research behavior:

1. **HSF Score is the headline opportunity ranking.**
2. Supporting scanner/model outputs are available as **Model details**.
3. Today's **Top setups** contains only existing `STRONG` opportunities.
4. A completed scan with no qualifying result now says so plainly instead of
   promoting the least-weak names.
5. The internal column historically called `AI Confidence` is presented to
   users as **5D outcome probability**, which describes its actual calibrated
   target.

## Score Inventory

| Metric | Main surfaces | Prior label | Intended role / final presentation |
|---|---|---|---|
| Canonical opportunity score | Today, Scanner, Market Brief, My Stocks, Stock Intelligence, alerts, historical evidence | HSF Score | **Primary.** 0-100 opportunity ranking; never a probability or expected return. |
| Scanner technical score | Scanner details/table, Stock Intelligence, Breakout alerts, exports | BreakoutScore / Breakout score | Secondary **Breakout score**. It remains explicit on Breakout alerts because that alert contract uses this scale, not HSF Score. |
| Early-setup model | Scanner details, Early Breakout, Market Brief, Stock Intelligence, model/research views | PreBreakout / confidence / likelihood | Secondary **PreBreakout setup probability** for its defined future-quality-setup event. |
| Five-day outcome classifier | Admin model diagnostics, supporting tables, watchlist model view, exports | AI Confidence | Secondary **5D outcome probability**: calibrated probability of reaching +4% before -2% within five trading days. The internal schema name remains unchanged. |
| HSF components | Stock Intelligence | score breakdown / model metrics | Explanation under **Why HSF scored it this way**, subordinate to the headline. |
| Historical rates | Stock Intelligence, Historical Research, Intelligence Performance | historical context / positive outcome rate | Descriptive evidence about saved observations, separate from live ranking and never a guarantee. |
| Alert Priority | Market Brief, alert research | Alert Priority | Attention-order signal only, not an HSF score or return probability. |
| Day Trader score/tier | Day Trader | DT Score / Strong / Developing / Weak | Separate intraday setup-coherence system; it is not presented as the HSF opportunity score. |

Internal training, automation, persistence, and CSV field names are retained so
this copy pass does not break schemas or model consumers.

## Final Hierarchy

**Primary:** HSF Score. It answers: “How strongly does HSF rank this current
opportunity?” It receives first position on Today, Scanner cards/results, and
Stock Intelligence.

**Secondary:** Breakout score, PreBreakout setup probability, 5D outcome
probability, and component evidence. These explain supporting quantitative
views. They do not replace HSF Score and are not all claimed to be direct HSF
inputs.

Scanner result cards remain intentionally compact: ticker, HSF Score, setup,
market context, and the Stock Intelligence action. The raw table keeps model
fields available behind the existing Model details toggle.

## Today's Top Setups

- **Threshold:** HSF Score `75`.
- **Source:** `HSF_STRONG_MIN` in `ui.opportunities`, the existing lower bound
  for canonical `STRONG` status. The status function and Today import one
  constant; Run 79 did not add a parallel score policy.
- **Qualifying behavior:** scores at or above 75 appear, highest first, up to
  the existing five-item limit.
- **Boundary:** exactly 75 qualifies.
- **Weak-market behavior:** a completed scan with no score at 75+ displays “No
  high-quality setups meet the current HSF threshold” and directs users to the
  full Scanner.
- **Empty-scan behavior:** the existing unavailable-scan state remains distinct
  and appears before the Today sections render.
- **Scanner preservation:** the default `top_setups` helper remains unfiltered;
  only Today passes the optional display minimum. Stored results, full Scanner
  results, recap, research capture, and scheduled scans are unchanged.

## Model Details

- Stock Intelligence now separates **Why HSF scored it this way** from a
  dedicated **Model details** expander.
- The Model details section explicitly says its outputs are separate from the
  headline HSF Score.
- Scanner's existing hidden-by-default model details use accurate display
  labels.
- Screenshot/demo Scanner rows no longer place PreBreakout and Breakout model
  numbers beside the HSF headline.
- Secondary metrics remain available in the full model-detail/raw-data paths.

## Terminology Changes

| Before | After |
|---|---|
| AI Confidence | 5D outcome probability |
| AI Confidence calibration | 5D outcome probability calibration |
| AI Confidence target/model | 5D outcome target/model |
| PreBreakout model confidence | PreBreakout setup probability |
| No pre-breakout predictions | No PreBreakout model outputs |
| HSF Score breakdown & model metrics | Why HSF scored it this way + Model details |

The implementation continues to read/write `AI Confidence` and
`PreBreakoutProb%` internally. This avoids a data migration and keeps exports,
alerts, trained artifacts, and historical rows backward compatible.

## Methodology Verification

The methodology now states:

- HSF Score is an opportunity ranking, not a probability, expected return,
  prediction, or guarantee.
- Breakout score is technical setup strength.
- PreBreakout estimates its defined setup/outcome event.
- 5D outcome probability is calibrated for +4% before -2% within five trading
  days, not “AI confidence” in a general sense.
- Live ranking asks what looks interesting now; Historical Research separately
  describes how similar saved observations behaved afterward.

Negative trust language such as “not a probability of profit” remains where it
prevents misinterpretation. Unsupported certainty claims are regression-tested
on the customer-facing files touched by this run.

## Files Changed

- `ui/opportunities.py`
- `ui/market_scans.py`
- `ui/today.py`
- `ui/headline_score.py`
- `ui/result_helpers.py`
- `ui/results.py`
- `ui/results_intelligence.py`
- `ui/stock_intelligence.py`
- `ui/watchlist_intelligence.py`
- `ui/ai_confidence_explain.py`
- `ui/prebreakout_tab.py`
- `ui/paper_trade.py`
- `ui/methodology.py`
- `analytics/intelligence_view.py`
- `analytics/signal_leaderboard.py`
- `tests/test_p07_headline_score.py`
- `tests/test_run79_information_hierarchy.py`
- `docs/RUN_79_INFORMATION_HIERARCHY_TRUST.md`

## Tests

Run 79 adds boundary, mixed-result, weak-market, empty-scan, Scanner
preservation, model-detail hierarchy, methodology, and high-risk-copy tests.

- Targeted Runs 62/70-79 hierarchy regression: 151 passed, 0 failed/skipped.
- Full repository suite: 1,830 passed, 0 failed, 26 optional-dependency skips,
  172 subtests, no warning summary.
- Isolated Run 78 billing/pricing/tier gate: 85 passed, 0 failed, 1
  Streamlit-only skip, 6 subtests. Its 57 warnings are existing
  FastAPI/Starlette deprecations.
- Ruff (`E9,F,I` with repository-standard ignores): passed.
- `git diff --check`: passed.
- Streamlit startup smoke (`--timeout 60`): passed.

## Frozen-Core Verification

Run 79 does **not** modify:

- HSF Score formula, weights, components, ranking, version, or persisted data
- Breakout Score or PreBreakout formulas, artifacts, weights, or training
- `scan/engine.py`, `scan/breakout.py`, scanner universes, or schedules
- research capture, maturation, Gate U, Autonomous Research Mode, or certified
  Run 56/58/61 behavior
- alert thresholds/semantics, including the Run 77 Breakout score contract
- pricing, tiers, billing, or the Run 78 billing CI gate

## Backlog Status

- **P2-11: DONE** — Today reuses canonical STRONG >=75 and handles a valid weak
  market separately from an unavailable scan.
- **P2-12: DONE** — HSF is the headline; secondary outputs remain accessible
  under Model details and accurate supporting labels.
- **P2-15: DONE** — model/probability terminology and methodology now match the
  implemented targets across the relevant customer surfaces.
- **P1-22 remains protected** by the dedicated Run 78 billing contract job.

Unless production evidence reveals a higher-severity issue, the next backlog
work is Run 80: P2-13 + P2-14 mobile UX and persistent user-state polish.
