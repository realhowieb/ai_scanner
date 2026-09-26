# Run 76 - HSF Score Presentation Ordering

## Before

The scanner engine ranked and stored results using its existing internal logic,
where Breakout Score can influence row order. The UI then added canonical HSF
Score without changing that order. Consequently, a lower-HSF row could appear
above a higher-HSF row even though HSF Score is the product's headline
opportunity-ranking number.

Ordering audit:

| Surface | Before Run 76 |
|---|---|
| Scheduled `cron` / `US_MARKET` storage | Existing engine order, unchanged |
| Scanner table and cards | Incoming engine/stored order |
| Lenses and saved screens | Filtered incoming order without re-ranking |
| CSV export | Current displayed dataframe order |
| Interactive grid | Incoming order initially; user sorting available |
| Custom/watchlist/single-ticker scan | Engine output stored in session, then displayed in engine order |
| Today top setups | Already canonical HSF Score order |
| Market Brief opportunities | Already canonical HSF consolidation order |
| Session recap standouts | Already canonical HSF Score order |
| New-since-last-visit | Membership comparison; displayed as an alphabetical name list |
| Stock Intelligence handoff | Ticker lookup, not positional; carries matching row and opportunity |

The scanner and custom-scan engines can apply `top_n` before the presentation
layer receives rows. Run 76 ranks every available result, but intentionally does
not expand or alter that certified engine boundary.

## Decision

Any collection presented as ranked HSF opportunities defaults to descending
HSF Score. Internal scanner order remains an engine/research detail and stored
runs are unchanged.

Ties use:

1. HSF Score descending;
2. Breakout Score descending when available, preserving a meaningful internal
   scanner signal as the secondary rank;
3. ticker ascending;
4. stable source position for otherwise identical rows.

Missing or malformed HSF Scores sort last.

## Architecture

```text
raw scanner or stored results (unchanged)
  -> canonical opportunity consolidation
  -> add canonical HSF Score to a copy
  -> rank presentation copy by HSF Score
  -> lenses / saved screens
  -> table or cards / CSV / Stock Intelligence handoff
```

The shared `rank_hsf_opportunities()` helper is intentionally UI-only. It
preserves columns, index values, dataframe attributes, and the caller's frame.

## Changes

- `ui/headline_score.py`: added the canonical presentation-ranking helper.
- `app.py`: ranks after adding HSF Score and before lenses or display limits.
- `ui/results_tabs.py`: applies the same ranking when viewing saved scan history.
- `ui/results.py`: export continues to use the displayed HSF-ranked dataframe;
  corrected auto-selection terminology to "Top HSF opportunity."
- `ui/result_cards.py`: documented its shared ranked input.
- Tests: added ordering, ties, malformed values, lenses, cards/table parity,
  handoff, CSV/history wiring, and raw-frame immutability coverage.

## Consistency Verification

- **Today:** already uses `consolidate_scanner_results`, whose primary key is
  descending canonical HSF Score.
- **Scanner:** ranks the full dataframe available to UI immediately after the
  canonical HSF Score column is added.
- **Cards and table:** consume the identical post-lens dataframe.
- **Lenses and saved screens:** only filter the ranked frame and preserve order.
- **Interactive grid:** starts HSF-ranked; Streamlit's user-selected column sort
  remains available and is not persisted over by a second render-time sort.
- **Stock Intelligence:** ticker-based handoff remains correct after reordering;
  no positional lookup was introduced.
- **Custom scans:** raw session output remains untouched; the UI works on a
  scored and ranked copy.
- **CSV:** represents the currently displayed opportunity results and exports
  in their current HSF-ranked order.
- **Market Brief and recap:** already use canonical HSF ranking.

## Frozen-Core Verification

Run 76 does not modify `scan/engine.py`, scanner scoring formulas, the HSF Score
formula, model inference/training, scheduled execution, Gate U, research capture,
cohort/maturation logic, or Run 56/58/61 research behavior. No stored scan row is
rewritten. This is a presentation-only ranking change.

## Backlog Reconciliation

- **P1-11:** DONE by Run 76.
- **P1-18:** DONE by Run 75 - the guided tour is the single onboarding UI.
- **P1-19:** DONE by Run 75 - the Scanner has one Custom scan entry.
- **P1-20:** DONE by Run 75 - authenticated sessions land on Today once.
- **P1-12:** OPEN - next expected item.
- **P1-21:** WAIT; no dependency change was found.
- **P1-22:** OPEN.

## Tests

Run 76's final lint, focused tests, complete regression, and Streamlit smoke
results are recorded in the completion report.

## Next Item

Proceed to **P1-12 - Breakout alert threshold on a visible scale**, unless a
higher-severity issue appears.
