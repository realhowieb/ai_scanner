# Day Trade Stair-Stepper Outcome Validation

## Purpose

This instrumentation measures the existing Day Trader Stair-stepper without
changing its detection, filters, ranking, or user-facing recommendation. The
supported fitted windows are 10, 15, 20, 30, 45, and 60 one-minute bars. The
production default remains 45 bars.

No window is presented as predictive or "best" until the report's evidence
gate is met.

## Existing Pipeline

1. `ui.stair_stepper.fetch_recent_minute_bars` requests Alpaca one-minute bars
   for up to 40 symbols already present on the Day Trader page. It first uses a
   four-hour lookback and falls back to four days when the market is closed.
2. `analytics.stair_step.latest_session_bars` orders valid bars and keeps only
   the latest US/Eastern trading date, so a fitted window never bridges an
   overnight gap.
3. `stair_step_metrics` fits close against elapsed minutes. It returns direction,
   R-squared, price slope per minute, trend percent per hour, fitted/reference
   price, coverage, and the deepest close-based pullback.
4. `is_stair_stepper` applies the existing direction, R-squared, trend, and
   pullback rules. Research evaluates both directions but uses the exact visible
   numeric thresholds. The user's selected window still controls the displayed
   results.
5. Qualifying records are stored through the canonical immutable
   `hsf_observations` infrastructure. Outcome records remain separate.

## Observation Contract

Context: `day_trader:stair_stepper`

Each immutable observation contains:

- canonical observation ID, symbol, exact detection timestamp, and 30-minute
  detection bucket;
- trading date and PREMARKET, REGULAR, or AFTERHOURS session;
- window, UP/DOWN direction, R-squared, slope per minute, current price,
  fitted/reference price, pullback, bar coverage, and fitted bar count;
- the exact R-squared, pullback, and trend qualification thresholds;
- source, feed, source price timestamp, and detector version;
- requested outcome horizons: 5, 10, 15, and 30 minutes.

Unavailable context is not manufactured or zero-filled.

## Deduplication

The ID is deterministic for:

`symbol + fitted window + direction + 30-minute UTC bucket + detector version`

Repeated refreshes in the same interval therefore resolve to the same canonical
record and the database's first-write-wins constraint rejects duplicates. A
30-minute interval limits refresh-frequency bias while retaining distinct
intraday setup episodes. Different windows and directions remain separate so
window overlap can be measured.

## Outcome Maturation

The existing `Mature Observations` workflow performs the work. It batches each
symbol's Alpaca bars once and reuses them for all observations. Stair-stepper
records take a context-specific path for 5/10/15/30-minute outcomes; all other
observation contexts retain their existing 5/15/30/60-minute behavior.

For each available horizon the worker records:

- raw return from the frozen entry price;
- direction-adjusted return, where positive means the detected direction won;
- MFE and MAE over only the bars between detection and that horizon;
- future high/low, target time, actual source-bar time, and target lag.

Bars must be strictly later than the detection. A target bar may be at most two
minutes late. Missing bars remain missing. Outcomes are constrained to the same
US/Eastern date and market session as the detection, so a regular-session setup
cannot mature on after-hours bars. Outcomes are first-write-wins by observation
and horizon.

## Reproducible Report

Run:

```bash
python -m scripts.analyze_stair_stepper \
  --out artifacts/automation/stair_stepper
```

This creates:

- `stair_stepper_validation.json`: machine-readable window, overlap, episode,
  R-squared, pullback, consensus, and verdict data;
- `stair_stepper_validation.md`: compact human-readable summary.

The scheduled maturation workflow creates and uploads both alongside the
companion maturation report.

The report includes all six windows even when one has no observations. It
compares every required horizon, distinguishes first detection from first
confirmation, and reports common simultaneous window combinations. Missing
outcomes are not treated as zero returns. Failed retrieval counts remain in the
companion maturation report rather than being guessed from missing outcomes.

## Evidence Gate

A role requires at least 30 matured 15-minute outcomes across at least five
trading days for a window. With sufficient evidence, the deterministic ranking
considers median direction-adjusted return across horizons, weakest-horizon
return, win rate, MFE-minus-MAE, and daily stability. It may report:

- `FASTEST_SIGNAL`
- `BEST_CONFIRMATION`
- `BEST_RISK_REWARD`
- `MOST_CONSISTENT`

Before the gate is met, the only verdict is `INSUFFICIENT_EVIDENCE`. These are
research classifications only and do not alter the production selector or UI.

## Current Limitation

Stair-stepper observations begin accumulating only when the on-demand Day
Trader check is run. Sparse IEX minute coverage can leave outcomes missing.
The generated report is the authoritative source for the current sample; no
predictive claim should be made from an immature or unrepresentative sample.

At implementation time no production database credentials were available in
the local verification environment. The only locally discovered records were
synthetic `AAA`/`CCC` UI-test records with no matured outcomes; they are not a
research sample. That render test is now isolated from persistence. The current
trustworthy local sample is therefore zero and the verdict remains
`INSUFFICIENT_EVIDENCE`; the scheduled Neon report after deployment is the
authoritative production count.
