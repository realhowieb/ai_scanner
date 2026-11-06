# Saved-signal ranking study — methodology v2

Historical signal research, not executable fills, a capital-constrained portfolio
backtest or proof of profitability. Outcome Intelligence's overall record remains
a separate close-to-close observation study.

## Selection and availability

Use scheduler-owned snapshots (`username=cron`, `label=daily_snapshot`) only;
private or legacy user-created saved scans cannot feed public per-pick provenance.
Use the latest retained eligible snapshot per New York exchange date in the bounded
lookback, ordered by saved timestamp then ID. Existing score-epoch boundaries
remain. Weekend, exchange-holiday and unsupported-calendar snapshots are excluded
before ranking. This preserves the latest-snapshot rule across universes, rather
than claiming to pool every universe's signals. Source-selection exclusions are
not selected-pick return exclusions.

`db.runs.save_daily_snapshot` refreshes its payload and `created_at` together.
Use the full saved timestamp as a conservative availability bound, never a bar
date or scan start. Missing/naive timestamps cannot support time-aware entry.
Overwritten historical payloads cannot be reconstructed from the current row.

Select up to five distinct tickers by saved BreakoutScore or PreBreakout
probability, preserving saved order for ties. Captured prediction provenance
wins over display values; explicit unavailable status is excluded. Saved legacy
probabilities remain supported, but no usable predictions/all-zero fallback
snapshots are not ranked. Never load today's model to reconstruct past picks.
`ranking_unavailable_snapshots` reports those snapshots; no invented replacement
picks enter the denominator. Snapshot payload/read failures also do not create
picks: coverage measures selected picks from successfully loaded retained snapshots,
not completeness of the original scanner population.

## Entry, exit and provenance

Primary version `track-v2-next-open`: first regular-session open **strictly after**
snapshot availability. Premarket can enter that session; at-open/intraday and
post-close timestamps enter the next eligible session. Exit is the close of the
h-th session including entry: 1d is entry-session open to close; 5d ends at the
fifth session's close. A not-yet-ended window is excluded as incomplete.

Use the repository's explicit NYSE calendar, America/New_York conversion, DST,
holidays and early closes. Outside the bundled calendar's supported years, fail
closed and extend the calendar before evaluating. Daily bar timestamps are session
date labels, including Alpaca's midnight ET labels; do not shift UTC-midnight
fixture labels back a date. Stock and SPY must have the exact same required
session dates and fields. Never shift only one leg or substitute Close for Open.
These reference prices still do not guarantee an available fill.

The sibling leaderboard's close comparison is separately versioned
`track-v2-next-close`: first close strictly after availability, exit h sessions
later. It is not mixed with the primary study. Date-only `_forward_return` stays
a legacy close-to-close research helper for economic-label callers, not a signal
execution policy; it also refuses unavailable/invalid Open rather than substituting.

Per-pick provenance records snapshot ID when available, symbol, signal/entry/exit
timestamps, entry policy, methodology, actual stock/SPY reference prices and
returns, or an exclusion code. It is retained in persisted JSONB summary payloads.

## Returns, denominators and dependence

Absolute = stock exit / stock entry - 1. Excess = absolute minus matched SPY
return. Positive-return rate uses strict absolute > 0; benchmark-outperformance
rate uses strict excess > 0. Zero returns and benchmark ties count false in their
respective rates; averages/medians include zeros. API returns are fractions.
Deprecated `avg_return`, `median_return`, `win_rate` keep their excess/outperformance
meanings. Preferred fields distinguish both types. Outperforming SPY can still
lose money (stock -2%, SPY -5% gives +3 percentage points excess).

Every selected pick is evaluated or excluded: selected = evaluated + excluded,
and reason counts sum to excluded. Coverage = 100 * evaluated / selected, null
when no picks were selected. Missing returns never become zero or another stock.
If both legs fail, benchmark failure takes precedence, giving one primary reason
per excluded pick. Codes cover missing symbol/benchmark bars, unavailable entry/
exit fields, invalid entry/exit prices, invalid bar indexes, incomplete windows,
unsupported calendars and unknown availability; benchmark-specific failures use
`BENCHMARK_`. Provider failures retain selected picks as missing-data exclusions.

`sample_size`/`observation_count` count evaluated pick occurrences, not independent
trades. Report unique evaluated/selected tickers, signal dates, considered
snapshots and evaluated dates (`runs_used`) separately. Adjacent selections of
the same ticker can overlap. The UI's 25-row display threshold is not statistical
sufficiency. No naive confidence intervals are added. A future uncertainty study
should predeclare paired entry-date block resampling long enough for overlapping
horizons, with regime sensitivity checks.

No fees, spreads, slippage, liquidity constraints, capital allocation, simultaneous
position limits or reinvestment are modeled. Daily bars are means of signal excess
returns, not portfolio returns. The running-sum overlay is no longer displayed.

## Versioning and operational follow-up

Summary/daily ranking namespaces, leaderboard entry modes, API cache keys and
frontend caches include methodology. Readers require current namespace and
matching summary payload; legacy rows stay for audit and never act as fallback.
Writers add JSONB payload storage; primary summary reads do not migrate schemas.
No production migration or recomputation was performed for this change. After a
separately approved deployment and authorized scheduled refresh, new-method rows
can appear; an empty ranking study until then is correct.

Bars may be revised; provider feed/adjustment settings can change. Recorded prices
are provenance, not an immutable provider archive. A durable point-in-time bar
archive, source-snapshot completeness, calendar extension and real execution-cost
research are follow-up work before stronger claims or a longer sample.

The morning digest uses the same versioned explicit fields and coverage. Its
150-occurrence/eight-date gate is a display convention, not a significance test.
