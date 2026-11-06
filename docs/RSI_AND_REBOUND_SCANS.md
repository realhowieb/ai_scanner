# Daily RSI and experimental rebound scans

Custom Scan → Advanced filters provides inclusive minimum and maximum daily
Wilder RSI(14) bounds. Leave either blank for no bound on that side. For stocks
near 30, use 28–32; for stocks at or below 30, leave the minimum blank and use a
maximum of 30. The AI scan builder and saved custom views support these fields.
Only daily period 14 is supported. The latest daily bar may still be forming.
Missing or invalid price history cannot pass an explicit RSI screen.

The experimental Rebound candidates checkbox requires all of:

- RSI at or below 30 on at least one of the previous five daily bars.
- Latest RSI greater than 30 and at most 50.
- Latest close greater than the previous close.
- Relative volume at least 1.0, using the scanner's existing volume measure.

Optional RSI bounds intersect with these rules. A maximum of 30 and rebound
cannot both match: one finds oversold stocks; the other requires recovery.
The screen finds research candidates, not predicted breakouts or validated buy
signals. It has no claimed historical return. Existing HSF scores still rank
results; no model, score formula, or research confidence gate changes.

Filters run across eligible price histories before ranking and the row cap.
Universe membership, liquidity and other selected filters still apply. With an
RSI screen active and the gap checkbox off, the implicit default gap is disabled.
Score all cannot be combined with RSI or MACD filters. RSI values appear in
custom results when either screen is active. Existing saved views without these
fields retain their behavior.

Before claiming a predictive edge, evaluate the fixed rule against a baseline
using time-separated observations, transaction costs, and drawdowns. Keep the
rule experimental until that evidence supports a stronger claim.

## Momentum recovery and uptrend pullback

Choose at most one RSI setup (enforced by the API as well as the form):

- **Momentum recovery:** previous daily RSI <=50, latest RSI >50, latest close
  above previous close, and relative volume >=1.0. The crossing must occur on
  the latest daily bar. This rule does not impose an upper RSI bound; optional
  manual bounds still intersect with it.
- **Uptrend pullback:** previous RSI in [40,50], latest RSI above previous RSI but
  still <=50, latest close above its SMA50, an up close, and relative volume >=1.0.
  At least 50 valid daily closes are required. The label refers to price above
  SMA50; it does not additionally require the SMA itself to be rising.

These are fixed experimental screening definitions, not performance claims.

## Optional MACD confirmation

Daily MACD(12,26,9) reuses `scan.indicators.macd`: EMA12 minus EMA26, its EMA9
signal, and the histogram (MACD minus signal). EMA calculations use adjust=False.
At least 35 valid closes are required before a MACD filter can match.

- **Fresh bullish crossover:** latest histogram >0 and previous histogram <=0.
- **MACD above signal line:** latest histogram >0. MACD itself may be below zero;
  this filter does not require a positive MACD line or a new crossover.

Use MACD alone or intersect it with one RSI setup and optional RSI bounds.
Unknown MACD modes and multiple RSI setups are rejected. Score all conflicts
with any active RSI/MACD filter. Other selected filters still apply. Relative
volume is latest daily volume divided by the prior 20 daily bars' average, the
same measure used by the scanner. No extra provider requests are made for these
screens. Standard MACD reference: [Fidelity's indicator guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd).

## Stock-page evidence and snapshot boundaries

Custom result ticker links retain the account-owned scan ID. The stock page
loads that scan through the existing ownership-checked job endpoint, finds the
matching ticker, and shows the saved indicator evidence and scan start time.
The evidence includes the daily bar date, RSI and previous RSI, prior five-bar
RSI low/date, daily close/change, relative volume, SMA50, and MACD readings and
checks. Saved rows also carry `daily-indicators-v1` to identify the rule set.

Latest chart-bar checks are computed separately from the chart's existing bars
and explicitly labelled. The latest chart may use more history, a different
snapshot, or a different volume feed from the selected custom scan. Never use
those newer readings to claim that an older scan qualified. Missing, expired,
unowned or older scans without evidence receive an unavailable message. A
stock absent from the selected scan also receives no historical qualification.
Ordinary ticker links continue to work without scan context. No database
migration, model retraining, or changes to HSF score/confidence gates are needed.

### Comparing calculation histories

New evidence records also include `history_bars` and `history_start`. Older
records leave these fields unknown. The scanner price fetch defaults to 60d;
the stock chart requests up to 260 displayed bars (backed by a two-year provider
fetch). Cache hits can supply different history lengths. Both paths call the
same indicator helper, but Wilder RSI and EMA-based MACD depend on the initial
history used to seed their smoothing. Identical final closes and even identical
recent bars need not yield identical RSI/MACD when the earlier input differs.
Revised bars can also matter. The displayed history count and first date expose
this difference for future records; they do not establish the cause of a
particular legacy difference without its original input history.

The evidence table applies the app's standard table styles, row headings,
column padding, wrapping, and horizontal scrolling on narrow screens. For a
selected setup, required rules appear first. Other indicator checks are in a
separate collapsed section and explicitly are not required for that scan.
A selected MACD confirmation remains in the required table. Saved qualification
is not recalculated or overwritten by newer chart readings.

### Streamlined stock-page view

When a stock opens from a custom scan, its saved scan evidence is the primary
view. Latest chart readings appear only after opening **Compare with latest**;
closing the comparison removes those details from the view. Comparisons start
collapsed for each stock/scan, including after navigating from another ticker.
Direct stock visits show the latest indicator checks without a second source
section. If saved evidence cannot be loaded, the error remains explicit and
latest readings remain available through the comparison rather than silently
replacing the selected scan. Calculation history and technical notes are under
**Calculation details**. No screening rules, fetched price histories or saved
qualifications change.
