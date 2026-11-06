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
Score all cannot be combined with RSI or rebound filters. RSI values appear in
custom results when either screen is active. Existing saved views without these
fields retain their behavior.

Before claiming a predictive edge, evaluate the fixed rule against a baseline
using time-separated observations, transaction costs, and drawdowns. Keep the
rule experimental until that evidence supports a stronger claim.
