# DT Score validation — reconstruction ↔ live feature parity

Status: **audit only; no production or scoring code changed.** This documents
how the Run 33 historical reconstruction
(`analytics/day_trade_reconstruct.py`) compares, field by field, to the live
Day Trader feature builder (`market_data.build_day_trader_metrics` +
`market_data._range_metrics`) that the scanner scores in production. It exists so
a held-out validation result can be trusted only over the rows that actually
match live fidelity — the distinction the earlier harness bug (`f661b04` audit,
[DT_V2_TIER_AUDIT.md](DT_V2_TIER_AUDIT.md)) obscured.

## The seven classifier inputs

`analytics.day_trade_intel` scores from exactly:
`chg_pct, gap_pct, rvol, vs_vwap_pct, adx, supertrend_direction, ewo`.

| Field | Live (`build_day_trader_metrics` / `_range_metrics`) | Reconstruct (`reconstruct_observations`) | Parity |
| --- | --- | --- | --- |
| `chg_pct` | `(last − prev_close) / prev_close` — total daily change vs **yesterday's close** (gap-inclusive) | `(price − day_open) / day_open` — intraday change **since today's open** (gap-exclusive) | ❌ **Mismatch (high)** |
| `gap_pct` | `(today_open − prev_close) / prev_close` | `(day_open − D-1 close) / D-1 close` | ✅ Aligned |
| `vs_vwap_pct` | `(last − vw) / vw`, where `vw` = Alpaca daily-bar provider VWAP | `(price − vwap) / vwap`, `vwap` = cumulative typical-price `(h+l+c)/3` VWAP from minute bars ≤ T | ⚠️ **Proxy (medium)** |
| `rvol` | session cumulative volume / `fetch_avg_daily_volume` (period 20) | cumulative intraday volume ≤ T / mean of last 20 daily volumes ≤ D-1 | ⚠️ Close (low) |
| `adx` | `adx(frame,14).iloc[-1]` on a live 150d/1d frame | `adx(daily,14)` as-of **D-1** | ⚠️ As-of point (medium) |
| `supertrend_direction` | `supertrend(frame,13,2.0)["direction"].iloc[-1]` | `supertrend(daily,13,2.0)` direction as-of **D-1** | ⚠️ As-of point (medium) |
| `ewo` | `ewo(frame,5,35).iloc[-1]` | `ewo(daily,5,35)` as-of **D-1** | ⚠️ As-of point (medium) |

Indicator **periods** all match (ADX 14, SuperTrend 13/2.0, EWO 5/35). The gaps
are in *how the value is derived*, not which indicator.

## Findings, by impact

### F1 — `chg_pct` semantics differ (HIGH)
Live `chg_pct` is the **whole-day** move measured from the prior close, so it
includes the opening gap. Reconstruct measures the **intraday** move from
today's open. For a stock that gapped +2% and then drifted +1% intraday, live
sees `chg_pct ≈ +3%` while reconstruct sees `≈ +1%`.

`chg_pct` drives the `momentum` direction vote, the `momentum` sub-score, and
three conflict rules (`Losing VWAP`, `Momentum disagreement`, `Gap fading`). A
systematic difference here shifts votes, agreement, the score, and the tier for a
non-trivial share of rows — a plausible contributor to the corrected run's
low-score / mostly-Weak skew. **This is a real parity break and the top
candidate to fix** (recompute reconstruct `chg_pct` from the prior daily close,
not the intraday open) before trusting any tier comparison.

### F2 — `vs_vwap_pct` uses a different VWAP (MEDIUM)
Live consumes Alpaca's provider `vw` (a trade-price session VWAP). Reconstruct
builds a cumulative **typical-price** VWAP from minute bars. Directionally
consistent, but the magnitude (and therefore the VWAP sub-score and the
`Mixed trend` / `Losing VWAP` conflict edges) can differ, especially early in a
session when few bars have accumulated.

### F3 — daily-indicator as-of point (MEDIUM)
Reconstruct deliberately carries ADX/SuperTrend/EWO as-of the **prior completed
session (D-1)** to guarantee no lookahead. Live computes `.iloc[-1]` on a
150d/1d frame fetched *during* the session; if that frame includes today's
in-progress (partial) daily bar, live indicators are effectively as-of **D
(partial)**. That is a genuine live↔validation difference, but the fix is **not**
to introduce lookahead into the backtest — it is to (a) confirm whether the live
frame includes the partial bar, and (b) if so, treat live's within-session
indicator as a partial-bar artifact rather than a target the backtest should
match. Document, do not "correct" by peeking.

### F4 — RVOL average-volume window (LOW)
Both use a 20-session average of daily volume over current-session cumulative
volume. Minor differences possible in whether the live `fetch_avg_daily_volume`
window includes the current day; unlikely to move tiers materially.

## Coverage logging (added in this change)

Every observation is now tagged with `feature_source` (`"reconstruct"` for the
full-feature path, `"intraday_fallback"` for the minute-only subset). The
validation report gains a `feature_coverage` block
(`analytics.day_trade_validation.feature_coverage`, pure/tested) reporting, per
run:

- per-field present count and %,
- **full-feature** rows (all 7 inputs present — true live fidelity),
- **daily-missing** rows (every daily-derived input absent — the intraday-only
  fallback, which can never fire Strong because it has no ADX/RVOL
  confirmation), and
- breakdowns `by_source` and `by_symbol`.

The harness prints a one-line coverage summary and renders a coverage table in
`day_trader_validation.md`. This makes silent fallback impossible to miss: a run
dominated by `daily_missing` / `intraday_fallback` rows is not a valid
full-feature test, regardless of its aggregate score distribution.

## Recommended order (unchanged from the tier audit)
1. **Parity first.** Fix F1 (`chg_pct` base) so reconstruct feeds the classifier
   the same momentum signal live does; decide F2/F3 explicitly (proxy accepted,
   or aligned) and record the choice. Do not tune tier constants yet.
2. **Re-baseline v1** on a held-out window restricted to `full_feature` rows
   (now measurable via coverage logging).
3. **Only then** revisit any score/tier candidate, with fresh held-out
   validation. Current evidence does not justify a scoring change.

No production scoring, PreBreakout, or scanner code is modified by this audit.
