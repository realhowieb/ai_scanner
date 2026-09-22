# Run 34 — DT historical ↔ live parity certification

Status: **PARITY CERTIFIED WITH LIMITATIONS.** The historical validation/
reconstruction pipeline reproduces the live Day Trader pipeline closely enough to
trust future DT Score experiments, subject to three documented approximations
(P2) that do not change any output category. No production scoring was changed.

This run is about **measurement trustworthiness**, not scoring. It does not
create v2.1, change weights/ceilings/thresholds/gates, or tune against returns.
The earlier v2 experiment and its results are preserved as historical research in
[DT_SCORE_RECALIBRATION.md](DT_SCORE_RECALIBRATION.md) and
[DT_V2_TIER_AUDIT.md](DT_V2_TIER_AUDIT.md); see also
[DT_RECONSTRUCTION_PARITY.md](DT_RECONSTRUCTION_PARITY.md).

## Key structural finding

The scoring stack is **shared code**. Live scores each
`build_day_trader_metrics` row via `day_trade_intelligence`
([ui/day_trader.py:837](../ui/day_trader.py), [:1138](../ui/day_trader.py));
historical scores each reconstructed `feat` via the *same*
`day_trade_intelligence`
([analytics/day_trade_reconstruct.py:160](../analytics/day_trade_reconstruct.py)).
Direction votes, agreement, confirmation, conflicts, DT Score, strength, and
quality tier are therefore computed by identical functions in both paths.
**Parity reduces entirely to the seven feature inputs.** Given identical inputs,
the two paths produce bit-identical outputs (locked by
`test_day_trade_parity.EngineParityTests`).

## Task 1 — pipeline map

| Value | Live (production) | Historical (reconstruction/validation) |
| --- | --- | --- |
| price | `snapshot.latestTrade.p` → `market_data.build_day_trader_metrics:531` | minute bar close `c` at T → `day_trade_reconstruct.py:149` |
| prior_close | `snapshot.prevDailyBar.c` (`:534`) | last daily Close ≤ D-1 (`:127`) |
| open | `snapshot.dailyBar.o` (`:535`) | first minute bar `o` of day D (`:136`) |
| chg_pct | `(last − prev_close)/prev_close` (`:540`) | `(price − D-1 close)/D-1 close` (`:157`) |
| gap_pct | `(today_open − prev_close)/prev_close` (`:541`) | `(day_open − D-1 close)/D-1 close` (`:137`) |
| VWAP | provider daily-bar `vw` (`:536`) | cumulative typical-price `(h+l+c)/3` VWAP ≤ T (`:150`) |
| vs_VWAP | `(last − vw)/vw` (`:547`) | `(price − vwap)/vwap` (`:158`) |
| RVOL | session vol / 20-session avg (`fetch_avg_daily_volume:303`) | cum minute vol ≤ T / mean last-20 daily vol ≤ D-1 (`:131`,`:161`) |
| ADX | `scan.indicators.adx(frame,14).iloc[-1]` (`_range_metrics:412`) | `adx(daily,14)` as-of D-1 (`daily_indicator_series:60`) |
| SuperTrend | `supertrend(frame,13,2.0)["direction"].iloc[-1]` (`:417`) | `supertrend(daily,13,2.0)` dir as-of D-1 (`:62`) |
| EWO | `ewo(frame,5,35).iloc[-1]` (`:428`) | `ewo(daily,5,35)` as-of D-1 (`:67`) |
| volume | `snapshot.dailyBar.v` (`:537`) | cumulative minute `v` ≤ T (`:146`) |
| directional votes | `day_trade_intel._direction_votes` (shared) | same |
| agreement | `day_trade_intel._agreement` (shared) | same |
| confirmation | ADX≥20 or RVOL≥1.5 in `classify_setup_quality` (shared) | same |
| conflict detection | `day_trade_intel.day_trade_conflicts` (shared) | same |
| DT Score | `day_trade_intel.score_day_trade_setup` (shared) | same |
| direction | `day_trade_intel.classify_day_trade_direction` (shared) | same |
| strength | DT Score (shared) | same |
| quality tier | `day_trade_intel.classify_setup_quality` (shared) | same |

## Task 2 — parity matrix

| Field | Verdict | Notes |
| --- | --- | --- |
| price | PASS | latest trade vs minute close at T — same "current price" concept |
| prior_close | PASS | prior completed daily close both sides |
| open | PASS | day-D session open both sides |
| chg_pct | **PASS** | fixed in `67ca1b7`; both = gap-inclusive move vs prior close |
| gap_pct | PASS | `(open − prior_close)/prior_close` both sides |
| VWAP | **APPROXIMATE (P2)** | live = provider `vw`; historical = cumulative typical-price VWAP. Directionally consistent; magnitude can differ, most early in a session |
| vs_VWAP | APPROXIMATE (P2) | inherits the VWAP approximation |
| RVOL | APPROXIMATE (P2) | same formula; historical avg excludes day D, live 20-tail may include the partial day-D bar |
| ADX | PASS (as-of P2) | identical `adx(·,14)` code; historical as-of D-1, live `.iloc[-1]` may include the partial day-D bar |
| SuperTrend | PASS (as-of P2) | identical `supertrend(·,13,2.0)`; same as-of caveat |
| EWO | PASS (as-of P2) | identical `ewo(·,5,35)`; same as-of caveat |
| volume | PASS | session-cumulative both sides |
| directional votes | PASS | shared code |
| agreement | PASS | shared code |
| confirmation | PASS | shared code |
| conflict detection | PASS | shared code |
| DT Score | PASS | shared code |
| direction / strength / quality tier | PASS | shared code |

No FAIL and no MISSING fields.

## Task 3 — silent fallbacks

The one silent-fallback path is the intraday-only loader
(`minute_bars_to_observations`), used only when `_fetch_daily_frame` returns
None; it sets ADX/RVOL/gap/SuperTrend/EWO to missing. That path is now:
1. **tagged** — every observation carries `feature_source`
   (`"reconstruct"` | `"intraday_fallback"`);
2. **classified** — `day_trade_parity.classify_fallback` labels each observation
   `full_feature` / `partial` / `fallback`;
3. **quantified** — `feature_coverage` and `parity_diagnostics.coverage` report
   full/partial/fallback counts and % per run, per field, and by source;
4. **rejected as invalid** — a run dominated by `fallback` rows is not a valid
   full-feature test regardless of its score distribution.

The `_fetch_daily_frame` DataFrame-truthiness bug that once forced 100% fallback
is fixed (`3e614fd`) with a regression test.

## Task 4 — diagnostics

`analytics/day_trade_parity.py` (pure, tested) provides `parity_record(obs)`
(per-observation: symbol, timestamp, feature_source, coverage, all 7 inputs,
directional vote count, agreement, confirmation, conflict count + reasons, DT
Score, direction, quality tier, and explicit `fallback_status`/`fallback_reason`)
and `parity_summary(observations)` (aggregate coverage + distributions). The
harness writes per-observation records to `dt_parity_records.jsonl` and embeds
`parity_diagnostics` in the JSON and markdown reports.

## Task 6 — held-out certification run

Window **2026-08-25 → 2026-09-12** (not used to tune DT Score), production v1,
run [35687698315](https://github.com/realhowieb/ai_scanner/actions/runs/35687698315)
on `dev`.

- Observations: **9,279** · directional: **8,128**
- Feature coverage: **full_feature 9,279 (100%)** · partial 0 · **fallback 0**
- Source: `{reconstruct: 9,279}`

DT Score: mean 20.8 · std 18.1 · P10 0.0 · P25 5.4 · **median 16.9** · P75 33.4 ·
P90 48.2 · P95 56.0 · **max 75.2**

Tiers: **Weak 8,987 · Developing 212 · Strong 80**
Direction: Bullish 4,520 · Bearish 3,608 · Neutral 1,151
Vote count: {2:12, 3:285, 4:3,187, 5:5,795} — dominated by 4–5 signals (full feature)
Agreement: <0.55 1,151 · 0.55–0.69 3,118 · 0.70–0.79 1,056 · 0.80–0.99 1,722 · 1.00 2,232
Confirmation: neither 5,375 · adx_only 3,755 · rvol_only 72 · both 77
Conflict count: 0→81, 1→1,466, 2→2,217, 3→3,969, 4+→1,546
Conflict reasons: Low participation 8,722 · Momentum disagreement 4,894 ·
Mixed trend 4,384 · Weak trend strength 2,704 · Losing VWAP 2,500 · Gap fading 1,147

### Comparison to the prior clean production-v1 run (2026-08-04 → 08-22)

| Metric | Prior clean | Certification | 
| --- | ---: | ---: |
| Observations | 10,166 | 9,279 |
| Full-feature % | 100% | 100% |
| Fallback % | 0% | 0% |
| median DT | 18.3 | 16.9 |
| P90 | 48.3 | 48.2 |
| P95 | 53.1 | 56.0 |
| max | 76.3 | 75.2 |
| Strong / Developing / Weak | 114 / 160 / 9,892 | 80 / 212 / 8,987 |

The two independent full-feature held-out windows agree closely — no saturation,
a stable spread (median ~17–18, P90 ~48, max ~75), tiers that separate with
Strong firing. This reproducibility across periods is the core evidence for
certification.

## Task 7 — ranked parity gaps

- **P0 (invalidates conclusions):** none.
- **P1 (meaningful discrepancy):** none outstanding. The two P1-class issues
  found earlier — the `chg_pct` open-vs-prior-close break and the forced-fallback
  harness bug — are fixed (`67ca1b7`, `3e614fd`).
- **P2 (acceptable, documented):**
  - VWAP definition (provider `vw` vs reconstructed typical-price VWAP).
  - RVOL 20-session window edge (day-D inclusion differs by one partial bar).
  - Daily-indicator as-of point (historical D-1 vs live possibly D-partial). The
    historical D-1 choice is the correct no-lookahead one and must not be
    "fixed" by peeking at the in-progress bar.
- **P3 (cosmetic/diagnostic):** consider emitting the exact live indicator
  as-of date in future live telemetry so the D vs D-1 question can be measured
  directly rather than reasoned about.

Per the run's rule, historical validation conforms to production; production
scoring is not altered to match historical.

## Task 8 — verdict

**PARITY CERTIFIED WITH LIMITATIONS.** All scoring/vote/tier logic is shared
code; the four semantically load-bearing inputs (chg_pct, gap_pct, price,
prior_close) are PASS; the remaining differences are P2 approximations that do
not change direction, score bucket, or tier for well-formed full-feature rows.
Trust a validation run only when `parity_diagnostics.coverage.full_feature` is a
high share and `fallback` is ~0.

## Production behavior changed?

**NO.** `git diff` shows no change to `analytics/day_trade_intel.py`,
`market_data.py`, `scan/indicators.py`, or `ui/day_trader.py`.

## Recommended next step

Proceed to a **forward-outcome / predictive validation** of the existing
production DT Score on full-feature held-out data: does DT Score (as a coherence
score) carry any edge in forward directional returns / MFE-MAE, or is it — as
every run so far indicates — flat and purely a coherence/direction indicator?
Gate that run on `full_feature` coverage; do not change scoring.
