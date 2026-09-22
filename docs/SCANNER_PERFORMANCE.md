# Scanner performance scoreboard

Run 38. Measures **which HSF scanners produce the strongest, most consistent
subsequent outcomes** — measurement only. No scanner, threshold, model, DT Score,
or Opportunity Score was changed. DT research remains CLOSED.

## Scanner inventory (verified in code)

| Scanner | Trigger logic | Score/rank | Required features | Universe | Session |
| --- | --- | --- | --- | --- | --- |
| **Breakout** (`scan.breakout`) | breakout detection | `BreakoutScore`, `IsBreakout` | daily OHLCV, indicators | scan universe | regular |
| **gap_up** (`scan.strategies`) | `GapPct > 0` | sort GapPct desc | GapPct | scan results | regular |
| **gap_down** | `GapPct < 0` | sort GapPct asc | GapPct | scan results | regular |
| **most_active** | top by dollar/volume | sort DollarVol20 | DollarVol20/Volume | scan results | regular |
| **unusual_vol** | `VolRel20 >= 2` | sort VolRel20 | VolRel20 | scan results | regular |
| **momentum** | `Trend20D% > 0 and Trend10D% > 0` | sort Trend20D% | Trend10/20D% | scan results | regular |
| **breakout_only** | `IsBreakout == True` | sort BreakoutScore | IsBreakout | scan results | regular |
| **PreBreakout** (`ml_prebreakout`) | XGB model, `prebreakout-xgb-v16` | calibrated probability | engineered features | scan universe | regular |
| **AI-ranked** (`scan.ai_confidence`) | XGB model, `ai-confidence-xgb-v1` | confidence | engineered features | scan universe | regular |
| **HSF Opportunity** (`ui.opportunities`) | composite HSF score, `HSF_SCORE_VERSION 1.0` | status + score | multi-signal | Market Brief / Scanner | regular |
| **Pre/Post** (`scan.pre_post`) | premarket/afterhours movers | — | extended-hours data | scan universe | pre/post |
| Day Trader (`day_trade_intel`) | DT coherence/direction — **research CLOSED**, excluded from ranking | DT Score | intraday | Day Trader | intraday |

## Methodology

`analytics/scanner_performance.py` (pure, reusable, machine-readable) groups
normalized outcome records by scanner and computes, per horizon: sample size,
hit rate (with **Wilson 95% CI**), average and median **direction-adjusted**
return (short setups inverted), plus average MFE/MAE and a risk/reward ratio.
`scripts/scanner_scoreboard.py` produces `artifacts/scanner_scoreboard.json`.

- **Sample-size protection (Task 4):** `MIN_SAMPLE = 30` (INSUFFICIENT below it);
  `STRONG_SAMPLE = 100` required for PROVEN. A scanner is never ranked #1 on a
  handful of observations — Wilson CIs make small-N uncertainty explicit.
- **Segmentation:** by regime (bullish/bearish/neutral), session
  (premarket/open/morning/midday/afternoon/afterhours), and liquidity — each a
  descriptive split; no regime/threshold is invented to flatter a result.
- **Overlap (Task 8):** records are grouped by (symbol, day); co-fire outcomes
  are compared to the single-scanner baseline and bucketed by agreement count
  (1 / 2 / 3+). Only combinations with ≥ MIN_SAMPLE are reported.

### Classification (evidence-based)
- **PROVEN** — n ≥ 100, hit-rate CI lower bound > 0.5, positive average return.
- **PROMISING** — point-estimate edge (hit > 0.5, positive return) but CI still
  straddles 0.5.
- **NEUTRAL** — no clear edge either way.
- **WEAK** — CI upper bound < 0.5, or clearly negative average with hit < 0.5.
- **INSUFFICIENT_DATA** — n < 30.

## Data availability (the key limitation)

The scoreboard runs on **matured outcome data**. Verified in code, the only
populated price-outcome source today is the `signal_outcomes` table, written for:
- `source="opportunity"` → `hsf_opportunity` (frozen HSF opportunities), and
- `source="alert_event"` → per alert type.

Outcomes there are **daily** horizons (`return_1d/3d/5d`, `mfe_5d`, `mae_5d`).

Therefore, **today**:
- **HSF Opportunity** and **alerts** are evaluable (daily horizons), subject to N.
- The six strategy filters (gap_up, gap_down, most_active, unusual_vol, momentum,
  breakout_only), **PreBreakout as a standalone scanner**, and all **intraday**
  horizons (+5m…+60m) are **INSUFFICIENT_DATA** — those signals are not persisted
  per-scanner with price outcomes yet. The Run 36 canonical observation/outcome
  store (`db.hsf_observations`) is designed to capture them but is **not yet
  populated** (its wiring is the recommended next step). The engine already reads
  it via `from_canonical_observations` the moment it is populated.

No result is fabricated: horizons/scanners without matured data report `None` /
`INSUFFICIENT_DATA`. In an environment without the production DB, the script
emits `{"status": "INSUFFICIENT_DATA", "n_records": 0}`.

**Update (Run 38A):** scheduled scans now capture per-scanner canonical
observations (see [PRODUCTION_OBSERVATION_CAPTURE.md](PRODUCTION_OBSERVATION_CAPTURE.md)),
and once the maturation worker is scheduled the scoreboard's
`from_canonical_observations` adapter will feed it real multi-scanner intraday
data. The clock has started; the scoreboard should be rerun only once enough
**matured, clean, complete** production observations have accumulated (see that
doc's caveat that scheduled breakout observations are currently PARTIAL quality).

## Running it

```bash
python -m scripts.scanner_scoreboard --out artifacts --primary-horizon 5d
```

Output `artifacts/scanner_scoreboard.json` (`schema: hsf-scanner-scoreboard-1.0`):
`overall`, `by_regime`, `by_session`, `by_liquidity`, and `overlap` — suitable
for a future dashboard.

## Results

To be populated from a production run against `signal_outcomes`. As of this run
(measurement infrastructure only), the credible evidence base is limited to
HSF Opportunity + alert signals at daily horizons; the per-scanner intraday
scoreboard is **pending Run 36 observation wiring**, and everything else is
correctly reported as INSUFFICIENT_DATA rather than guessed.

## Limitations

- Daily-horizon outcomes only where data exists; intraday horizons await
  canonical-store population.
- Most scanners lack per-scanner persisted outcomes today (see above).
- Regime/session/liquidity segmentation requires those fields on the records;
  `signal_outcomes` does not carry them yet, so segmentation is available only
  once observations are enriched (Run 36 schema already has the slots).
- Overlap analysis needs multiple scanners persisted per symbol/day — also
  pending canonical-store population.
