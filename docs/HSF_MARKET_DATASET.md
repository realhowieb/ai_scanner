# HSF Market Dataset — data architecture, canonical schema & outcomes

Run 36. Purpose: turn the data HSF already generates into a durable, versioned
market-intelligence dataset for future validation, ML research, alerts, and
analytics. This run is **data architecture + audit**; it introduces a canonical
schema and an opt-in store and changes **no** production scan, scoring, model, or
UI behavior.

DT Score research is CLOSED — DT Score is treated here purely as a technical
coherence/direction indicator (see
[DT_PREDICTIVE_VALIDATION.md](DT_PREDICTIVE_VALIDATION.md)).

## 1. Current data-flow map

Backends: **Neon Postgres** (production) + **SQLite** (`scanner.sqlite`, local/CI)
via `db/engine.py` (`get_neon_conn` / `get_sqlite_conn`, warm-pooled).

| Dataset | Source | Transform | Storage | Retention | Consumers | Survives run? |
| --- | --- | --- | --- | --- | --- | --- |
| Scan results (rows) | Alpaca snapshots/bars → `scan.engine`, `market_data.build_day_trader_metrics` | scoring, ranking | `runs.results_json` (`db/runs.py`) | **non-snapshot runs DELETED ~daily** (`cron_runner.py:424`); one `is_snapshot` row per universe/day kept | Scanner UI, track record, digests | **partial** — only the daily snapshot |
| Daily snapshot | promoted scan run | idempotent per universe/day | `runs` (`is_snapshot=TRUE`) | long-lived | track record, leaderboard, digest | yes |
| Opportunity snapshot | Market Brief / Scanner opportunities | dedupe → (ticker, score, status) | `opportunity_snapshots` (Neon, JSONB) | long-lived | score-movement / status transitions | yes (thin) |
| Frozen opportunities | signal-time opportunities | freeze | `signal_outcomes` (`source='opportunity'`) | long-lived | outcome maturation | yes |
| Opportunity outcomes | frozen obs + later snapshot | classify (persist/fade/…) | `hsf_opportunity_outcomes` | long-lived, idempotent, first-write-immutable | outcome intelligence, Stock Intelligence | yes |
| Alert outcomes | fired alerts | evaluate | `intelligence_alert_outcomes` / `alert_outcomes` | long-lived | alert quality | yes |
| PreBreakout | `ml_prebreakout` (xgb) | probability + isotonic calib | inside scan rows / calibration records | with the run | Scanner, Market Brief | partial |
| AI confidence | `scan/ai_confidence` (xgb) | probability | inside scan rows | with the run | Scanner | partial |
| DT intel | `analytics/day_trade_intel` on rows | direction/score/quality | **computed at render time** (`ui/day_trader.py:837`) | — | Day Trader UI | **NO — not persisted** |
| BTC/Kalshi outcomes | Kalshi API | settle 15-min | `btc_outcomes` | long-lived | BTC model training | yes |
| Prices | Alpaca | cache | `db/prices.py`, in-memory caches | short | scans | partial |
| Scan summaries | cron | JSON | GitHub Actions artifacts (`artifacts/automation/latest_scan.json`) | **Actions retention (~90d) then gone** | automation | **NO (expires)** |
| Streamlit session/cache | UI | `@st.cache_data` (ttl) | process memory | ephemeral | UI | no |

## 2. Existing data inventory (verified in code)

Present and available at scan/observation time (from `build_day_trader_metrics`
+ `_range_metrics` + scan rows + models):

- **Identity/time:** ticker, trade_ts, snapshot_time (opportunity), scan run time.
- **Market:** last (price), previous_close, open, close_today, high/low (bars),
  volume, change_dollar.
- **Intraday:** chg_pct, gap_pct, vwap, vs_vwap_pct, rvol.
- **Daily indicators:** adx (14), supertrend_direction (13,2), ewo (5,35),
  ema_cross (9/21), atr_pct, donchian_pos, bb_pctb, bb_squeeze.
- **Scanner:** BreakoutScore and related columns (breakout scan); opportunity
  score + status.
- **Models:** PreBreakout probability/raw (`prebreakout-xgb-v16`), AI confidence
  (`ai-confidence-xgb-v1`).
- **DT:** DT Score, DT direction, DT quality (computed in UI; run32 v1).
- **Versioning present:** `HSF_SCORE_VERSION="1.0"`, PreBreakout `MODEL_VERSION`,
  AI-confidence `MODEL_VERSION`, per-record `score_version` in the opportunity path.

Verified **absent** as first-class persisted fields: EMA9/EMA21 numeric values
(only the cross label), RSI, an explicit market-session tag, sector/industry, a
unified feature/schema version, and DT intel outputs (never persisted).

## 3. Data currently being LOST

1. **Intraday scan granularity** — non-snapshot `runs` are pruned daily; only one
   daily snapshot per universe survives. Every intraday scan's per-row feature
   vector is discarded.
2. **DT intel outputs** — DT Score/direction/quality are computed at UI render
   time and never written anywhere.
3. **Full computed context in opportunity snapshots** — only (ticker, score,
   status) is kept; indicators/model outputs at that instant are dropped.
4. **Model outputs at signal time** — PreBreakout/AI-confidence values live only
   inside `results_json`, so they vanish with the pruned run.
5. **Scan-summary artifacts** — expire with GitHub Actions retention.
6. **No canonical, typed, versioned observation** with data-quality metadata
   existed before this run.

## 4. Canonical observation schema (`analytics/hsf_observation.py`)

`OBSERVATION_SCHEMA_VERSION = "hsf-obs-1.0"`. One immutable record per
`(symbol, timestamp, context)`; `observation_id` = deterministic
sha256(symbol|timestamp|context)[:16] (idempotent dedupe).

```
{
  schema_version, observation_id, symbol, timestamp, scan_timestamp,
  context, session, universe_version,
  market:        {price, previous_close, open, high, low, volume},
  indicators:    {ema9, ema21, rsi, rvol, adx, vwap, vs_vwap_pct,
                  supertrend_direction, ewo, gap_pct, chg_pct, atr_pct},
  scanners:      [ {name, version, triggered, score, rank, ...}, ... ],  # multi
  models:        {prebreakout:{version,probability,...}, ai_confidence:{...}},
  market_context:{market_regime, sector, watchlist, ...},
  versions:      {schema, hsf_score, prebreakout_model, ai_confidence_model,
                  dt_score},
  data_quality:  {feature_completeness, present_fields, missing_fields,
                  fallback_used, fallback_reason, stale, price_timestamp,
                  data_source},
}
```

Multiple scanners firing on one symbol are a **list** — no information lost. Only
recognized, present fields are copied; missing fields are recorded, never
zero-filled. Immutable: a scoring change bumps a version and writes a NEW
observation.

## 5. Outcome schema

`OUTCOME_SCHEMA_VERSION = "hsf-outcome-1.0"`. Separate namespace, attached AFTER
the fact via `attach_outcome` (returns a copy; never mutates features).

```
{
  schema_version, observation_id, symbol, observation_timestamp,
  horizon,              # +5m | +15m | +30m | +60m | market_close | next_trading_day
  evaluation_time,      # MUST be strictly after observation_timestamp
  raw_return, directional_return, mfe, mae, future_high, future_low, hit,
  data_status,          # MATURED | PENDING | UNAVAILABLE
}
```

**Lookahead is structurally prevented:** `build_outcome`/`attach_outcome` raise if
a MATURED outcome's `evaluation_time` is at or before the observation timestamp.
Return/MFE/MAE computation reuses `analytics.day_trade_validation`
(`forward_returns`, `mfe_mae`, `directional_return`), which measures only from
bars strictly after the signal.

## 6. Persistence (`db/hsf_observations.py`) — opt-in, not yet wired

Dual-backend (Neon JSONB / SQLite TEXT), idempotent, non-fatal. Two tables:
`hsf_observations` (PK `observation_id`, first-write-wins) and
`hsf_observation_outcomes` (PK `(observation_id, horizon)`, first-write-wins). API:
`save_observation`, `save_outcome`, `load_observation` (rehydrates outcomes),
`load_recent_observations`. Accepts an explicit `conn` for tests.

**Activated for scheduled scans in Run 38A** — `scheduler.cron_runner.run_and_save`
now captures per-scanner observations (side-effect only, non-fatal, hour-bucketed
dedupe) via `analytics.observation_capture` + `save_observations_batch`. See
[PRODUCTION_OBSERVATION_CAPTURE.md](PRODUCTION_OBSERVATION_CAPTURE.md). Manual/UI
paths remain intentionally uncaptured. Outcome maturation worker
(`scripts/mature_observations.py`) is ready but not yet scheduled.

## 7. Versioning strategy

Reuse existing identifiers; never overwrite an observation when logic changes —
bump the version and write a new record. `resolve_versions()` snapshots:
`schema` (hsf-obs-1.0), `hsf_score` (1.0), `prebreakout_model`
(prebreakout-xgb-v16), `ai_confidence_model` (ai-confidence-xgb-v1), `dt_score`
(run32-v1). Each scanner entry also carries its own `version`.

## 8. Retention strategy

- Observations & outcomes: **append-only, immutable, long-lived** (no pruning) —
  this is the durable dataset the pruned `runs` table cannot provide.
- Existing `runs` pruning is unchanged; the canonical store is additive.

## 9. Known limitations

- The store is not yet populated by production (opt-in only) — Run 37.
- EMA9/EMA21 numeric values, RSI, session tag, and sector are not yet produced by
  the live builder, so they will be absent (flagged in `data_quality`) until a
  producer adds them.
- Neon JSONB vs SQLite TEXT differ in query ergonomics; heavy analytical queries
  should target Neon.
- No backfill of historical pruned runs is attempted (they are already gone).

## 10. Migration strategy

Incremental and backward-compatible: (1) schema + store shipped now (this run),
no writes wired; (2) Run 37 wires `freeze`/cron to build+save observations from
the existing scan rows (a thin adapter, no scoring change); (3) an outcome
maturation job reuses `day_trade_validation` to attach price outcomes; (4)
analytics/ML migrate to reading the canonical store. No large migration is forced
for elegance; existing stores keep working untouched.

## Examples

```python
from analytics.hsf_observation import build_observation, build_outcome, attach_outcome
from db.hsf_observations import save_observation, save_outcome

obs = build_observation(
    symbol="NVDA", timestamp="2026-08-25T13:35:00Z", context="scanner",
    market={"price": 131.0, "previous_close": 129.0, "open": 129.5},
    indicators={"rvol": 2.4, "adx": 31, "vs_vwap_pct": 0.8,
                "supertrend_direction": "green", "ewo": 12.4, "gap_pct": 0.4, "chg_pct": 1.5},
    scanners=[{"name": "day_trader", "version": "run32-v1", "score": 58.0,
               "direction": "bullish", "quality": "developing"}],
    models={"prebreakout": {"version": "prebreakout-xgb-v16", "probability": 0.27}},
    data_source="alpaca_iex")
save_observation(obs)                         # idempotent; first write wins

oc = build_outcome(observation_id=obs["observation_id"], symbol="NVDA",
                   observation_timestamp=obs["timestamp"], horizon="+15m",
                   evaluation_time="2026-08-25T13:50:00Z",
                   raw_return=0.012, directional_return=0.012, mfe=0.02, mae=-0.005, hit=True)
save_outcome(oc)                              # separate table; never edits the observation
```
