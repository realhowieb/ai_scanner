# Observation integrity & research readiness (Run 46)

Read-only audit of HSF's canonical historical observation pipeline: what is
stored, how complete it is, whether it is point-in-time safe, and whether it can
support trustworthy future research. **No scoring/scanner/ML/intelligence change;
no historical data is mutated, deduplicated, or deleted.**

## Observation pipeline (architecture map)

```
US_MARKET (data/us_market_universe) → price fetch (scan.engine + data.prices)
  → breakout scan (scan.breakout) → filters/ranking → top-N candidate rows (df)
  → analytics.observation_capture.build_scan_observations
       (analytics.hsf_observation.build_observation)
  → db.hsf_observations.save_observations_batch  (Neon JSONB / SQLite)
  → consumers: analytics.replay (Historical Replay), analytics.scanner_performance,
               scripts.mature_observations (outcomes, SEPARATE table)
```
Creation `analytics/hsf_observation.py`; capture `analytics/observation_capture.py`;
persistence/query `db/hsf_observations.py` (`save_observations_batch`,
`load_recent_observations`, `load_observations_for_symbol`); dedup = deterministic
`observation_id` + first-write-wins; replay `analytics/replay.py`; outcomes
`analytics/observation_capture.compute_matured_outcomes` +
`hsf_observation_outcomes` table; integrity/export `analytics/observation_integrity.py`.

## Current capture strategy — **Option A (candidate-only)**

Capture serializes `results.to_dict("records")` — the scan's **top-N candidate
rows only** (`CRON_TOP_N`, default 100) — not every evaluated symbol. Verified in
`scheduler/cron_runner.py` (`rows = results.to_dict(...)` → `capture_scan_observations`).

- Successfully evaluated per scan (live Run 44/45): ~11,631 priced symbols.
- Persisted per scan: ~100 candidate observations.
- **observation_capture_rate ≈ 100 / 11,631 ≈ 0.86%** per scan (by design).

This is intentional (candidate research), but carries **selection bias** — no
negative/control examples for "why didn't this fire?". Recommendation below.

## Schema inventory

`schema_inventory()` classifies every field into IDENTITY / MARKET_STATE /
TECHNICAL / HSF_INTELLIGENCE / ML / CONTEXT / PROVENANCE / OUTCOMES with type,
source, nullability, point-in-time safety, and expected population. Key rows:
identity (observation_id/symbol/timestamp non-null); market (price/volume from the
candidate row); technical (rvol/vs_vwap/adx/gap/chg present; **ema9/ema21/rsi/
supertrend/ewo/vwap absent** — the breakout candidate df doesn't carry them);
ML (prebreakout probability); provenance (`versions`, `data_quality`); outcomes
(separate table, `point_in_time_safe=false`).

## Point-in-time integrity

Each observation stores features known at T; **outcomes live in a separate table**
(`hsf_observation_outcomes`), never inside the feature record.
`check_point_in_time()` flags (a) any outcome marker (`raw_return`, `mfe`, `mae`,
`future_high/low`, `return_`) inside market/indicators/scanners/models, and (b) a
MATURED outcome whose `evaluation_time ≤ observation anchor`. Tests enforce both.

## Timestamp integrity

`timestamp` = hour-bucketed scan time (dedupe key), `scan_timestamp` = precise
scan start (maturation anchor), DB `created_at` = write time (distinct — never
used as market-data time). All parsed to tz-aware UTC. Bucketing collapses
workflow retries within a slot; premarket/postmarket route to the session scan
(not US_MARKET), so US_MARKET observations are regular-session.

## Duplicate semantics

Logical key = `observation_id` = sha256(symbol | hour-bucketed-timestamp |
context). Store PK is first-write-wins, so production cannot persist duplicates.
`duplicate_analysis()` still reports **exact** (same id, same features) vs
**conflicting** (same id, materially different features) without dropping
anything — it identifies, not hides.

## Completeness & validity

`completeness_report()` gives per-field population %. `validate_observation()`
flags price≤0, volume<0, rvol<0, RSI∉[0,100], probability∉[0,100], non-finite,
malformed ticker, invalid direction/lifecycle, missing identity. **Validation
only — no signal is recomputed.** Expected missingness (ema/rsi/supertrend from
the candidate path) is distinguished from suspicious missingness by the schema
inventory's `expected population`.

## Provenance

Every observation carries `versions` (schema `hsf-obs-1.0`, `hsf_score`,
`prebreakout_model` e.g. `prebreakout-xgb-v16`, `ai_confidence_model`, `dt_score`)
plus per-scanner `version` and `data_quality`. Future research can distinguish
observations generated under materially different HSF logic.

## Scan-health linkage

`market_context` carries `scan_id` + `coverage_health` (Run 37/45). The dataset
health report and research export surface these, and `research_export(healthy_only
=True)` excludes DEGRADED-scan observations, so a degraded scan never silently
appears equivalent to a healthy full-market cycle.

## Data scale (from actual config)

Scheduled US_MARKET runs ~3 regular slots/weekday (14:35 / 17:00 / 20:30 UTC;
21:10 is postmarket → session scan). Candidate-only:
- ~100 obs/scan × ~3 = **~300 obs/day** → ~6k/month → ~75k/year (~2 KB each ⇒
  < 200 MB/year). Negligible.

Full-universe capture (Option B) would be ~11,600 × 3 = **~35k obs/day** → ~1M/
month → ~12M/year (~24 GB/year) and a **~116× write increase** per scan.
Feasible on Neon but materially heavier; not justified without a research need.

## Recommended future capture strategy — **Option C (tiered)**

Keep full candidate records (Option A) and, in a later run, add a compact,
deterministically-sampled snapshot of **non-candidate** symbols per scan (negative/
control examples) — bounded (e.g. a fixed sample or near-miss band), not the whole
universe. This gives research negative examples without the ~116× write blow-up.
**Not implemented in Run 46** — this run measures and recommends; it does not
change capture volume.

## Research export

`research_export()` / `scripts.observation_health --export`: stable column set
(`EXPORT_COLUMNS`), deterministic ordering (timestamp, symbol), **features known
at T only — outcome fields excluded by default**, optional `healthy_only` and
date range. Suitable for Run 47 feature/outcome joins by `observation_id`.

## Outcome separation (Run 47 readiness)

Features (observation) and outcomes (`hsf_observation_outcomes`) are already
physically separate, joined only by `observation_id`. Run 47 can attach +5m/+15m/
+30m/+60m/EOD/MFE/MAE outcomes without ever contaminating the point-in-time
feature record. The architecture is ready.

## Known limitations

- **Selection bias:** candidate-only capture (Option A) → no negative examples.
- **Sparse technicals:** the breakout candidate row lacks EMA/RSI/SuperTrend/EWO/
  VWAP (expected 0% population), so those columns are null in observations.
- **Live audit pending on main:** the read-only `observation-health` workflow must
  land on the default branch before it can be dispatched (workflow_dispatch
  constraint); until then, validation is NON-LIVE fixtures.

## Running the report

```bash
python -m scripts.observation_health --limit 50000            # dataset health
python -m scripts.observation_health --export out.json --healthy-only
```
or dispatch `.github/workflows/observation-health.yml` (reads Neon).
