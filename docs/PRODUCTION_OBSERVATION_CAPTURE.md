# Production observation capture (Run 38A)

Activates the Run 36 canonical observation infrastructure for **scheduled
whole-market scans**, so HSF accumulates durable, versioned, per-scanner market
observations for future performance analysis (Run 39). **Side-effect only** — it
observes completed scans and never changes scanner logic, thresholds, ranking, or
output. If persistence fails or is removed, scans produce identical results.

## Capture boundary

One canonical boundary: `scheduler.cron_runner.run_and_save`, **after** the scan
completes and its results are saved (the authoritative "completed observation"
point), **before** results are handed to UI/automation transforms. The scan's
result DataFrame is the source; capture reads a copy (`to_dict("records")`) and
never mutates it. Gated by `HSF_OBSERVATION_CAPTURE` (default on; `=0` disables)
and `HSF_OBSERVATION_CAPTURE_DRYRUN` (`=1` builds + counts without writing).

## Which paths are captured

| Path | Captured? | Why |
| --- | --- | --- |
| Scheduled whole-market scans (`run_and_save`) | **YES** | Consistent, unbiased sampling; the authoritative boundary |
| Manual Streamlit scans | **NO** (by design) | A user re-pressing Scan would contaminate the research set with duplicates; no compelling reason yet (Task 9). Would be `context="manual"` if ever added |
| Market Brief / watchlist / pre-post render paths | **NO** | UI/rendering paths risk duplicate observations of the same underlying scan |

## What an observation captures

One canonical `hsf-obs-1.0` record per result (trigger) row. Fields present in
the scheduled breakout output are mapped; **absent fields are flagged missing,
never invented**:
- market: `price` (Last), `volume`.
- indicators: `gap_pct`, `chg_pct`, `rvol` (VolRel20), `atr_pct` (Volatility20D%).
- **scanners (multi):** `breakout` always; plus `gap_up`/`gap_down`/`unusual_vol`/
  `momentum`/`breakout_only`/`most_active` when the row satisfies the same
  predicates the strategy filters use (mirrored read-only — the filters are not
  called or changed). Each carries name, version, score, direction.
- market_context: `source="scheduled"`, `scan_id`, `coverage_health`.
- versions: schema, hsf_score, prebreakout_model, ai_confidence_model, dt_score.
- data_quality: completeness, missing fields, fallback, stale, source.

Note: the scheduled **breakout** scan does not compute ADX/SuperTrend/EWO/VWAP/
EMA9/EMA21/RSI or PreBreakout/AI-confidence, so those are absent and the records
are **PARTIAL quality** (correctly flagged). DT fields are not modified or added.

## Trigger vs control strategy (Task 4)

Chosen: **all triggers + a compact scan-level denominator** (option 3). Every
result row (a scanner trigger) becomes an observation; non-triggers are
represented by the Run 37 coverage denominator (`eligible`, `price_success`,
`non_trigger_estimate`) recorded in the capture stats, **not** as millions of
per-symbol rows. Full per-symbol control sampling would require the scan engine
to expose its pre-filter computed frame (a future change); it is intentionally
deferred to avoid unbounded storage.

## Storage estimate

- Scheduled cadence: 4 slots/day × Mon–Fri × 3 universes (SP500, NASDAQ, COMBO),
  each `top_n≈100`.
- **~1,200 trigger observations/day** (~6k/week, ~300k/year).
- ~1.5–2.5 KB JSON each ⇒ ~2–3 MB/day, **~0.5–0.75 GB/year** — well within Neon.
- Hour-bucketed ids dedupe workflow retries within a slot (slots are ≥1h apart),
  so retries add no rows. These are estimates, not guarantees.

## Duplicate control (Task 5)

Identity = deterministic `observation_id = sha256(symbol | hour-bucketed
timestamp | context)[:16]`, `context = "scheduled:<universe>"`. Batch persist is
first-write-wins (`ON CONFLICT DO NOTHING`). Retries, Streamlit reruns, and
multiple readers therefore never create duplicates. The precise scan time is kept
in `scan_timestamp` for outcome measurement.

## Failure isolation (Task 6/7)

- The whole capture block is wrapped in try/except; any error is logged +
  reported to monitoring and the scan continues unchanged.
- Persistence is one batched connection + one commit (no per-symbol network
  call); per-row errors are counted, not raised.
- A dead DB yields `write_failures` and `capture_health ∈ {DEGRADED, FAILED}`,
  never a dropped result or altered ranking.

## Outcome maturation (Task 10/11)

**Run 38B: maturation is now automated** via
`.github/workflows/mature-observations.yml` (every 30 min during market hours +
post-close sweep). See [OUTCOME_MATURATION.md](OUTCOME_MATURATION.md).

`scripts/mature_observations.py`: finds
observations whose horizons matured (age ≥ `--min-age-min`), fetches minute bars,
computes `+5m/+15m/+30m/+60m` outcomes via `analytics.observation_capture.
compute_matured_outcomes` (reuses `day_trade_validation`; structural no-lookahead
guard), and attaches them idempotently (`save_outcome`, first-write-wins).
Measured from the precise `scan_timestamp`, never the bucketed id. Safe to rerun;
skips already-matured pairs; records failures. To schedule: a daily post-close
GitHub Actions job with Alpaca secrets running `python -m
scripts.mature_observations --min-age-min 75`.

## Research data separation (Task 12) & quality gate (Task 15)

Every observation carries `context` (`scheduled:<universe>` today; `manual`/
`test`/`reconstruct` reserved) and `market_context.source`, so research can
select **only clean scheduled production observations**. `data_quality`
(completeness / missing / fallback / stale) lets Run 38 analytics filter
complete vs partial vs fallback vs stale — scheduled breakout observations are
currently PARTIAL and must be treated as such.

## Privacy (Task 13)

Market data only. No email, user id, billing, or auth data is attached.

## Observability (Task 14)

Each scan logs an `HSF Observation Capture` block (attempted / written /
duplicates / write failures / trigger obs / control denominator / quality /
capture health) and writes `artifacts/automation/coverage_<universe>_capture.json`
(uploaded by the scheduled-scans workflow).
