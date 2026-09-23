# Run 53A — production research data trust + maturation verification

Can the live production research dataset be trusted for Run 54 effectiveness
analysis? Verification/integrity run. **No scoring/ranking/ML/threshold change.**
One outcome-formula bug was proven objectively incorrect and given the smallest
justified fix (direction vocabulary); everything else is measurement or read-only
tooling.

## Executive summary — verdict: **RESEARCH DATA CONDITIONALLY TRUSTED**

- **A critical, proven outcome bug was found and fixed.** The scanner emits
  `long`/`short`; the outcome math (`directional_return`, `mfe_mae`) only
  recognized `bullish`/`bearish`. Every production matured outcome therefore has
  `directional_return = NULL`, `mfe = NULL`, `mae = NULL` — only `raw_return` is
  valid. Fixed by additive synonym normalization (`long==bullish`,
  `short==bearish`); `bullish`/`bearish` behavior is unchanged. LONG **and** SHORT
  are now verified by deterministic winner/loser tests.
- **The live per-row cohort audit (Parts 1–2) could not be executed here** — this
  environment has no production Neon credentials and CI log counts are
  secret-masked. Rather than fabricate numbers, a **read-only CI workflow**
  (`.github/workflows/research-integrity-audit.yml`) now runs the Run 53 auditor
  against Neon and publishes the report. Run it to populate the cohort table.
- **The Run 52 anti-redundancy fix is NOT deployed** (it is on `dev`; production
  `main` is at `a357829`). The latest production maturation run confirms the
  redundant-reprocessing bug is **still live** and the **backlog is growing**.
- Cohort *machinery* remains structurally correct and point-in-time-safe
  (re-confirmed from source, Run 53).

**`RUN54_READY = CONDITIONAL`** — deterministic gate in `scripts/run54_readiness.py`.

## Part 1 — production cohort audit: NOT YET RUN (no creds here)

The requested per-cohort table (total / explicitly-tagged / legacy-inferred /
distinct symbols / distinct runs / obs-per-run / first / latest / matured /
unmatured / maturity % / missing fields / invalid values / exact & conflicting
duplicates / PIT violations / leakage / missing tags) is produced by
`scripts.audit_research_cohorts` (read-only). It **cannot run from this
environment**: `get_neon_conn()` returns `None` (no `DATABASE_URL`/secrets), and
GitHub masks numeric counts in scan logs. To get real numbers without exposing
secrets to this session, run the new workflow:

```
Actions → "Research Integrity Audit" → Run workflow   (needs the DATABASE_URL secret)
```

It uploads `cohort_audit.json` and prints it to the step summary, split into
`MODERN_EXPLICIT_DATASET` (rows with an explicit `research_cohort` tag →
`explicitly_tagged`) and `LEGACY_DATASET` (`legacy_inferred`). Legacy and Run-47+
rows are **never silently combined** — the auditor reports them separately per
cohort. Run 54 selects the modern set by filtering on the explicit tag.

## Part 2 — cohort separation: code-verified; live overlap check ready

Re-confirmed from source (Run 53, unchanged):
- **CANDIDATE** = `df.head(top_n)` (`scan/engine.py:775`) — the exact production
  selection; tagged `top_n_candidate`.
- **NEAR_MISS** = `df.iloc[top_n:top_n+nm_n]` (`scan/engine.py:760`) — the ranked
  rows immediately below the cut; production output stays `df.head(top_n)`, so a
  near-miss can never enter production top-N.
- **CONTROL** = seeded `sha256(scan_run_id|symbol)` sample of evaluated
  non-candidates, `exclude=candidate_symbols` — deterministic, no outcome input.

The auditor now also computes **cohort overlap within a scan_run_id**
(`candidate_control`, `candidate_near_miss`, `near_miss_control`); a symbol tagged
both CANDIDATE and CONTROL for the same event is a hard separation failure. By
construction controls exclude candidates, so the expected value is 0 — the live
audit must confirm it. Tested on synthetic data (overlap detected / clean case).

## Part 3 — LONG/SHORT outcome verification (critical) — **FIXED + VERIFIED**

**Root cause (proven):** `analytics/observation_capture.derive_scanner_triggers`
tags scanners `direction="long"|"short"`; the maturation worker passes that string
into `analytics/day_trade_validation.directional_return()` and `mfe_mae()`, which
matched only `"bullish"`/`"bearish"` and returned `None`/`{None,None}` for anything
else. Reproduced:

```
directional_return('long',  0.05)  -> None     (should be +0.05)
directional_return('short', -0.05) -> None     (should be +0.05)
mfe_mae([100,101,102],0,2,'long')  -> {mfe:None, mae:None}
```

**Consequence for existing data:** all outcomes matured before this fix have
`directional_return/mfe/mae = NULL`. Because outcomes are immutable/first-write-wins,
re-maturation will **not** backfill them. `raw_return` is present and correct;
`directional_return` is recoverable at analysis time from `raw_return` + the
observation's direction, but **MFE/MAE are not recoverable** (bars aren't persisted).

**Fix (smallest, isolated, backward compatible):** `_norm_direction()` maps
`long→bullish`, `short→bearish` (also `buy`/`sell`), leaving `bullish`/`bearish`
and neutral behavior identical. Applied at the root (the two helpers), which also
repairs `analytics/scanner_performance.py` (same mismatch). Verified:

| case | input | raw_return | directional_return |
|---|---|---|---|
| LONG winner | long, price ↑ | + | **+** |
| LONG loser | long, price ↓ | − | **−** |
| SHORT winner | short, price ↓ | − | **+** |
| SHORT loser | short, price ↑ | + | **−** |

Deterministic tests cover all four at the helper level and end-to-end through
`compute_matured_outcomes` (MFE/MAE now populated). **Live SHORT inspection:
`SHORT_LIVE_EVIDENCE = INSUFFICIENT`** — no live short samples were confirmable
from this environment; SHORT is proven deterministically only and must be excluded
from unsupported live conclusions until the audit surfaces real short outcomes.

## Part 4 — maturation before/after Run 52 — **INSUFFICIENT_POST_FIX_EVIDENCE**

The Run 52 fix (`3e5320e`, outcome-aware loading) is on `dev`; production `main`
is at `a357829` and **does not contain it** (`git merge-base --is-ancestor` = NO).
Every scheduled maturation run executes `main`, so **no post-fix run exists in
production.**

Latest production maturation run **35918775967** (2026-09-23 21:11 UTC, main, pre-fix):

| metric | value |
|---|---|
| scanned / eligible | 2233 / 1486 |
| ready_symbols | **1042** |
| deferred (cap 400) | **642 (62%)** |
| +5m new / **already** / failed | 630 / **600** / 256 |
| +60m new / already / failed | 553 / 471 / 347 |
| Failures | INSUFFICIENT_FUTURE_BARS=883, PRICE_DATA_UNAVAILABLE=306 |
| attached | 2336 |

**PRE-FIX redundancy (live):** `already=600` vs `new=630` at +5m ⇒ **~49% of the
fetch budget re-processes already-matured observations** — Run 52's finding,
confirmed still active on main. **POST-FIX: no evidence** (never deployed).

Trend across pre-fix runs shows the backlog **growing**: ready 738 → **1042**,
deferred 338 → **642**, scanned 1870 → 2233 over ~90 min.

**Backlog status: `BACKLOG_GROWING`.** The single highest-value action is to
**promote `dev`→`main`** so the Run 52 anti-join and this run's direction fix take
effect, then re-measure with the telemetry already added.

## Part 5 — no-data / infinite-retry risk

Live evidence per run: `PRICE_DATA_UNAVAILABLE` 306–716, `INSUFFICIENT_FUTURE_BARS`
339–883. These never produce an outcome row, so they never leave the ready set and
are re-fetched **every** run — a permanent budget drain that the Run 52 anti-join
does **not** fix (it only removes *matured* work). Per-symbol retry count/age are
not currently persisted, so exact retry depth needs the terminal-state table below.

**Smallest safe policy (designed, recommended P1 — not implemented in this
verification run, to avoid stacking a second behavioral change on top of the
still-undeployed Run 52 fix):** classify a pending (observation_id, horizon)
deterministically and record a **terminal marker** so it stops re-queuing:

| class | rule (deterministic) | action |
|---|---|---|
| `TEMPORARY_PROVIDER_FAILURE` | `PROVIDER_ERROR` / alpaca timeout | retry (transient) |
| `NOT_YET_AVAILABLE` | `INSUFFICIENT_FUTURE_BARS` and anchor age < 1 trading day | retry later |
| `PERMANENT_NO_DATA` | `PRICE_DATA_UNAVAILABLE` and anchor age > 1 trading day | terminal |
| `TERMINAL_INVALID_OBSERVATION` | `INVALID_TIMESTAMP` | terminal |

Store terminal states in a new `hsf_maturation_state (observation_id, horizon,
state, attempts, last_reason, updated_at)` table — **separate from outcomes**, so
no outcome formula/definition changes; the maturation loader anti-joins it exactly
like outcomes. Deterministic (age-based), observable (counts in the report),
tested, and backward compatible. No aggressive dropping — observations are never
deleted, only marked non-eligible.

## Point-in-time integrity

`build_outcome`/`attach_outcome` enforce `evaluation_time > observation_timestamp`
(raises on violation); features come only from point-in-time scan rows; outcomes
live in a separate table. The auditor counts live `point_in_time_violations` per
cohort (tested with an injected violation) — expected 0, pending the live run.

## Legacy vs modern dataset

`MODERN_EXPLICIT_DATASET` = rows with an explicit `research_cohort` tag
(`explicitly_tagged`). `LEGACY_DATASET` = untagged rows `cohort_of()` would infer
as CANDIDATE (`legacy_inferred`). The auditor reports both per cohort and never
merges them; Run 54 must filter on the explicit tag. **Deterministic exclusion is
available today** (Part 6 gate #8 = PASS).

## Remaining data risks

1. **Direction fix undeployed** → all *existing* matured outcomes lack
   directional/MFE/MAE; only outcomes matured after deploy + re-maturation are
   fully usable. (raw_return + direction still allow recomputing directional_return.)
2. **Run 52 fix undeployed** → backlog growing, ~49% redundant fetching live.
3. **No-data infinite retry** (Part 5) → permanent capacity drain until the
   terminal-state policy ships.
4. **Live audit not yet run** → duplicates/invalid/PIT/overlap unconfirmed on real
   data.
5. **SHORT live evidence insufficient** → exclude SHORT from live claims for now.

## Run 54 readiness gate

`scripts/run54_readiness.py` (deterministic; reads `cohort_audit.json` when present):

| gate | status | why |
|---|---|---|
| 1 modern_cohort_tagging | CONDITIONAL | needs live audit |
| 2 cohort_separation | CONDITIONAL | needs live audit (overlap check ready) |
| 3 pit_integrity | CONDITIONAL | needs live audit |
| 4 outcome_formula | **PASS** | fixed + verified (Run 53A) |
| 5 long_direction | **PASS** | deterministic tests |
| 6 short_direction | CONDITIONAL | deterministic only; live INSUFFICIENT |
| 7 maturation_usable | CONDITIONAL | fixes undeployed; pre-fix outcomes NULL direction |
| 8 legacy_excludable | **PASS** | explicit tag + auditor split |
| 9 no_conflicting_duplicates | CONDITIONAL | needs live audit |
| 10 honest_sample_sizes | **PASS** | auditor reports all n |

**`RUN54_READY = CONDITIONAL`.** To reach YES: (a) promote `dev`→`main` (Run 52 +
Run 53A fixes), (b) let fresh maturation produce direction-complete outcomes,
(c) run the Research Integrity Audit against Neon and confirm
`pit_violations=0 / conflicting_duplicates=0 / candidate∩control=0`, (d) confirm
live SHORT samples or formally exclude SHORT from conclusions.

## Final verdict

**RESEARCH DATA CONDITIONALLY TRUSTED.** The pipeline is structurally sound, the
critical direction bug is fixed, and legacy isolation is deterministic — but full
trust is gated on deploying the fixes and running the live audit; existing
pre-fix outcomes are directionally incomplete.

## Test / lint

Full suite **1,394 passed** / 9 skipped (+8: long/short synonyms, LONG/SHORT
outcome e2e, cohort overlap, readiness gate); ruff `E9,F,I` clean.
