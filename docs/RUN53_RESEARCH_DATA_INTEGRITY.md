# Run 53 — live research dataset integrity audit

Are the Run 47 research cohorts (CANDIDATE / NEAR_MISS / CONTROL) being captured
correctly on real production scans, and are they fit for future effectiveness
analysis? Investigation only — **the scanner was not tuned and scoring was not
modified.** A read-only auditor (`scripts/audit_research_cohorts.py`) was added to
make the per-cohort integrity table reproducible.

## Verdict: **PARTIALLY READY**

The capture machinery is correct, point-in-time-safe, deterministic, and running
on every scheduled production scan, with cohort proportions matching the Run 47
design. It is **not yet certified RESEARCH READY** for two reasons: (1) the live
per-row integrity table (duplicates / invalid values / PIT violations across the
whole table) requires running the new auditor against the production Neon DB — it
could not be executed from this environment (no DB credentials; CI log counts are
secret-masked); and (2) two known design gaps affect analysis: legacy pre-cohort
observations are inferred as CANDIDATE unless callers filter on the explicit tag,
and outcome maturation collapses direction to the primary (long) scanner.

## What was verified (code + tests, authoritative)

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Candidates are the actually-selected candidates | **PASS** | `scan/engine.py:775` `df = df.head(top_n)`; the same `top_n` rows are captured with `research_cohort="CANDIDATE"`, `selection_reason="top_n_candidate"` (`cron_runner.py:508`). |
| 2 | Near-misses are genuinely below the boundary | **PASS** | `scan/engine.py:760` `near_miss = df.iloc[top_n:top_n+nm_n]` — the ranked rows immediately below the cut, same scorer/ranking; production output is unchanged (extra rows never enter results/snapshots). |
| 3 | Controls deterministically sampled, no outcome knowledge | **PASS** | `select_control_symbols` ranks by `sha256(scan_run_id|symbol)`, excludes candidates, sampled from `evaluated_symbols` (all priced symbols). No returns/outcomes involved; reproducible across machines. |
| 4 | No outcome/future info enters selection features | **PASS** | Features come only from point-in-time scan result rows; `build_observation` reads no outcomes. Outcomes live in a separate table, attached after the fact. |
| 5 | Captured on scheduled production scans | **PASS** | Live `[research_cohorts] US_MARKET: candidates=…00 near_miss=50 control=…00 written=… (evaluated=…487)` on scheduled runs 35912400576, 35903364523 (2026-09-23). |
| 6 | Proportions ≈ Run 47 design | **PASS** | `near_miss=50` exact (`DEFAULT_NEAR_MISS_N`), control ≈100 (`DEFAULT_CONTROL_N`), candidates ≈ top_n, over ~11.5K evaluated — matches design; `HARD_CAP=500` bounds all three. |
| 7 | Outcomes stay separate from PIT features | **PASS** | Separate `hsf_observation_outcomes` table keyed `(observation_id, horizon)`; `attach_outcome` returns a copy and never mutates features. |
| 8 | LONG/SHORT direction handling | **PARTIAL** | Direction is captured per scanner (`gap_down → short`, others `long`). But maturation uses `scanners[0].direction`, which is always the long `breakout` primary — short sub-signals are matured as long. Fine for the long breakout thesis; a gap here for short-specific outcome study. |
| 9 | Legacy cannot silently contaminate | **PARTIAL** | `cohort_of()` infers untagged rows as CANDIDATE (`LEGACY_CANDIDATE`). Post-Run-47 rows carry an explicit `research_cohort`; pre-cohort rows do not. Not silent **only if** analysis filters on the explicit tag — the new auditor reports `explicitly_tagged` vs `legacy_inferred` per cohort so contamination is visible and separable. |
| 10 | Capture failures can't break scanning | **PASS** | Both capture blocks are `try/except`, non-fatal, side-effect-only, with kill switch `HSF_OBSERVATION_CAPTURE=0` and dry-run `HSF_OBSERVATION_CAPTURE_DRYRUN=1`; a failure prints and the scan continues with identical results. Confirmed by the `except Exception` guards at `cron_runner.py:522` and `:81`. |

Idempotency underpinning the dataset: `observation_id = sha256(symbol|timestamp|context)[:16]` with hour-bucketed timestamps, so workflow retries in the same scheduled slot collapse to one row; `save_observations_batch` is `ON CONFLICT DO NOTHING` (first-write-wins). Observations are immutable by design — a scoring change bumps `schema_version` and writes a new row rather than overwriting.

## Live per-cohort integrity table — requires an auditor run against prod

The requested metrics (observations, distinct symbols, distinct runs, obs/run,
first/latest timestamp, matured/unmatured, missing fields, duplicates, conflicting
duplicates, invalid values, point-in-time violations) are computed by
`scripts/audit_research_cohorts.py`. It is **read-only** (never writes
observations/outcomes/scans, never touches scoring). It could not be run here: the
production store is Neon and no credentials are present in this environment, and
GitHub masks the numeric counts in scan logs (`candidates=***00`). Run it where the
DB is reachable to populate the table:

```bash
python -m scripts.audit_research_cohorts   # writes artifacts/automation/cohort_audit.json
```

It emits, per cohort: `observations, distinct_symbols, distinct_runs,
observations_per_run, first_observation, latest_observation, matured, unmatured,
missing_field_observations, invalid_values, point_in_time_violations,
explicitly_tagged, legacy_inferred`, plus table-wide `duplicate_observation_ids`
and `conflicting_duplicates`. The auditor is unit-tested (per-cohort counts,
legacy separation, matured/unmatured, injected PIT violation, invalid-value and
conflicting-duplicate detection).

### What live evidence is available now (from unmasked log structure)

- All three cohorts are present on scheduled scans (Task 5). ✅
- `near_miss = 50` exactly; `control`/`candidate` in the ~100 range; `evaluated ≈
  11,485–11,487` — proportions match design (Task 6). ✅
- `written=…` is non-zero each run → cohorts persist, not just labelled. ✅

## Risks / recommendations (no scoring change)

- **P1 — Run the auditor against prod** and gate on it: certify RESEARCH READY only
  when `conflicting_duplicates=0`, `invalid_values=0`, `point_in_time_violations=0`,
  and `legacy_inferred` is separated out of any effectiveness comparison.
- **P1 — Legacy isolation (Task 9):** effectiveness analysis should filter on the
  explicit `research_cohort` tag (or `schema_version` + first-cohort-scan date), not
  `cohort_of()`'s inferred label, so pre-Run-47 candidates never mix into cohort
  comparisons. The auditor's `explicitly_tagged`/`legacy_inferred` split supports
  this; consider adding a persisted `cohort_epoch`/capture-version marker.
- **P2 — Direction (Task 8):** if short setups become part of the study, mature
  outcomes per scanner-direction rather than `scanners[0]`, so `gap_down` shorts get
  correctly-signed returns. No change needed for the current long breakout thesis.
- **P2 — Control feature sparsity is by design:** controls carry only identity +
  price/volume (they were filtered before technical scoring), so
  `missing_field_observations` will be high for CONTROL. Analysis must compare
  cohorts only on features all cohorts share, or treat controls as a
  base-rate denominator — this is correct, not a defect.

## Test / lint

Full suite **1,386 passed** / 9 skipped (4 new cohort-audit tests); ruff `E9,F,I`
clean.
