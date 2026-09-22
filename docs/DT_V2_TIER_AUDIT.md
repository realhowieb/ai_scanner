# Rejected DT Score v2.0 tier audit

Status: **tested on `dev`; do not promote the old held-out result**. This is the Day Trader DT
Score experiment, despite the request's PreBreakout label. No PreBreakout model
or production scoring code was changed.

## Rollback and evidence

Commit `f661b04` reverted the v2 scoring change in `a6820a9`. Production
`analytics/day_trade_intel.py` uses v1 ceilings (ADX 40, RVOL 3, VWAP 1%,
momentum/gap 3%) and v1 quality thresholds (Strong 65/agreement 0.75,
Weak below 40). The v2 source remains in Git history. The previously reported
held-out aggregates were 11,378 total and 9,248 directional observations,
median 68, P75/P90/P95 77.1, and zero Strong or Weak. Two manual dev runs now
explain those results. The original v2 run artifact itself contained only
aggregates; the diagnostic reruns contain per-observation inputs and gates.

## Dev replay results

| Metric | Intraday fallback [35684934572](https://github.com/realhowieb/ai_scanner/actions/runs/35684934572) | Corrected daily lookup [35685104895](https://github.com/realhowieb/ai_scanner/actions/runs/35685104895) |
| --- | ---: | ---: |
| Total observations | 11,378 | 11,141 |
| Directional observations | 9,248 | 9,389 |
| Strong / Developing / Weak | 0 / 9,248 / 0 | 87 / 249 / 9,053 |
| Median raw score | 72.71 | 29.42 |
| Median final score | 72.7 | 16.2 |
| Score >=55 | 7,966 | 96 |
| Confirmation gate passed | 0 | 4,156 |
| Conflict gate passed | 9,248 | 1,745 |
| Final score 77.1 | 3,642 | 0 |

Both diagnostic runs have zero score/direction mismatches and no missing
diagnostic inputs. The corrected run has a different observation population,
so its tier rates must not be presented as a direct apples-to-apples improvement.

The decisive validation bug was `frames.get(symbol.upper()) or frames.get(symbol)`
in `scripts/validate_day_trade_score.py:_fetch_daily_frame`: evaluating a
pandas DataFrame as a boolean raises `ValueError`, which the surrounding broad
exception handler converted to `None`. Every symbol then fell back to
`minute_bars_to_observations`, which sets ADX, RVOL, gap, SuperTrend, and EWO
to missing. Commit `3e614fd` fixes the lookup and adds a regression test.

### Why the old tiers collapsed

In the fallback replay, every directional row has exactly two agreeing votes
(VWAP and intraday momentum), zero confirmations, and zero conflicts. Strong
requires confirmation, so it is impossible in that dataset: 7,966 pass every
other Strong gate but fail confirmation. Weak is also impossible there: no
directional row has score <35, agreement <0.55, RVOL <1, or >=3 conflicts.
Missing RVOL is *not* treated as low RVOL. Developing is the correct fallback
under those incomplete inputs.

### Exact 77.1 path

For 3,636 rows, the v2 component vector is exactly agreement `0.64`, VWAP
`1.0`, and momentum `1.0`; available weight is `0.35+0.10+0.10=0.55`.
The raw score is `(0.35*0.64+0.10+0.10)/0.55*100 = 77.090909...`.
There is no conflict penalty, so one-decimal rounding displays `77.1`.
Six more rows are close enough to round to `77.1`. All 3,642 have distinct
underlying input values; most collapse because VWAP and momentum are capped,
not because the provider duplicated identical bars. The corrected full-feature
run has zero rows at `77.1`.

### Full-feature blocker profile

The corrected run has 5,168 rows with five directional signals and 3,766 with
four. Strong's sequential funnel is 9,389 -> 96 pass score -> 96 pass
agreement -> 90 pass confirmation -> 87 pass conflicts. Of the top raw-score
quartile (2,348 rows), 2,252 fail the displayed-score gate; blocker overlap is
common. Weak conditions fire frequently: low score in 7,487 rows, low RVOL in
8,524, and >=3 conflicts in 4,892. Low participation is the most common
conflict (8,524 rows). Twelve rows satisfy a Weak condition but are classified
Strong because Strong is intentionally evaluated first. These are diagnostic
counts, not evidence to change thresholds.

## Responsible logic

- Rejected v2 agreement completeness and normalized sub-scores:
  `a6820a9:analytics/day_trade_intel.py:190-206`.
- Weighted raw score, conflict penalty, clamp, and rounding:
  `a6820a9:analytics/day_trade_intel.py:208-213`.
- Strong is evaluated before Weak, then Developing is the fallback:
  `a6820a9:analytics/day_trade_intel.py:216-236`.
- Production v1 score and classifier:
  `analytics/day_trade_intel.py:163-216`.

The v2 Strong rule requires **all** of score >=55, agreement >=0.70,
ADX >=20 or RVOL >=1.5, and at most one conflict. The v2 Weak rule requires
**any** of score <35, agreement <0.55, RVOL <1, or at least three conflicts.
Developing is reached only after both decisions fail. The rules are not
unreachable: focused fixtures produce Strong and Weak. For example, a synthetic
row with ADX 55, RVOL 5, VWAP +2%, intraday change -5%, gap -5%, green
SuperTrend, and positive EWO has raw score 72, final score 62, agreement 0.60,
two conflicts, and Developing. These fixtures explain possible mechanisms;
they are **not** held-out examples.

## What must be measured

The diagnostic JSON now contains per-observation raw/final score, vote counts,
confirmation flags/count, conflict flags/count, Strong and Weak gate results,
before/after conflict eligibility, and normalized score components. Aggregates
include top-quartile raw-score blocker combinations, Weak-rule reachability,
tier funnel, individual conflict frequency, representative observed examples,
and exact component/input fingerprints for score 77.1. The Markdown report
renders these tables and the responsible source references.

The two dev reruns identify the dominant cause of the **old** 9,248-row result:
the intraday fallback supplied no daily confirmation or conflict inputs. The
corrected run demonstrates that the full-feature classifier does separate
tiers, but changes the dataset count. Before any v2.1 decision, validate daily
bar coverage and the reconstruction's alignment to live scanner inputs across
the held-out window. Do not tune tier constants against either diagnostic run.

## Conditional v2.1 research order

1. **Verify full daily-indicator coverage and live/validation feature parity**
   (highest expected impact on validity, low regression risk). Record when
   historical fallback occurs rather than silently treating partial rows as
   full fidelity.
2. **Reassess score and Weak-tier behavior on a stable, full-feature population**
   (high diagnostic value, no production risk). The corrected run is mostly
   Weak, a different question from the original all-Developing claim.
3. **Only then consider candidate score or tier changes with new held-out
   validation** (unknown performance impact, high regression risk). Current
   evidence does not justify a conflict or confirmation threshold adjustment.

None of these changes is proposed for production in this audit.
