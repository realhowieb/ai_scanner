# Rejected DT Score v2.0 tier audit

Status: **awaiting original held-out observations**. This is the Day Trader DT
Score experiment, despite the request's PreBreakout label. No PreBreakout model
or production scoring code was changed.

## Rollback and evidence

Commit `f661b04` reverted the v2 scoring change in `a6820a9`. Production
`analytics/day_trade_intel.py` uses v1 ceilings (ADX 40, RVOL 3, VWAP 1%,
momentum/gap 3%) and v1 quality thresholds (Strong 65/agreement 0.75,
Weak below 40). The v2 source remains in Git history. The previously reported
held-out aggregates are 11,378 total and 9,248 directional observations,
median 68, P75/P90/P95 77.1, and zero Strong or Weak. They are user-supplied
aggregates, not locally reproduced measurements. The checked-in validation
artifact reports `NO_INTRADAY_DATA`; it cannot reconstruct indicator rows.

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

No evidence currently identifies the dominant held-out blocker, why Weak is
absent, or the exact cause of 77.1. Those conclusions require the original
per-observation indicators or a fresh held-out replay with the same universe,
feed, dates, and sampling interval. A rerun with a different directional count
must be labeled a new dataset.

## Conditional v2.1 research order

1. **Measure shared 77.1 component fingerprints** (high diagnostic value, no
   regression risk). This distinguishes repeated inputs from normalization,
   clipping, penalty, and rounding effects.
2. **Inspect top-quartile blocker combinations and Weak gates** (high diagnostic
   value, no regression risk). A single threshold cannot fix multiple failed
   gates; rule precedence matters only where Strong and Weak overlap.
3. **Consider a conflict-gate change only if it uniquely blocks coherent
   high-score rows** (potentially high impact, high regression risk).
4. **Consider confirmation, agreement, or score-threshold changes only if the
   measured blocker counts and follow-through support them** (unknown impact,
   high regression risk).

None of these changes is proposed for production in this audit.
