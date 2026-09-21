# DT Score Recalibration Plan

Status: **planning** (no scoring changed). Grounded in the Run 33 validation
(`.github/workflows/validate-dt-score.yml` → `scripts/validate_day_trade_score.py`).

## 0. Evidence baseline
Wider validation run (30 symbols, ~7 weeks, **24,324 observations / 19,476
directional**):
- **Ceiling saturation** — median DT = **100**; P25=84.6, P50=P75=P90=100;
  ~67% of scores in 90–100.
- **No follow-through ranking** — direction-aware 15m hit rate is flat-to-inverted
  (60–69 → 59%, 70–79 → 51%, 80–89 → 49%, 90–100 → **48%**). Higher score ≠ better.
- **Dead quality tiers** — 100% of directional observations are "developing";
  Strong/Weak never fire.

Caveat: the no-lookahead reconstruction carries daily indicators as-of D-1, which
tend to agree in trending large-caps — partly inflating agreement. This is the
honest best proxy, but a note for interpretation and a reason to sample broadly.

### Cross-regime confirmation (2026-06-01 → 2026-07-10, 20 symbols incl. SPY)
A genuine **down/bearish** regime (14,401 obs / 11,366 directional; bearish 7,890
vs bullish 3,476, ≈69% bearish) reproduces the same structural findings:
- **Saturation is regime-independent** — median DT = **100** again (P90 = 100).
- **No follow-through edge** — 15m hit by bucket: 60–69 44%, 70–79 46%, 80–89 51%,
  90–100 **49%**; flat ~48–51% at every high bucket, coin-flip at the top. Bearish
  setups specifically also ~50% at 90–100. Higher DT ≠ better follow-through in a
  down market either.
- **Direction/strength separation VALIDATES** — the score correctly produced
  predominantly bearish high-scores (6,999 bearish vs 2,794 bullish in 90–100),
  the mirror of the up-week's bullish skew. DT Score tracks direction/regime
  faithfully; it just does not predict 15–60m follow-through and does not spread.

Takeaway: saturation (→C1/C2) and dead tiers (→C4) are structural, not artifacts
of one bullish week; DT Score is validated as a coherence/direction indicator,
not a predictor.

## 1. Objectives / non-goals
**Goals:** (a) spread the score across 0–100 so it discriminates; (b) make
Strong/Developing/Weak separate; (c) keep direction/strength separation and the
"coherence, not prediction" framing.
**Non-goals:** no ML, no new indicators, no chasing hit rate. **If follow-through
stays flat after recalibration, DT Score remains a coherence indicator and is
never presented as predictive** — spread + honest tiers is the win.

## 2. Root-cause hypotheses (why it saturates)
1. **Agreement maxes too easily** — `(agreement−0.5)/0.5` = 1.0 for any unanimous
   set; trending liquid names agree on all 4–5 signals → full 35%.
2. **Normalization ceilings too low** — ADX 40 / RVOL 3× / VWAP 1% / MOM 3% /
   GAP 3%. Liquid movers clear these routinely → every sub-score ≈ 1 → total ≈ 100.
3. **Evidence completeness ignored** — 3 agreeing signals score like 6.
4. **Conflict penalty rarely triggers** in clean trends.

## 3. Candidate changes (each gated on evidence; OLD → NEW → WHY)
| # | Change | OLD | NEW (candidate) | Why |
|---|---|---|---|---|
| C1 | Agreement stricter + count-weighted | `(agr−0.5)/0.5` | scale by signal count so 5/5 > 3/3; > bare-majority for high credit | stop unanimity auto-maxing |
| C2 | Raise normalization ceilings | ADX 40 / RVOL 3× / VWAP 1% / MOM 3% / GAP 3% | ADX ~55 / RVOL ~5× / VWAP ~2% / MOM ~5% | spread mid-range; typical values shouldn't hit 1.0 |
| C3 | Evidence-completeness factor | none | mild multiplier for # available directional signals | 3-signal setups ≠ 6-signal setups |
| C4 | Re-fit quality tiers to distribution | Strong ≥65 / Weak <40 | percentile-based on comparable setups | fixed cutoffs can't separate a saturated distribution |
| C5 | Confirmation gate on top scores | none | cap agreement credit unless ADX/RVOL genuinely strong | high score should require real participation |

Pick the **smallest subset** that restores spread — likely **C2 + C1 first**;
add C3/C4/C5 only if needed. Prefer simple, explainable constants.

## 4. Validation gate (nothing ships without this)
- **Regimes:** ≥3 windows incl. down + choppy weeks. **Symbols:** ~40 across
  large/small-cap and high/low vol (not just liquid trenders).
- **Split:** tune on one set, confirm on a held-out set. Never fit the whole history.
- **Acceptance criteria:**
  1. **Spread** — median DT ~55–75, P90 < 100, no percentiles pinned at 100.
  2. **Tiers separate** — Strong/Developing/Weak each get a real share; Strong is
     the best (or at least non-worst) on follow-through + MFE/MAE.
  3. **No inversion** — hit-rate-by-bucket monotone non-decreasing, OR explicitly
     accepted as flat → then coherence-only, and we say so.
  4. **Stability** — holds on the held-out regime.
- Every candidate carries **OLD / NEW / WHY / evidence**, run through the
  `validate-dt-score` workflow.

## 5. Rollout & guardrails
- Version the change (`DT_SCORE_VERSION`) so history stays interpretable and
  telemetry can compare v1 vs v2.
- Preserve direction/strength separation (bearish can still score high).
- Re-run all Run 32/33 tests; add tests locking the new spread + tier behavior.
- No change to Scanner / PreBreakout / Market Brief / ML.

## 6. Decision tree after validation
- Spread + tiers fixed, follow-through improves → ship v2 as a ranked coherence score.
- Spread + tiers fixed, follow-through still flat → ship v2 for discrimination,
  keep strictly "coherence, not prediction" wording.
- Can't fix spread with simple changes → stop; DT Score stays as-is with honest
  coherence framing; a predictive score becomes a separate (likely ML) project.

## 7. Data collection
`validate-dt-score` runs weekly (scheduled) over a rolling window to accumulate
multi-regime samples; artifacts (`day_trader_validation.{json,md}`) are kept per
run for the step-4 gate. Manual dispatch remains for targeted windows.
