# DT Score — forward-outcome (predictive) validation

Status: **DT Score is a coherence/direction indicator, NOT a follow-through
predictor.** Confirmed on two independent, 100%-full-feature held-out windows
after parity certification ([DT_PARITY_CERTIFICATION.md](DT_PARITY_CERTIFICATION.md)).
No scoring was changed; this is a measurement of the *existing* production v1
score.

## Question
Does a higher DT Score rank better *direction-aware* forward outcomes (bearish
returns inverted)? Measured with `day_trade_validation.predictive_summary`:
Spearman(score, directional_return) per horizon, plus the top-minus-bottom score
tertile return spread and per-tertile hit rate.

## Results (production v1, full-feature, 0 fallback)

**2026-08-25 → 09-12** — 9,279 obs (8,128 directional); median DT 16.9, max 75.2.

| Horizon | n | Spearman | top−bottom return | hit top | hit bottom |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5m  | 7,997 | −0.0005 | +0.00002 | 0.483 | 0.468 |
| 15m | 7,780 | +0.0011 | −0.00000 | 0.480 | 0.482 |
| 30m | 7,438 | +0.0061 | +0.00013 | 0.476 | 0.473 |
| 60m | 6,758 | +0.0069 | +0.00012 | 0.480 | 0.473 |

**2026-05-04 → 05-22** — 11,083 obs (9,566 directional); median DT 25.2, max 92.1
(a more coherent/trending regime — higher scores, Strong n=562).

| Horizon | n | Spearman | top−bottom return | hit top | hit bottom |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5m  | 9,427 | +0.0105 | +0.00013 | 0.501 | 0.476 |
| 15m | 9,178 | +0.0171 | +0.00024 | 0.500 | 0.476 |
| 30m | 8,795 | +0.0220 | +0.00041 | 0.506 | 0.482 |
| 60m | 8,035 | +0.0342 | +0.00087 | 0.521 | 0.494 |

## Interpretation
- **August: no edge.** Spearman ≈ 0 at every horizon (|ρ| ≤ 0.007, under ~1
  standard error for n≈8k where SE(ρ) ≈ 1/√n ≈ 0.011); hit rates ~0.47–0.48 in
  both tertiles. A coin flip.
- **May: a tiny, horizon-growing edge.** Spearman rises to +0.034 at 60m (~3 SE,
  weakly distinguishable from zero) with a top-minus-bottom spread of **+0.087%**
  at 60m and top-tertile hit 0.521 vs bottom 0.494. Real but economically
  marginal — ρ ≈ 0.03 explains ~0.1% of return variance, and the spread is small
  relative to spreads/slippage.
- **Inconsistent across regimes.** The edge is essentially absent in August and
  only faintly present in the more trending May window. It is not a dependable,
  tradable signal.

## Verdict
DT Score does **not** reliably predict 5–60m direction-aware follow-through. This
reproduces every prior regime (up-week, down-week, and now two clean
full-feature windows). Keep DT Score framed and used as a **coherence / setup-
quality / direction** indicator — never as a probability of profit. Any
predictive scorer is a separate project (likely ML with richer features), out of
scope here.

Non-goals honored: no change to weights, ceilings, thresholds, gates, or the
score distribution; nothing tuned against forward returns.

## Recommended next step
If a predictive signal is desired, scope it as a **new, separately-validated
model** (distinct from DT Score), trained/tested with a train→validation→test
split and gated on full-feature coverage — do not retrofit DT Score. Otherwise,
DT Score is validated for its current role and needs no further calibration.
