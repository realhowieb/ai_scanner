# ML v3 market context readiness and regime performance

## Readiness (historically safe availability at observation time)

| CONTEXT | STATUS | EVIDENCE |
|---|---|---|
| SPY trend | SAFE_TO_DERIVE | Not stored on observations. Derivable from SPY daily closes strictly before the entry day (used below for regime labels only). |
| QQQ trend | SAFE_TO_DERIVE | Same as SPY; not stored. |
| market volatility | SAFE_TO_DERIVE | SPY realized volatility from prior closes; no VIX stored. |
| breadth | INSUFFICIENT_HISTORY | Only derivable from the same scan's candidate rows; never frozen (REGIME_CAPTURE_UNAVAILABLE). |
| sector performance | UNSAFE | Only today's sector map exists. |
| sector relative strength | UNSAFE | Needs the historical sector map. |
| stock-vs-SPY strength | AVAILABLE | rs_vs_spy stored by Run 57+ scans; null on 87.7% of observations. |
| stock-vs-QQQ strength | MISSING | Not computed or stored. |
| stock-vs-sector strength | UNSAFE | Needs the historical sector map. |

## Regime performance (5-day, out-of-sample predictions)

Rule: SPY 20-session return and annualized volatility from closes strictly before the entry day; bullish > +2%, bearish < -2%, high volatility >= 20%.

Entry days per regime (trend/volatility): {"sideways/low": 12, "bearish/low": 1}

### HSF Score (production heuristic)

| DIMENSION | REGIME | N | ROC-AUC [95% CI] | WIN RATE | MEDIAN RETURN |
|---|---|---|---|---|---|
| trend | bearish | 0 | n/a | n/a | n/a |
| trend | sideways | 25 | 0.507 [0.071, 0.583] | 60.0% | 1.18% |
| volatility | low | 25 | 0.507 [0.071, 0.583] | 60.0% | 1.18% |

### Logistic regression (research)

| DIMENSION | REGIME | N | ROC-AUC [95% CI] | WIN RATE | MEDIAN RETURN |
|---|---|---|---|---|---|
| trend | bearish | 0 | n/a | n/a | n/a |
| trend | sideways | 25 | 0.493 [0.000, 0.583] | 60.0% | 1.18% |
| volatility | low | 25 | 0.493 [0.000, 0.583] | 60.0% | 1.18% |

### XGBoost, leakage-safe subset (research)

| DIMENSION | REGIME | N | ROC-AUC [95% CI] | WIN RATE | MEDIAN RETURN |
|---|---|---|---|---|---|
| trend | bearish | 0 | n/a | n/a | n/a |
| trend | sideways | 25 | 0.500 [0.500, 0.500] | 60.0% | 1.18% |
| volatility | low | 25 | 0.500 [0.500, 0.500] | 60.0% | 1.18% |

### PreBreakout % as served (production XGBoost)

| DIMENSION | REGIME | N | ROC-AUC [95% CI] | WIN RATE | MEDIAN RETURN |
|---|---|---|---|---|---|
| trend | bearish | 0 | n/a | n/a | n/a |
| trend | sideways | 0 | n/a | n/a | n/a |
| volatility | low | 0 | n/a | n/a | n/a |

Regimes with N < 20 are not evaluable; none is called statistically meaningful.

