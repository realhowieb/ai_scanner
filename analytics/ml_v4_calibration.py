"""ML v4 probability calibration: sigmoid (Platt) and isotonic, chosen chronologically.

Rules (pre-registered in the ML v4 plan, Phase 3):
  * a calibrator is fitted ONLY on out-of-fold predictions from EARLIER
    walk-forward folds, never on the fold it is applied to and never on the
    final holdout;
  * the method is chosen on held-out chronological data: both methods are
    fitted on all earlier folds but the latest one, scored (Brier) on that
    latest earlier fold, and the winner is refitted on every earlier fold.
    With only one earlier fold there is nothing held out to choose on, so the
    size rule from the ML v3 report applies (isotonic from 200 rows, else
    sigmoid) and the result says so;
  * raw model outputs are kept; calibrated values are a separate column.

Calibrators serialize to plain JSON. The isotonic form is the same x/y step map
AI Confidence already uses (``np.interp``), so serving needs only numpy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MIN_FIT_ROWS = 50
MIN_FIT_CLASS = 5
ISOTONIC_MIN_ROWS = 200  # ML v3 report #33: Platt below ~200 rows
_EPS = 1e-6


def _clean(y, p) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same length")
    return y, p


def fittable(y) -> bool:
    y = np.asarray(y, dtype=int)
    return len(y) >= MIN_FIT_ROWS and min(int(y.sum()), int(len(y) - y.sum())) >= MIN_FIT_CLASS


def fit_sigmoid(y, p) -> Optional[Dict[str, Any]]:
    """Platt scaling on the logit of the raw probability."""
    from sklearn.linear_model import LogisticRegression

    y, p = _clean(y, p)
    if not fittable(y):
        return None
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"method": "sigmoid", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]), "n": int(len(y)),
            "positive": int(y.sum())}


def fit_isotonic(y, p) -> Optional[Dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression

    y, p = _clean(y, p)
    if not fittable(y):
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    xs, ys = [float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_]
    if len(xs) < 2:  # a single step can't be interpolated; treat as unusable
        return None
    return {"method": "isotonic", "x": xs, "y": ys, "n": int(len(y)), "positive": int(y.sum())}


FITTERS = {"sigmoid": fit_sigmoid, "isotonic": fit_isotonic}


def apply(cal: Optional[Mapping[str, Any]], p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    if not cal:
        raise ValueError("no calibrator")
    if cal["method"] == "sigmoid":
        z = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))
    if cal["method"] == "isotonic":
        return np.clip(np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"])), 0.0, 1.0)
    raise ValueError(f"unknown calibration method {cal['method']!r}")


def brier(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def choose_and_fit(chunks: Sequence[Tuple[Sequence[int], Sequence[float]]]) -> Optional[Dict[str, Any]]:
    """Calibrator for the NEXT fold from ``chunks`` = [(y, raw_p) per earlier fold, oldest first]."""
    if not chunks:
        return None
    y_all = np.concatenate([np.asarray(c[0], dtype=int) for c in chunks])
    p_all = np.concatenate([np.asarray(c[1], dtype=float) for c in chunks])
    if not fittable(y_all):
        return None
    selection: Dict[str, Any]
    if len(chunks) >= 2:
        y_fit = np.concatenate([np.asarray(c[0], dtype=int) for c in chunks[:-1]])
        p_fit = np.concatenate([np.asarray(c[1], dtype=float) for c in chunks[:-1]])
        y_sel, p_sel = np.asarray(chunks[-1][0], dtype=int), np.asarray(chunks[-1][1], dtype=float)
        scores = {}
        for name, fitter in FITTERS.items():
            cal = fitter(y_fit, p_fit)
            if cal is not None and len(y_sel):
                scores[name] = round(brier(y_sel, apply(cal, p_sel)), 6)
        if scores:
            method = min(sorted(scores), key=lambda k: scores[k])
            selection = {"rule": "brier on the latest earlier fold", "brier": scores,
                         "selection_rows": int(len(y_sel))}
        else:
            method = "isotonic" if len(y_all) >= ISOTONIC_MIN_ROWS else "sigmoid"
            selection = {"rule": f"size rule (isotonic from {ISOTONIC_MIN_ROWS} rows); selection fit too small"}
    else:
        method = "isotonic" if len(y_all) >= ISOTONIC_MIN_ROWS else "sigmoid"
        selection = {"rule": f"size rule (isotonic from {ISOTONIC_MIN_ROWS} rows); one earlier fold, "
                             "nothing held out to choose on"}
    cal = FITTERS[method](y_all, p_all)
    if cal is None and method == "isotonic":
        method, cal = "sigmoid", fit_sigmoid(y_all, p_all)
    if cal is None:
        return None
    cal["selection"] = selection
    cal["fitted_on_folds"] = len(chunks)
    return cal


def reliability(y, p, bins: int = 10) -> List[Dict[str, Any]]:
    """Reliability table (fixed-width bins) with the observed hit rate per bin."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append({"bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": n,
                    "mean_predicted": round(float(p[m].mean()), 4) if n else None,
                    "hit_rate": round(float(y[m].mean()), 4) if n else None})
    return out


def ece(y, p, bins: int = 10) -> Optional[float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    total = 0.0
    for row in reliability(y, p, bins):
        if row["n"]:
            total += row["n"] / len(y) * abs(row["hit_rate"] - row["mean_predicted"])
    return round(total, 4) if math.isfinite(total) else None
