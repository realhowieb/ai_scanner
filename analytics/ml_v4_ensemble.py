"""ML v4 ensembles: combine provider probabilities without peeking at the fold scored.

Methods (Phase 4 of the ML v4 plan):
  * ``equal``     : plain average of the members (the benchmark; nothing fitted);
  * ``weighted``  : non-negative weights summing to 1 that minimise log loss on
                    EARLIER folds' out-of-fold predictions;
  * ``best``      : the single member with the lowest log loss on earlier folds;
  * ``stacking``  : logistic regression on the members' logits, fitted on earlier
                    folds' out-of-fold predictions only (those inputs are
                    themselves out of sample, which is what makes it leakage-safe).

Every fitted method needs enough earlier rows; until then it falls back to the
equal average and says so. Weights are never fitted on the final holdout.

Per row we also report model disagreement (spread of member probabilities).
Members are trained on the same data with similar features, so agreement is
NOT independent confirmation of a prediction.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

METHODS = ("equal", "weighted", "best", "stacking")
MIN_FIT_ROWS = {"weighted": 100, "best": 100, "stacking": 200}
_EPS = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _matrix(members: Sequence[str], probs: Mapping[str, Sequence[float]]) -> np.ndarray:
    lens = {len(probs[m]) for m in members}
    if len(lens) != 1:
        raise ValueError("member predictions have different lengths")
    return np.column_stack([_clip(probs[m]) for m in members])


def fit(method: str, members: Sequence[str], y_earlier, probs_earlier: Mapping[str, Sequence[float]]
        ) -> Dict[str, Any]:
    """Fit ``method`` on earlier out-of-fold predictions. Returns a JSON-safe spec."""
    if method not in METHODS:
        raise ValueError(f"unknown ensemble method {method!r}; expected one of {METHODS}")
    members = list(members)
    if len(members) < 2:
        raise ValueError("an ensemble needs at least 2 members")
    equal = {"method": "equal", "members": members, "weights": [1.0 / len(members)] * len(members)}
    if method == "equal":
        return {**equal, "fitted_rows": 0}
    y = np.asarray(y_earlier, dtype=int)
    if len(y) < MIN_FIT_ROWS[method] or not (0 < y.sum() < len(y)):
        return {**equal, "requested": method, "fitted_rows": int(len(y)),
                "fallback": f"needs >= {MIN_FIT_ROWS[method]} earlier rows with both classes; used equal weights"}
    X = _matrix(members, probs_earlier)
    if method == "best":
        losses = [log_loss(y, X[:, j]) for j in range(len(members))]
        j = int(np.argmin(losses))
        w = [0.0] * len(members)
        w[j] = 1.0
        return {"method": "best", "members": members, "weights": w, "selected": members[j],
                "earlier_log_loss": {m: round(v, 6) for m, v in zip(members, losses)}, "fitted_rows": int(len(y))}
    if method == "weighted":
        from scipy.optimize import minimize

        k = len(members)
        res = minimize(lambda w: log_loss(y, X @ w), np.full(k, 1.0 / k), method="SLSQP",
                       bounds=[(0.0, 1.0)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
        w = np.clip(res.x if res.success else np.full(k, 1.0 / k), 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
        return {"method": "weighted", "members": members, "weights": [round(float(v), 6) for v in w],
                "optimizer_success": bool(res.success), "fitted_rows": int(len(y))}
    from sklearn.linear_model import LogisticRegression

    Z = np.log(X / (1 - X))
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    return {"method": "stacking", "members": members, "coef": [float(c) for c in lr.coef_[0]],
            "intercept": float(lr.intercept_[0]), "fitted_rows": int(len(y)),
            "inputs": "member logits (out-of-fold predictions from earlier folds)"}


def predict(spec: Mapping[str, Any], probs: Mapping[str, Sequence[float]]) -> np.ndarray:
    X = _matrix(spec["members"], probs)
    if spec["method"] == "stacking":
        Z = np.log(X / (1 - X))
        return 1.0 / (1.0 + np.exp(-(Z @ np.asarray(spec["coef"]) + spec["intercept"])))
    return np.clip(X @ np.asarray(spec["weights"], dtype=float), 0.0, 1.0)


def disagreement(members: Sequence[str], probs: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """Per-row spread of member probabilities, summarised."""
    X = _matrix(members, probs)
    if len(X) == 0:
        return {"rows": 0}
    sd, rng = X.std(axis=1), X.max(axis=1) - X.min(axis=1)
    return {"rows": int(len(X)), "mean_std": round(float(sd.mean()), 4), "max_std": round(float(sd.max()), 4),
            "mean_range": round(float(rng.mean()), 4), "share_range_over_0_10": round(float((rng > 0.10).mean()), 4),
            "note": "members share training data and features; agreement is not independent confirmation"}


def row_record(spec: Mapping[str, Any], member_probs: Mapping[str, float], ensemble_prob: float, *,
               target: Mapping[str, Any], model_versions: Mapping[str, str], calibration_version: Optional[str],
               feature_version: Optional[str], inferred_at: str) -> Dict[str, Any]:
    """What one ensemble prediction captures (shadow mode stores this per observation)."""
    vals = [float(member_probs[m]) for m in spec["members"]]
    return {
        "member_probabilities": {m: round(float(member_probs[m]), 6) for m in spec["members"]},
        "ensemble_probability": round(float(ensemble_prob), 6),
        "disagreement_std": round(float(np.std(vals)), 6), "disagreement_range": round(max(vals) - min(vals), 6),
        "method": spec["method"], "target": dict(target), "model_versions": dict(model_versions),
        "calibration_version": calibration_version, "feature_version": feature_version, "inferred_at": inferred_at,
    }


def is_finite_spec(spec: Mapping[str, Any]) -> bool:
    vals = list(spec.get("weights") or []) + list(spec.get("coef") or []) + [spec.get("intercept", 0.0)]
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
