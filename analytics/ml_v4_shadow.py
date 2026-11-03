"""ML v4 shadow mode: score new observations with registry models, summarise, gate promotion.

Shadow scoring runs in a scheduled GitHub workflow (``scripts/ml_v4_shadow.py``)
and writes only ``ml_shadow_predictions``. Nothing here changes AI Confidence,
PreBreakout, alerts or anything customers see.

This module is numpy-only so the API can summarise and evaluate gates without
the training libraries. Scoring itself takes already-loaded model objects.

Promotion gates are pre-registered here (``GATES``) and evaluated on FORWARD
shadow predictions only (observations after the model entered shadow), compared
with AI Confidence as served on the very same matured rows.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from analytics import ml_v4_calibration as C

GATES_VERSION = "hsf-ml-v4-gates-1"
GATES: Dict[str, Any] = {
    "min_labeled_rows": 200,        # matured forward shadow predictions
    "min_entry_days": 20,           # distinct trading days among them
    "min_auc_delta_vs_served": 0.0,  # shadow AUC - AI Confidence AUC on the same rows
    "max_brier_vs_served": 0.0,      # shadow Brier - AI Confidence Brier (lower is better)
    "max_ece": 0.10,                 # calibration error of the probability it would serve
    "min_served_overlap": 100,       # rows with AI Confidence as served, for the comparison
}


def auc(y, p) -> Optional[float]:
    """ROC-AUC via average ranks (ties count half); numpy only."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if len(y) == 0 or n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _r(v, d=4):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), d)


# ----------------------------------------------------------------------------- scoring
def served_probability(model: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    """The probability a model would serve: its registered calibrator if any, else raw."""
    cal = model.get("calibration")
    if isinstance(cal, Mapping) and cal.get("method") in ("sigmoid", "isotonic"):
        return C.apply(cal, raw)
    return np.asarray(raw, dtype=float)


def score(model: Mapping[str, Any], estimator: Any, rows: Sequence[Any], *, scored_at: str) -> List[Dict[str, Any]]:
    """One prediction per audit row for one registry model (``estimator`` has ``predict``)."""
    if not rows:
        return []
    raw = np.asarray(estimator.predict(list(rows)), dtype=float)
    cal = served_probability(model, raw)
    has_cal = isinstance(model.get("calibration"), Mapping) and bool(model["calibration"].get("method"))
    out = []
    for r, a, b in zip(rows, raw, cal):
        out.append({
            "model_id": int(model["id"]), "model_version": model["model_version"],
            "observation_id": int(r.observation_id), "ticker": r.ticker, "entry_day": r.entry_day,
            "observed_at": r.observed_at, "target_label": model["target_label"],
            "horizon_days": int(model["horizon_days"]), "raw_probability": _r(a, 6),
            "calibrated_probability": _r(b, 6) if has_cal else None,
            "feature_schema_version": model.get("feature_schema_version"), "scored_at": scored_at,
        })
    return out


def forward_rows(rows: Sequence[Any], shadow_start: Optional[dt.datetime]) -> List[Any]:
    """Only observations made after the model entered shadow (true forward test)."""
    if shadow_start is None:
        return []
    return [r for r in rows if r.observed_at >= shadow_start]


# ----------------------------------------------------------------------------- summaries
def _prob(p: Mapping[str, Any]) -> Optional[float]:
    v = p.get("calibrated_probability")
    if v is None:
        v = p.get("raw_probability")
    return None if v is None else float(v)


def summarize_model(preds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Forward shadow metrics for one model, plus AI Confidence as served on the same matured rows."""
    preds = list(preds)
    labeled = [p for p in preds if p.get("label") is not None and _prob(p) is not None]
    out: Dict[str, Any] = {
        "scored": len(preds), "labeled": len(labeled),
        "entry_days": len({str(p["entry_day"]) for p in labeled}),
        "first_scored_at": min((str(p["scored_at"]) for p in preds), default=None),
        "last_scored_at": max((str(p["scored_at"]) for p in preds), default=None),
        "mean_probability": _r(np.mean([_prob(p) for p in preds if _prob(p) is not None])) if preds else None,
    }
    if not labeled:
        out["note"] = "no matured shadow predictions yet"
        return out
    y = np.array([int(p["label"]) for p in labeled])
    pr = np.array([_prob(p) for p in labeled])
    out.update({"hit_rate": _r(y.mean()), "roc_auc": _r(auc(y, pr)), "brier": _r(C.brier(y, pr)),
                "ece": C.ece(y, pr), "reliability": C.reliability(y, pr)})
    both = [p for p in labeled if p.get("ai_confidence") is not None]
    if both:
        yb = np.array([int(p["label"]) for p in both])
        pb = np.array([_prob(p) for p in both])
        sv = np.clip(np.array([float(p["ai_confidence"]) / 100.0 for p in both]), 0.0, 1.0)
        a_m, a_s = auc(yb, pb), auc(yb, sv)
        out["vs_ai_confidence_as_served"] = {
            "rows": len(both), "model_auc": _r(a_m), "served_auc": _r(a_s),
            "auc_delta": _r(a_m - a_s) if a_m is not None and a_s is not None else None,
            "model_brier": _r(C.brier(yb, pb)), "served_brier": _r(C.brier(yb, sv)),
            "brier_delta": _r(C.brier(yb, pb) - C.brier(yb, sv)),
        }
    return out


def disagreement(preds_by_model: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Spread of probabilities across models on the observations they all scored."""
    by_obs: Dict[int, Dict[int, float]] = defaultdict(dict)
    for mid, preds in preds_by_model.items():
        for p in preds:
            v = _prob(p)
            if v is not None:
                by_obs[int(p["observation_id"])][int(mid)] = v
    k = len(preds_by_model)
    shared = [list(v.values()) for v in by_obs.values() if len(v) == k]
    if k < 2 or not shared:
        return {"models": k, "rows": 0}
    X = np.array(shared)
    sd, rng = X.std(axis=1), X.max(axis=1) - X.min(axis=1)
    return {"models": k, "rows": len(shared), "mean_std": _r(sd.mean()), "max_std": _r(sd.max()),
            "mean_range": _r(rng.mean()), "share_range_over_0_10": _r((rng > 0.10).mean()),
            "note": "models share training data and features; agreement is not independent confirmation"}


# ----------------------------------------------------------------------------- promotion gates
def evaluate_gates(model: Mapping[str, Any], summary: Mapping[str, Any],
                   gates: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Pre-registered promotion gates for shadow -> champion. Every check must pass."""
    g = {**GATES, **(gates or {})}
    metrics = model.get("metrics") or {}
    vs = summary.get("vs_ai_confidence_as_served") or {}
    checks = []

    def check(name: str, value: Any, threshold: Any, ok: bool, note: str = "") -> None:
        checks.append({"gate": name, "value": value, "threshold": threshold, "passed": bool(ok), "note": note})

    check("evaluation_attached", metrics.get("evaluation_version"), "present",
          bool(metrics.get("evaluation_version")), "walk-forward + holdout report on the registry row")
    hold = (metrics.get("holdout") or {})
    check("holdout_evaluated", (hold.get("raw") or {}).get("n"), "> 0", bool((hold.get("raw") or {}).get("n")),
          "final holdout scored once")
    check("status_is_shadow", model.get("status"), "shadow", model.get("status") == "shadow")
    check("labeled_shadow_rows", summary.get("labeled", 0), f">= {g['min_labeled_rows']}",
          summary.get("labeled", 0) >= g["min_labeled_rows"])
    check("shadow_entry_days", summary.get("entry_days", 0), f">= {g['min_entry_days']}",
          summary.get("entry_days", 0) >= g["min_entry_days"])
    check("served_overlap_rows", vs.get("rows", 0), f">= {g['min_served_overlap']}",
          vs.get("rows", 0) >= g["min_served_overlap"])
    d = vs.get("auc_delta")
    check("auc_vs_ai_confidence", d, f">= {g['min_auc_delta_vs_served']}",
          d is not None and d >= g["min_auc_delta_vs_served"], "same matured rows")
    b = vs.get("brier_delta")
    check("brier_vs_ai_confidence", b, f"<= {g['max_brier_vs_served']}",
          b is not None and b <= g["max_brier_vs_served"], "same matured rows")
    e = summary.get("ece")
    check("calibration_ece", e, f"<= {g['max_ece']}", e is not None and e <= g["max_ece"])
    return {"gates_version": GATES_VERSION, "passed": all(c["passed"] for c in checks), "checks": checks,
            "thresholds": g}


def evaluation_evidence(model: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Compact evidence for candidate -> validated, from the evaluation stored on the row."""
    m = model.get("metrics") or {}
    if not m.get("evaluation_version"):
        return None
    keep = ("evaluation_version", "dataset_version", "dataset_fingerprint", "fold_auc")
    ev = {k: m.get(k) for k in keep if k in m}
    for block in ("walk_forward_raw", "walk_forward_calibrated"):
        if isinstance(m.get(block), Mapping):
            ev[block] = {k: m[block].get(k) for k in ("n", "roc_auc", "roc_auc_ci95", "brier", "ece")}
    if isinstance(m.get("holdout"), Mapping):
        ev["holdout"] = {k: ({kk: v.get(kk) for kk in ("n", "roc_auc", "roc_auc_ci95", "brier", "ece")}
                             if isinstance(v, Mapping) else v) for k, v in m["holdout"].items()}
    return {"evaluation": ev}


def iter_chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
