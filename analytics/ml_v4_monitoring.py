"""ML v4 monitoring: is shadow scoring running, and are the shadow models drifting?

Pure and numpy-only (the API and the shadow workflow both call it). Inputs are a
registry model row and its stored shadow predictions; nothing here reads or
writes the database. Checks, each OK / WATCH / WARNING:

  * scoring_fresh       shadow scoring ran within the last 2 completed trading days
  * labels_filling      no prediction still unlabeled 10+ trading days after entry
  * probability_drift   PSI of recent served probabilities vs the model's first
                        shadow weeks (0.10 watch, 0.25 warning)
  * calibration_recent  ECE on the latest matured predictions (0.10 watch, 0.15 warning)
  * vs_ai_confidence    recent AUC minus AI Confidence as served on the same rows
                        (below -0.05 is a watch: the model is falling behind)
  * feature_schema      the model's feature schema matches what scoring builds today

A model's status is its worst check. Thresholds are operational, fixed here,
and reported with every result.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analytics import market_calendar as mc
from analytics import ml_v4_calibration as C
from analytics import ml_v4_shadow as S

MONITORING_VERSION = "hsf-ml-v4-monitoring-1"
THRESHOLDS: Dict[str, Any] = {
    "stale_trading_days": 2,
    "label_overdue_trading_days": 10,
    "psi_watch": 0.10, "psi_warning": 0.25, "psi_min_rows": 100, "baseline_rows": 200, "recent_days": 10,
    "ece_watch": 0.10, "ece_warning": 0.15, "recent_labeled_rows": 150, "min_labeled_rows": 50,
    "auc_watch_delta": -0.05,
}
ORDER = {"OK": 0, "WAITING": 0, "WATCH": 1, "WARNING": 2}


def _dt(v: Any) -> Optional[dt.datetime]:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        d = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _day(v: Any) -> Optional[dt.date]:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def psi(expected: Sequence[float], actual: Sequence[float], bins: Optional[int] = None) -> Optional[float]:
    """Population stability index over the expected sample's quantiles (deciles from
    500 rows each, quintiles below, so small samples don't read as drift)."""
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if len(e) == 0 or len(a) == 0:
        return None
    bins = bins or (10 if min(len(e), len(a)) >= 500 else 5)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return round(float(np.sum((pa - pe) * np.log(pa / pe))), 4)


def _check(name: str, status: str, value: Any, detail: str) -> Dict[str, Any]:
    return {"check": name, "status": status, "value": value, "detail": detail}


def monitor_model(model: Mapping[str, Any], preds: Sequence[Mapping[str, Any]], *, now: dt.datetime,
                  current_feature_schema: Optional[str] = None,
                  thresholds: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    t = {**THRESHOLDS, **(thresholds or {})}
    preds = sorted(preds, key=lambda p: (str(p.get("scored_at")), int(p.get("observation_id") or 0)))
    checks: List[Dict[str, Any]] = []

    # Freshness: only judged once the model has been in shadow over a completed trading day.
    last = max((_dt(p.get("scored_at")) for p in preds if _dt(p.get("scored_at"))), default=None)
    since = _dt(model.get("status_changed_at"))
    if last is None:
        waited = mc.completed_trading_days_since(since, now) if since else 0
        st = "WARNING" if waited >= t["stale_trading_days"] else "WAITING"
        checks.append(_check("scoring_fresh", st, None, "no shadow predictions stored yet"
                             + ("" if st == "WAITING" else f" after {waited} trading days in shadow")))
    else:
        stale = mc.completed_trading_days_since(last, now)
        checks.append(_check("scoring_fresh", "WARNING" if stale >= t["stale_trading_days"] else "OK", stale,
                             f"last scored {last.isoformat()} ({stale} completed trading days ago)"))

    # Labels: a prediction still unlabeled long after its outcome should have matured.
    today = now.astimezone(mc.ET).date()
    overdue = 0
    for p in preds:
        d = _day(p.get("entry_day"))
        if p.get("label") is None and d is not None:
            if sum(1 for _ in mc.trading_days_between(d, today)) - 1 >= t["label_overdue_trading_days"]:
                overdue += 1
    checks.append(_check("labels_filling", "WARNING" if overdue else "OK", overdue,
                         f"{overdue} predictions unlabeled {t['label_overdue_trading_days']}+ trading days after entry"))

    # Probability drift: first shadow weeks vs the last few days.
    probs = [(p, S._prob(p)) for p in preds if S._prob(p) is not None]
    cutoff = now - dt.timedelta(days=t["recent_days"])
    base = [v for _, v in probs[:t["baseline_rows"]]]
    recent = [v for p, v in probs[t["baseline_rows"]:] if (_dt(p.get("scored_at")) or now) >= cutoff]
    if len(base) < t["psi_min_rows"] or len(recent) < t["psi_min_rows"]:
        checks.append(_check("probability_drift", "WAITING", None,
                             f"needs {t['psi_min_rows']}+ baseline and {t['psi_min_rows']}+ recent predictions "
                             f"(have {len(base)} and {len(recent)})"))
    else:
        v = psi(base, recent)
        st = "WARNING" if v >= t["psi_warning"] else "WATCH" if v >= t["psi_watch"] else "OK"
        checks.append(_check("probability_drift", st, v, f"PSI of the last {t['recent_days']} days vs the first "
                             f"{len(base)} shadow predictions"))

    # Calibration and ranking on the latest matured predictions.
    labeled = [p for p, v in probs if p.get("label") is not None][-t["recent_labeled_rows"]:]
    if len(labeled) < t["min_labeled_rows"]:
        checks.append(_check("calibration_recent", "WAITING", None,
                             f"needs {t['min_labeled_rows']}+ matured predictions (have {len(labeled)})"))
        checks.append(_check("vs_ai_confidence", "WAITING", None, "not enough matured predictions yet"))
    else:
        y = np.array([int(p["label"]) for p in labeled])
        pr = np.array([S._prob(p) for p in labeled])
        e = C.ece(y, pr)
        st = "WARNING" if e is not None and e >= t["ece_warning"] else "WATCH" if e is not None and \
            e >= t["ece_watch"] else "OK"
        checks.append(_check("calibration_recent", st, e, f"ECE on the latest {len(labeled)} matured predictions"))
        both = [p for p in labeled if p.get("ai_confidence") is not None]
        if len(both) < t["min_labeled_rows"]:
            checks.append(_check("vs_ai_confidence", "WAITING", None, "too few rows with AI Confidence as served"))
        else:
            yb = np.array([int(p["label"]) for p in both])
            a_m = S.auc(yb, [S._prob(p) for p in both])
            a_s = S.auc(yb, [float(p["ai_confidence"]) / 100.0 for p in both])
            if a_m is None or a_s is None:
                checks.append(_check("vs_ai_confidence", "WAITING", None, "matured rows have only one outcome"))
            else:
                d = round(a_m - a_s, 4)
                checks.append(_check("vs_ai_confidence", "WATCH" if d < t["auc_watch_delta"] else "OK", d,
                                     f"AUC {a_m:.3f} vs AI Confidence {a_s:.3f} on the same {len(both)} rows"))

    if current_feature_schema is not None:
        same = model.get("feature_schema_version") == current_feature_schema
        checks.append(_check("feature_schema", "OK" if same else "WARNING", model.get("feature_schema_version"),
                             "matches today's features" if same else
                             f"scoring now builds {current_feature_schema}; retrain before trusting new scores"))

    status = max((c["status"] for c in checks), key=lambda s: ORDER[s], default="OK")
    return {"model_id": model.get("id"), "model_version": model.get("model_version"), "status": status,
            "checks": checks}


def monitor(models: Sequence[Mapping[str, Any]], preds_by_model: Mapping[int, Sequence[Mapping[str, Any]]], *,
            now: dt.datetime, feature_schemas: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    out = []
    for m in models:
        cur = (feature_schemas or {}).get(str(m.get("feature_set"))) if feature_schemas else None
        out.append(monitor_model(m, preds_by_model.get(int(m["id"]), []), now=now, current_feature_schema=cur))
    status = max((r["status"] for r in out), key=lambda s: ORDER[s], default="OK")
    return {"monitoring_version": MONITORING_VERSION, "generated_at": now.isoformat(), "status": status,
            "models": out, "thresholds": THRESHOLDS}
