"""Predeclared feature comparison for the December recipe lock (pure, no I/O).

Question: on development data only, do RSI-14 and/or the SPY/SMA20 regime add
anything on top of the six repaired core inputs? Everything that decides the
answer is fixed in the protocol file BEFORE any result is seen:

  * rows: signal-day picks from a frozen research-schema-v2 dataset, inside the
    development window; the reserved forward test (Dec 23+) is never read;
  * target: the primary ML v4 target (+4% before -2% in 5 trading days);
  * arms: core_v2, core_rsi_v2, core_regime_v2, core_rsi_regime_v2, all on the
    identical rows, folds and model (LightGBM, the fixed engine defaults);
  * periods: one validation block per ISO week; training = every earlier row
    whose 5-day outcome window closed before the week (purge + embargo);
  * keep rule: an addition must beat core in enough weeks AND by enough on
    average, in the main run and (direction only) in a check run trained from
    the first RSI/regime day, so a gain that only comes from "RSI missing =
    older data" doesn't count.

``decide`` turns per-week AUCs into the locked recommendation with no
discretion left.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analytics import ml_v3_audit as A

ARMS = ("core_v2", "core_rsi_v2", "core_regime_v2", "core_rsi_regime_v2")
BASELINE = "core_v2"
COMBINED = "core_rsi_regime_v2"


def iso_week(d: _dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02}"


def check_window(rows: Sequence[A.AuditRow], protocol: Mapping[str, Any]) -> None:
    """Refuse anything outside the development window (before reading labels is
    the loader's job; this is the second, independent check)."""
    lo = _dt.date.fromisoformat(protocol["development_start"])
    hi = _dt.date.fromisoformat(protocol["development_end_inclusive"])
    reserved = _dt.date.fromisoformat(protocol["reserved_start"])
    if hi >= reserved:
        raise ValueError("protocol development window reaches the reserved test")
    for r in rows:
        day = r.observed_at.date()
        if day >= reserved:
            raise ValueError("row inside the reserved forward-test window; refusing")
        if not lo <= day <= hi:
            raise ValueError("row outside the protocol development window; refusing")


def weekly_folds(data: Sequence[Tuple[A.AuditRow, int]], protocol: Mapping[str, Any], *,
                 train_from: Optional[_dt.date] = None) -> List[Dict[str, Any]]:
    """One fold per validation ISO week (deterministic)."""
    p = protocol["periods"]
    h, embargo = int(protocol["horizon"]), int(p["embargo_trading_days"])
    vstart = _dt.date.fromisoformat(p["first_validation_day"])
    by_week: Dict[str, List[Tuple[A.AuditRow, int]]] = defaultdict(list)
    for r, y in data:
        if r.entry_day >= vstart:
            by_week[iso_week(r.entry_day)].append((r, y))
    label = {r.observation_id: y for r, y in data}
    pool = [r for r, _ in data if train_from is None or r.entry_day >= train_from]
    folds = []
    for week in sorted(by_week):
        block = by_week[week]
        start = min(r.entry_day for r, _ in block)
        ys = [y for _, y in block]
        train, purged, emb = A.purge_train(pool, start, h, embargo)
        tys = [label[r.observation_id] for r in train]
        status = "ok"
        if len(block) < p["min_validation_rows"] or min(sum(ys), len(ys) - sum(ys)) < p["min_validation_per_class"]:
            status = "validation_too_small"
        elif len(train) < p["min_training_rows"] or not 0 < sum(tys) < len(tys):
            status = "training_too_small"
        folds.append({"week": week, "status": status, "validation": [r for r, _ in block], "y_val": ys,
                      "train": train, "y_train": tys, "purged": purged, "embargoed": emb})
    return folds


def auc(y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def within_day_auc(rows: Sequence[A.AuditRow], y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    """Mean AUC inside each entry day (ranking among same-day picks only)."""
    by_day: Dict[_dt.date, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r.entry_day].append(i)
    vals = [auc([y[i] for i in idx], [p[i] for i in idx]) for idx in by_day.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def run_arms(folds: Sequence[Mapping[str, Any]], model_factory: Callable[[str], Any],
             arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """Fit every arm on every usable fold; identical rows per fold across arms."""
    weeks = []
    for f in folds:
        entry: Dict[str, Any] = {"week": f["week"], "status": f["status"], "validation_rows": len(f["validation"]),
                                 "validation_positives": int(sum(f["y_val"])), "training_rows": len(f["train"]),
                                 "purged": f["purged"], "embargoed": f["embargoed"], "auc": {}, "within_day_auc": {}}
        if f["status"] == "ok":
            for arm in arms:
                m = model_factory(arm).fit(f["train"], f["y_train"])
                p = m.predict(f["validation"])
                entry["auc"][arm] = auc(f["y_val"], p)
                entry["within_day_auc"][arm] = within_day_auc(f["validation"], f["y_val"], p)
        weeks.append(entry)
    return {"weeks": weeks}


def _deltas(weeks, arm, ref):
    return [w["auc"][arm] - w["auc"][ref] for w in weeks
            if w["status"] == "ok" and w["auc"].get(arm) is not None and w["auc"].get(ref) is not None]


def passes(weeks: Sequence[Mapping[str, Any]], arm: str, ref: str, rule: Mapping[str, Any]) -> Dict[str, Any]:
    d = _deltas(weeks, arm, ref)
    out = {"arm": arm, "versus": ref, "weeks": len(d),
           "weeks_better": sum(1 for x in d if x > 0),
           "mean_auc_gain": round(float(np.mean(d)), 4) if d else None}
    if len(d) < rule["min_weeks"]:
        out.update(passed=False, reason="too few usable weeks")
        return out
    share = out["weeks_better"] / len(d)
    out["share_better"] = round(share, 3)
    ok = share >= rule["min_share_of_weeks_better"] - 1e-9 and out["mean_auc_gain"] >= rule["min_mean_auc_gain"]
    out.update(passed=bool(ok), reason="meets rule" if ok else "below rule")
    return out


def decide(main: Mapping[str, Any], check: Mapping[str, Any], rule: Mapping[str, Any]) -> Dict[str, Any]:
    """Locked recommendation from the main run + the from-first-RSI-day check."""
    weeks, cweeks = main["weeks"], check["weeks"]
    tests = {}
    for arm in ARMS[1:]:
        t = passes(weeks, arm, BASELINE, rule)
        cd = _deltas(cweeks, arm, BASELINE)
        t["check_mean_auc_gain"] = round(float(np.mean(cd)), 4) if cd else None
        if t["passed"] and not (cd and float(np.mean(cd)) > 0):
            t.update(passed=False, reason="gain disappears when training starts at the first RSI/regime day")
        tests[arm] = t
    singles = [a for a in ("core_rsi_v2", "core_regime_v2") if tests[a]["passed"]]
    choice, why = BASELINE, "no addition met the keep rule; keep the six repaired core inputs"
    if tests[COMBINED]["passed"]:
        if not singles:
            choice, why = COMBINED, "only RSI + regime together met the keep rule"
        else:
            best = max(singles, key=lambda a: tests[a]["mean_auc_gain"])
            vs = passes(weeks, COMBINED, best, rule)
            tests[f"{COMBINED}_vs_{best}"] = vs
            choice, why = ((COMBINED, f"RSI + regime beat core and beat {best} alone") if vs["passed"]
                           else (best, f"{best} met the rule; adding the other input did not help enough"))
    elif singles:
        choice = max(singles, key=lambda a: tests[a]["mean_auc_gain"])
        why = f"{choice} met the keep rule"
    return {"recommended_feature_set": choice, "reason": why, "tests": tests}


def summary_auc(result: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    out = {}
    for arm in ARMS:
        v = [w["auc"][arm] for w in result["weeks"] if w["status"] == "ok" and w["auc"].get(arm) is not None]
        out[arm] = round(float(np.mean(v)), 4) if v else None
    return out


def coverage(rows: Sequence[A.AuditRow], names: Sequence[str]) -> Dict[str, Any]:
    """Per-week share of rows carrying each input (no labels)."""
    by_week: Dict[str, List[A.AuditRow]] = defaultdict(list)
    for r in rows:
        by_week[iso_week(r.entry_day)].append(r)
    out = {}
    for week in sorted(by_week):
        g = by_week[week]
        out[week] = {"rows": len(g), **{n: round(100 * sum(1 for r in g if _present(r.features.get(n))) / len(g), 1)
                                        for n in names}}
    return out


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return True
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return bool(v)
