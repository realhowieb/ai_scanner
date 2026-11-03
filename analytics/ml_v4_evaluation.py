"""ML v4 walk-forward evaluation: every model on identical, leakage-safe samples.

Methodology (Milestone 3 of the ML v4 plan):
  * Rows: matured, certified signal-days of a finalized research dataset (the
    ML v3 audit layer). When models are compared, only rows EVERY model can
    score are kept, so all models are judged on the same observations; how
    many were dropped is reported.
  * Final holdout: the most recent block (``final_holdout``), reserved before
    any fold is built. It is evaluated once, by models trained on purged,
    embargoed pre-holdout rows, with calibrators fitted only on walk-forward
    out-of-fold predictions. Nothing is tuned or chosen on it.
  * Walk-forward: expanding, chronological folds with purge (training labels
    may not read a validation-period bar) and embargo, the ML v3 engine. No
    shuffling. Hyperparameters are fixed (pre-registered); nothing is tuned.
  * Calibration: ``ml_v4_calibration.choose_and_fit`` on earlier folds only.
    Raw and calibrated probabilities are reported separately.
  * Uncertainty: entry-day block bootstrap CIs (rows of one day share the
    market move), paired for model-vs-model AUC differences.
  * Returns: the top-ranked fraction per fold, after a configurable round-trip
    cost, entry at the observation's frozen price, exit at the horizon close.
    AUC is never treated as proof of profitability; both are reported.

Pure apart from optional model libraries: no database, no network, no writes.
"""
from __future__ import annotations

import datetime as _dt
import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analytics import ml_v3_audit as A
from analytics import ml_v4_calibration as C
from analytics import ml_v4_ensemble as EN
from analytics import ml_v4_providers as P

EVALUATION_VERSION = "hsf-ml-v4-eval-1"
FOLD_KW = dict(min_train=60, min_val=30, min_val_class=5, embargo=1)  # pre-registered ML v4 spec
HOLDOUT_KW = dict(min_rows=100, min_days=5, min_remaining_folds=3)
TOP_FRACTIONS = (0.1, 0.2, 1 / 3)
DEFAULT_COST_BPS = 20.0  # round trip: commission + spread + slippage, conservative for liquid US names
DEFINITIVE_MIN_ROWS = 300
BOOT = 1000

ModelFactory = Callable[[], A.Model]


# ----------------------------------------------------------------------------- champion inputs
class ServedProbability(A.Model):
    """A probability already shown to users (e.g. AI Confidence %), used as is:
    no fitting, so its Brier and calibration are what customers actually saw."""
    kind = "served_probability"

    def __init__(self, name: str, key: str, scale: float = 100.0):
        self.name, self.key, self.scale = name, key, scale
        self.hyperparameters = {"input": key, "scale": scale, "fit": "none (as served)"}

    def _value(self, r: A.AuditRow) -> Optional[float]:
        v = r.extra.get(self.key)
        try:
            f = float(v) / self.scale
        except (TypeError, ValueError):
            return None
        return min(max(f, 0.0), 1.0) if math.isfinite(f) else None

    def available(self, r):
        return self._value(r) is not None

    def predict(self, rows):
        return np.array([self._value(r) for r in rows], dtype=float)


def attach_served(rows: Sequence[A.AuditRow], served: Mapping[int, Any], key: str = "ai_confidence_served") -> int:
    """Put frozen signal-time values (by observation id) on rows' ``extra``. Returns how many matched."""
    n = 0
    for r in rows:
        v = served.get(int(r.observation_id))
        if v is not None:
            r.extra[key] = v
            n += 1
    return n


def default_models(*, providers: Sequence[str] = P.PROVIDERS, feature_sets: Sequence[str] = P.DEFAULT_FEATURE_SETS,
                   include_champion: bool = False, params: Optional[Mapping[str, Mapping[str, Any]]] = None
                   ) -> Dict[str, ModelFactory]:
    """Baselines + every provider x feature set (+ the served champion)."""
    params = params or {}
    models: Dict[str, ModelFactory] = {
        "majority_baseline": A.Majority,
        "hsf_score": lambda: A.ScoreModel("hsf_score", "hsf_score"),
    }
    if include_champion:
        models["ai_confidence_as_served"] = lambda: ServedProbability("ai_confidence_as_served", "ai_confidence_served")
    for fs in feature_sets:
        for p in providers:
            models[f"{p}_{fs}"] = (lambda p=p, fs=fs: P.ProviderModel(p, fs, params.get(p)))
    return models


def default_ensembles(*, providers: Sequence[str] = P.PROVIDERS,
                      feature_sets: Sequence[str] = P.DEFAULT_FEATURE_SETS,
                      methods: Sequence[str] = EN.METHODS) -> Dict[str, Tuple[str, List[str]]]:
    """One ensemble per method x feature set over that feature set's providers."""
    if len(providers) < 2:
        return {}
    return {f"ensemble_{m}_{fs}": (m, [f"{p}_{fs}" for p in providers]) for fs in feature_sets for m in methods}


# ----------------------------------------------------------------------------- helpers
def _auc(y, p) -> Optional[float]:
    """ROC-AUC via average ranks (ties count half): same value as sklearn, fast
    enough for thousands of bootstrap resamples."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if len(y) == 0 or n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(np.asarray(p, dtype=float))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_bootstrap_auc(y, p, days, *, n_boot: Optional[int] = None, seed: int = P.SEED) -> Optional[Dict[str, Any]]:
    """95% CI for ROC-AUC resampling whole entry days (rows of one day share the market move)."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, d in enumerate(days):
        groups[d].append(i)
    keys = sorted(groups)
    n_boot = BOOT if n_boot is None else n_boot
    if len(keys) < 3 or n_boot <= 0 or _auc(y, p) is None:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in pick for i in groups[keys[k]]]
        a = _auc(y[idx], p[idx])
        if a is not None:
            vals.append(a)
    if len(vals) < n_boot * 0.5:
        return None
    return {"low": _r(np.percentile(vals, 2.5)), "high": _r(np.percentile(vals, 97.5)),
            "method": f"entry-day block bootstrap, {len(vals)} resamples"}


def _r(v, d=4):
    return None if v is None else round(float(v), d)


def paired_delta_auc(y, pa, pb, days, *, n_boot: Optional[int] = None, seed: int = P.SEED) -> Optional[Dict[str, Any]]:
    """AUC(a) - AUC(b) on the same rows with an entry-day paired block bootstrap 95% CI."""
    y, pa, pb = np.asarray(y, dtype=int), np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
    point_a, point_b = _auc(y, pa), _auc(y, pb)
    if point_a is None or point_b is None:
        return None
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, d in enumerate(days):
        groups[d].append(i)
    keys = sorted(groups)
    out = {"delta": _r(point_a - point_b), "ci95": None}
    n_boot = BOOT if n_boot is None else n_boot
    if len(keys) < 3 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in pick for i in groups[keys[k]]]
        a, b = _auc(y[idx], pa[idx]), _auc(y[idx], pb[idx])
        if a is not None and b is not None:
            vals.append(a - b)
    if len(vals) >= n_boot * 0.5:
        out["ci95"] = {"low": _r(np.percentile(vals, 2.5)), "high": _r(np.percentile(vals, 97.5)),
                       "resamples": len(vals)}
    return out


def _returns(rows: Sequence[A.AuditRow], h: int, cost_bps: float) -> Dict[str, Any]:
    vals = [r.labels.get(f"return_{h}d") for r in rows]
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    exc = [r.labels.get(f"excess_return_{h}d") for r in rows]
    exc = [float(v) for v in exc if v is not None and math.isfinite(float(v))]
    cost = cost_bps / 10000.0
    if not vals:
        return {"n": 0}
    net = [v - cost for v in vals]
    return {"n": len(vals), "mean_gross": _r(statistics.fmean(vals)), "mean_net": _r(statistics.fmean(net)),
            "median_net": _r(statistics.median(net)), "win_rate_net": _r(sum(1 for v in net if v > 0) / len(net)),
            "mean_excess_vs_spy": _r(statistics.fmean(exc)) if exc else None, "excess_n": len(exc)}


def _group_metrics(y, p, keys) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k if k is not None else "unknown"].append(i)
    out = []
    for k in sorted(groups, key=lambda x: (-len(groups[x]), str(x))):
        idx = groups[k]
        yy, pp = np.asarray(y)[idx], np.asarray(p)[idx]
        out.append({"group": str(k), "n": len(idx), "hit_rate": _r(yy.mean()), "mean_probability": _r(pp.mean()),
                    "roc_auc": _r(_auc(yy, pp)) if len(idx) >= 20 else None,
                    "note": None if len(idx) >= 20 else "fewer than 20 rows: no AUC"})
    return out


def _metrics_block(y, p, days, n_boot: Optional[int] = None) -> Dict[str, Any]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n": 0}
    m = A.classification_metrics(y, p, float(y.mean()))
    m["roc_auc_ci95"] = block_bootstrap_auc(y, p, days, n_boot=n_boot)
    m["ece"] = C.ece(y, p)
    m["reliability"] = C.reliability(y, p)
    return m


# ----------------------------------------------------------------------------- engine
def evaluate(data: Sequence[Tuple[A.AuditRow, int]], models: Mapping[str, ModelFactory], *,
             target: P.Target = P.PRIMARY_TARGET, fold_kw: Optional[Mapping[str, Any]] = None,
             holdout_kw: Optional[Mapping[str, Any]] = None, cost_bps: float = DEFAULT_COST_BPS,
             regime_by_day: Optional[Mapping[_dt.date, Mapping[str, Any]]] = None,
             reference: Optional[str] = None, detail: bool = True,
             ensembles: Optional[Mapping[str, Tuple[str, Sequence[str]]]] = None) -> Dict[str, Any]:
    """Walk-forward + holdout evaluation of ``models`` (and ``ensembles`` of them, as
    name -> (method, member names)) on the common scorable sample of ``data``."""
    h = target.horizon
    nb = None if detail else 0  # compact runs skip bootstrap CIs
    fk = dict(FOLD_KW if fold_kw is None else fold_kw)
    hk = dict(HOLDOUT_KW if holdout_kw is None else holdout_kw)
    probes = {name: f() for name, f in models.items()}
    common = [(r, y) for r, y in data if all(m.available(r) for m in probes.values())]
    out: Dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "target": {"label": target.label, "horizon_days": h, "description": target.description},
        "sample": {"eligible_rows": len(data), "common_rows": len(common), "dropped_not_scorable_by_all": len(data) - len(common),
                   "positive": int(sum(y for _, y in common)),
                   "positive_rate": _r(sum(y for _, y in common) / len(common)) if common else None,
                   "entry_days": len({r.entry_day for r, _ in common}), "tickers": len({r.ticker for r, _ in common})},
        "fold_spec": fk, "holdout_spec": hk, "cost_bps_round_trip": cost_bps,
        "models": {}, "comparisons": {}, "warnings": [],
    }
    hold = A.final_holdout(common, h, **hk, **fk) if common else {"valid": False, "reason": "no rows"}
    if hold.get("valid"):
        hstart = _dt.date.fromisoformat(hold["start"])
        wf = [(r, y) for r, y in common if r.entry_day < hstart]
    else:
        hstart, wf = None, list(common)
    folds = A.walk_forward_folds(wf, h, **fk)
    out["folds"] = [{k: getattr(f, k) for k in ("fold", "train_start", "train_end", "validation_start", "validation_end",
                                                  "purged", "embargoed")} | {"train_n": len(f.train_ids), "validation_n": len(f.val_ids)}
                    for f in folds]
    out["holdout"] = {"valid": bool(hold.get("valid")), "start": hold.get("start"), "rows": hold.get("rows", 0),
                      "reason": hold.get("reason")}
    by_id = {r.observation_id: (r, y) for r, y in common}
    pooled_ids: Optional[List[int]] = None
    oof: Dict[str, Dict[str, Any]] = {}
    raw_by_model: Dict[str, Dict[str, Any]] = {}  # per-fold raw predictions, reused by ensembles

    def finish(name: str, fold_preds, holdout_pred, *, calibrate: bool, extra: Optional[Dict[str, Any]] = None):
        """Calibration (earlier folds only), pooled metrics, breakdowns and holdout for one model."""
        nonlocal pooled_ids
        chunks: List[Tuple[List[int], List[float]]] = []
        rec = {"y": [], "raw": [], "cal": [], "ids": [], "days": [], "fold": []}
        fold_rows, cal_meta = [], []
        for fold_no, va, yva, raw in fold_preds:
            cal = C.choose_and_fit(chunks) if calibrate else None
            calp = C.apply(cal, raw) if cal else np.full(len(raw), np.nan)
            cal_meta.append({"fold": fold_no, "method": cal["method"] if cal else None,
                             "selection": cal["selection"]["rule"] if cal else (
                                 "not fitted (needs earlier folds)" if calibrate else "never recalibrated (as served)")})
            fold_rows.append({"fold": fold_no, "n": len(va), "roc_auc": _r(_auc(yva, raw)),
                              "brier_raw": _r(C.brier(yva, raw)) if len(va) else None,
                              "brier_calibrated": _r(C.brier(yva, calp)) if cal else None})
            chunks.append((yva.tolist(), raw.tolist()))
            for (r, y), a, b in zip(va, raw, calp):
                rec["y"].append(int(y))
                rec["raw"].append(float(a))
                rec["cal"].append(float(b))
                rec["ids"].append(r.observation_id)
                rec["days"].append(r.entry_day.isoformat())
                rec["fold"].append(fold_no)
        if pooled_ids is None:
            pooled_ids = rec["ids"]
        elif rec["ids"] != pooled_ids:  # identical samples are the point of the comparison
            raise AssertionError(f"{name} was evaluated on different rows")
        oof[name] = rec
        y = np.array(rec["y"], dtype=int)
        raw = np.array(rec["raw"], dtype=float)
        calp = np.array(rec["cal"], dtype=float)
        has_cal = ~np.isnan(calp)
        mres: Dict[str, Any] = {"folds": fold_rows, "calibration_per_fold": cal_meta,
                                "fold_auc": A.summarize([fr["roc_auc"] for fr in fold_rows])}
        if extra:
            mres.update(extra)
        mres["walk_forward_raw"] = _metrics_block(y, raw, rec["days"], nb)
        if has_cal.any():
            d = [rec["days"][i] for i in np.flatnonzero(has_cal)]
            mres["walk_forward_calibrated"] = _metrics_block(y[has_cal], calp[has_cal], d, nb)
            mres["walk_forward_calibrated"]["rows_with_calibration"] = int(has_cal.sum())
            mres["walk_forward_raw_same_rows"] = {"brier": _r(C.brier(y[has_cal], raw[has_cal])),
                                                  "ece": C.ece(y[has_cal], raw[has_cal])}
        if detail and len(y):
            rows = [by_id[i][0] for i in rec["ids"]]
            # Precision at top K (per fold, pooled) and trading returns after costs.
            topk, returns = {}, {}
            for frac in TOP_FRACTIONS:
                sel = []
                for fo in sorted(set(rec["fold"])):
                    idx = [i for i, ff in enumerate(rec["fold"]) if ff == fo]
                    order = sorted(idx, key=lambda i: (-raw[i], rec["ids"][i]))
                    sel += order[:max(1, int(round(len(idx) * frac)))]
                key = f"top_{round(frac * 100)}pct"
                topk[key] = {"n": len(sel), "hit_rate": _r(y[sel].mean()), "base_rate": _r(y.mean())}
                returns[key] = _returns([rows[i] for i in sel], h, cost_bps)
            returns["all_validation"] = _returns(rows, h, cost_bps)
            mres["precision_at_top"] = topk
            mres["returns_after_costs"] = returns
            score = np.where(has_cal, calp, raw) if has_cal.all() else raw
            mres["by_setup"] = _group_metrics(y, score, [r.setup for r in rows])
            if regime_by_day:
                mres["by_regime"] = _group_metrics(
                    y, score, [(regime_by_day.get(r.entry_day) or {}).get("trend") for r in rows])
        # Final holdout: evaluated once.
        if holdout_pred is not None:
            train_n, yh, hraw, hd = holdout_pred
            cal = C.choose_and_fit(chunks) if calibrate else None
            mres["holdout"] = {"train_n": train_n, "raw": _metrics_block(yh, hraw, hd, nb),
                               "calibrator": ({k: cal[k] for k in ("method", "n", "selection")} if cal else None)}
            if cal:
                mres["holdout"]["calibrated"] = _metrics_block(yh, C.apply(cal, hraw), hd, nb)
            oof[name]["holdout"] = {"y": yh.tolist(), "raw": hraw.tolist(), "days": hd}
        out["models"][name] = mres

    ho_rows = [by_id[i] for i in hold["ids"]] if hold.get("valid") else []
    ho_train = []
    if hold.get("valid"):
        kept, _, _ = A.purge_train([r for r, _ in wf], hstart, h, fk.get("embargo", 1))
        ho_train = [by_id[r.observation_id] for r in kept]
    for name, factory in models.items():
        fold_preds = []
        for f in folds:
            tr = [by_id[i] for i in f.train_ids]
            va = [by_id[i] for i in f.val_ids]
            m = factory().fit([r for r, _ in tr], np.array([y for _, y in tr], dtype=int))
            raw = np.asarray(m.predict([r for r, _ in va]), dtype=float)
            fold_preds.append((f.fold, va, np.array([y for _, y in va], dtype=int), raw))
        holdout_pred = None
        ytr = np.array([y for _, y in ho_train], dtype=int)
        if ho_rows and len(ho_train) and 0 < ytr.sum() < len(ytr):
            m = factory().fit([r for r, _ in ho_train], ytr)
            holdout_pred = (len(ho_train), np.array([y for _, y in ho_rows], dtype=int),
                            np.asarray(m.predict([r for r, _ in ho_rows]), dtype=float),
                            [r.entry_day.isoformat() for r, _ in ho_rows])
        calibrate = getattr(probes[name], "kind", "") != "served_probability"
        raw_by_model[name] = {"folds": fold_preds, "holdout": holdout_pred}
        finish(name, fold_preds, holdout_pred, calibrate=calibrate)

    # Ensembles: member predictions per fold, combined with weights fitted on EARLIER folds only.
    for ens_name, (method, members) in (ensembles or {}).items():
        members = [m for m in members if m in raw_by_model]
        if len(members) < 2:
            continue
        fold_preds, specs = [], []
        for k, (fold_no, va, yva, _) in enumerate(raw_by_model[members[0]]["folds"]):
            y_e = np.concatenate([raw_by_model[members[0]]["folds"][j][2] for j in range(k)]) if k else np.zeros(0)
            p_e = {m: (np.concatenate([raw_by_model[m]["folds"][j][3] for j in range(k)]) if k else np.zeros(0))
                   for m in members}
            spec = EN.fit(method, members, y_e, p_e)
            p_now = {m: raw_by_model[m]["folds"][k][3] for m in members}
            fold_preds.append((fold_no, va, yva, EN.predict(spec, p_now) if len(va) else np.zeros(0)))
            specs.append({"fold": fold_no, **{k2: v for k2, v in spec.items() if k2 != "members"}})
        all_y = np.concatenate([fp[2] for fp in raw_by_model[members[0]]["folds"]]) if fold_preds else np.zeros(0)
        all_p = {m: (np.concatenate([fp[3] for fp in raw_by_model[m]["folds"]]) if fold_preds else np.zeros(0))
                 for m in members}
        holdout_pred, holdout_spec = None, None
        if all(raw_by_model[m]["holdout"] is not None for m in members):
            holdout_spec = EN.fit(method, members, all_y, all_p)
            h0 = raw_by_model[members[0]]["holdout"]
            hp = EN.predict(holdout_spec, {m: raw_by_model[m]["holdout"][2] for m in members})
            holdout_pred = (h0[0], h0[1], hp, h0[3])
        extra = {"ensemble": {"method": method, "members": members, "per_fold": specs,
                              "holdout_weights": holdout_spec,
                              "disagreement": EN.disagreement(members, all_p) if len(all_y) else {"rows": 0}}}
        finish(ens_name, fold_preds, holdout_pred, calibrate=True, extra=extra)
        # Compare with the best single member (chosen on the same pooled out-of-fold log loss,
        # which flatters the member: a conservative bar for the ensemble).
        if len(all_y) and detail:
            best = min(members, key=lambda m: EN.log_loss(all_y, all_p[m]))
            cmp_ = {"best_member": best,
                    "walk_forward": paired_delta_auc(oof[ens_name]["y"], oof[ens_name]["raw"], oof[best]["raw"],
                                                     oof[ens_name]["days"]),
                    "log_loss_ensemble": _r(EN.log_loss(all_y, oof[ens_name]["raw"])),
                    "log_loss_best_member": _r(EN.log_loss(all_y, all_p[best]))}
            if "holdout" in oof[ens_name] and "holdout" in oof[best]:
                hr = oof[ens_name]["holdout"]
                cmp_["holdout"] = paired_delta_auc(hr["y"], hr["raw"], oof[best]["holdout"]["raw"], hr["days"])
            out["models"][ens_name]["vs_best_member"] = cmp_

    # Paired comparisons on identical rows.
    ref = reference if reference in oof else next((n for n in ("ai_confidence_as_served", "xgboost_production")
                                                   if n in oof), None)
    if ref and detail:
        for name, rec in oof.items():
            if name == ref:
                continue
            cmp_ = {"walk_forward": paired_delta_auc(rec["y"], rec["raw"], oof[ref]["raw"], rec["days"])}
            if "holdout" in rec and "holdout" in oof[ref]:
                hr = rec["holdout"]
                cmp_["holdout"] = paired_delta_auc(hr["y"], hr["raw"], oof[ref]["holdout"]["raw"], hr["days"])
            out["comparisons"][name] = cmp_
        out["comparison_reference"] = ref
    out["warnings"] = warnings(out)
    return out


def warnings(res: Mapping[str, Any]) -> List[str]:
    w = []
    s = res["sample"]
    if s["common_rows"] < DEFINITIVE_MIN_ROWS:
        w.append(f"INSUFFICIENT_DATA: {s['common_rows']} rows < {DEFINITIVE_MIN_ROWS}; results are diagnostic only")
    if len(res.get("folds") or []) < 3:
        w.append(f"FEW_FOLDS: {len(res.get('folds') or [])} walk-forward folds (< 3); stability can't be judged")
    if not res["holdout"]["valid"]:
        w.append(f"NO_HOLDOUT: {res['holdout'].get('reason')}")
    if s["positive"] < 30 or (s["common_rows"] - s["positive"]) < 30:
        w.append("CLASS_IMBALANCE: fewer than 30 rows in one class")
    if s["dropped_not_scorable_by_all"]:
        w.append(f"COMMON_SAMPLE: {s['dropped_not_scorable_by_all']} rows dropped because not every model can score them")
    for name, m in res["models"].items():
        ci = (m.get("walk_forward_raw") or {}).get("roc_auc_ci95")
        if ci and ci["high"] - ci["low"] > 0.2:
            w.append(f"WIDE_CI: {name} AUC 95% CI {ci['low']}-{ci['high']}")
            break
    return w


def evaluate_targets(rows: Sequence[A.AuditRow], targets: Sequence[P.Target], models: Mapping[str, ModelFactory],
                     **kw) -> List[Dict[str, Any]]:
    """Compact per-target results (performance by horizon and outcome definition)."""
    out = []
    for t in targets:
        data = A.eligible(rows, t.horizon, t.label)
        if not data:
            out.append({"target": t.label, "horizon_days": t.horizon, "rows": 0})
            continue
        res = evaluate(data, models, target=t, detail=False, **kw)
        out.append({"target": t.label, "horizon_days": t.horizon, "rows": res["sample"]["common_rows"],
                    "positive_rate": res["sample"]["positive_rate"], "folds": len(res["folds"]),
                    "models": {n: {"roc_auc": (m.get("walk_forward_raw") or {}).get("roc_auc"),
                                   "brier_raw": (m.get("walk_forward_raw") or {}).get("brier"),
                                   "ece_calibrated": (m.get("walk_forward_calibrated") or {}).get("ece")}
                               for n, m in res["models"].items()},
                    "warnings": res["warnings"]})
    return out


# ----------------------------------------------------------------------------- report
def _f(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _ci(ci, sep="-"):
    return f"{ci['low']:.3f}{sep}{ci['high']:.3f}" if ci else "-"


def render_markdown(result: Mapping[str, Any], *, title: str = "ML v4 walk-forward evaluation") -> str:
    s, t = result["sample"], result["target"]
    lines = [f"# {title}", "",
             f"Target: **{t['label']}** at {t['horizon_days']} trading days ({t['description']}).", "",
             f"Sample: {s['common_rows']} signal-days ({s['positive']} positive, {_f(s['positive_rate'], True)}), "
             f"{s['entry_days']} entry days, {s['tickers']} tickers; {s['dropped_not_scorable_by_all']} dropped so every "
             f"model sees the same rows. Folds: {len(result['folds'])}. Holdout: "
             + (f"from {result['holdout']['start']}, {result['holdout']['rows']} rows (evaluated once)."
                if result["holdout"]["valid"] else f"not reserved ({result['holdout']['reason']})."), ""]
    if result["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    lines += ["## Walk-forward (out of fold)", "",
              "| Model | AUC raw | AUC 95% CI | PR-AUC | Brier raw | Brier cal | ECE raw | ECE cal | Top 20% hit | Top 33% net 5d return |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in result["models"].items():
        raw, cal = m.get("walk_forward_raw") or {}, m.get("walk_forward_calibrated") or {}
        ci = raw.get("roc_auc_ci95")
        top = (m.get("precision_at_top") or {}).get("top_20pct") or {}
        ret = ((m.get("returns_after_costs") or {}).get("top_33pct") or {})
        lines.append(f"| {name} | {_f(raw.get('roc_auc'))} | {_ci(ci)} | "
                     f"{_f(raw.get('pr_auc'))} | {_f(raw.get('brier'))} | {_f(cal.get('brier'))} | {_f(raw.get('ece'))} | "
                     f"{_f(cal.get('ece'))} | {_f(top.get('hit_rate'), True)} | {_f(ret.get('mean_net'), True)} |")
    if any("holdout" in m for m in result["models"].values()):
        lines += ["", "## Final holdout (evaluated once)", "", "| Model | AUC raw | AUC 95% CI | Brier raw | Brier cal | ECE cal |",
                  "|---|---|---|---|---|---|"]
        for name, m in result["models"].items():
            ho = m.get("holdout")
            if not ho:
                continue
            raw, cal = ho.get("raw") or {}, ho.get("calibrated") or {}
            ci = raw.get("roc_auc_ci95")
            lines.append(f"| {name} | {_f(raw.get('roc_auc'))} | {_ci(ci)} | "
                         f"{_f(raw.get('brier'))} | {_f(cal.get('brier'))} | {_f(cal.get('ece'))} |")
    if result.get("comparisons"):
        lines += ["", f"## AUC difference vs {result['comparison_reference']} (paired, same rows)", "",
                  "| Model | Walk-forward delta | 95% CI | Holdout delta | 95% CI |", "|---|---|---|---|---|"]
        for name, c in result["comparisons"].items():
            wf, ho = c.get("walk_forward") or {}, c.get("holdout") or {}
            wci, hci = wf.get("ci95"), ho.get("ci95")
            lines.append(f"| {name} | {_f(wf.get('delta'))} | {_ci(wci, ' to ')} | "
                         f"{_f(ho.get('delta'))} | {_ci(hci, ' to ')} |")
    ens = {n: m["vs_best_member"] for n, m in result["models"].items() if m.get("vs_best_member")}
    if ens:
        lines += ["", "## Ensembles vs their best single member (paired, same rows)", "",
                  "| Ensemble | Best member | AUC delta | 95% CI | Log loss ens. | Log loss member | Holdout delta |",
                  "|---|---|---|---|---|---|---|"]
        for name, c in ens.items():
            wf, ho = c.get("walk_forward") or {}, c.get("holdout") or {}
            lines.append(f"| {name} | {c['best_member']} | {_f(wf.get('delta'))} | {_ci(wf.get('ci95'), ' to ')} | "
                         f"{_f(c.get('log_loss_ensemble'))} | {_f(c.get('log_loss_best_member'))} | {_f(ho.get('delta'))} |")
        lines += ["", "An ensemble is promotable only with a meaningful out-of-sample gain over its best member. "
                      "Members share data and features, so their agreement is not independent confirmation."]
    lines += ["", f"Returns: entry at the frozen observation price, exit at the horizon close, minus "
                  f"{result['cost_bps_round_trip']:.0f} bps round trip. AUC is not evidence of profitability. "
                  "Probabilities are for the target event above, not a probability of profit.", ""]
    return "\n".join(lines)
