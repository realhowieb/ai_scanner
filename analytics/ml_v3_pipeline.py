"""ML v3 audit orchestration: frozen dataset in, every audit result out.

Pure apart from optional inputs the caller loads (the active AI Confidence
model object, SPY closes for regime labels, the legacy runs-table frame).
No database writes, no model persistence, no score changes. See
``analytics.ml_v3_audit`` for the methodology.
"""
from __future__ import annotations

import datetime as _dt
import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analytics import ml_v3_audit as A
from analytics import research_dataset as rd
from analytics import research_schema as rs

PRIMARY_HORIZON = 5  # pre-declared (Outcome Intelligence DEFAULT_HORIZON), not chosen from results
FOLD_KW = dict(min_train=60, min_val=30, min_val_class=5, embargo=1)
HOLDOUT_KW = dict(min_rows=100, min_days=5, min_remaining_folds=3)
DEFINITIVE_MIN_ROWS = 300  # matured, certified signal-days at the primary horizon


# ----------------------------------------------------------------------------- as-of-freeze view
def as_of(rows: Sequence[Mapping[str, Any]], frozen_at: Optional[_dt.datetime]) -> List[Dict[str, Any]]:
    """Outcome columns exactly as they stood at ``frozen_at``.

    An outcome computed after the freeze is treated as pending; a benchmark
    computed after the freeze is treated as absent. Features are insert-once,
    so they never change. Reruns of the audit therefore see the same labels
    even while the cron keeps maturing rows and backfilling SPY."""
    if frozen_at is None:
        return [dict(r) for r in rows]
    out = []
    for r in rows:
        r = dict(r)
        oc = rd.to_dt(r.get("outcome_computed_at"))
        if oc is not None and oc > frozen_at:
            for k in ("return_1d", "return_3d", "return_5d", "mfe_5d", "mae_5d",
                      "benchmark_return_1d", "benchmark_return_3d", "benchmark_return_5d"):
                if k in r:
                    r[k] = None
            r["outcome_computed_at"] = None
        bc = rd.to_dt(r.get("benchmark_computed_at"))
        if bc is not None and bc > frozen_at:
            for k in ("benchmark_return_1d", "benchmark_return_3d", "benchmark_return_5d"):
                if k in r:
                    r[k] = None
        out.append(r)
    return out


# ----------------------------------------------------------------------------- data quality
def data_quality(raw_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                 obs_rows: Sequence[A.AuditRow], day_rows: Sequence[A.AuditRow]) -> Dict[str, Any]:
    base = rd.quality_report(raw_rows, records)
    n = len(obs_rows)
    same_time = Counter((r.ticker, r.observed_at) for r in obs_rows)
    same_day = Counter((r.ticker, r.observed_day) for r in obs_rows)
    prices = [r.features.get("price") for r in obs_rows]
    scores = [r.features.get("hsf_score") for r in obs_rows]
    mat5 = [r for r in obs_rows if r.matured(5)]
    null_rates = {c: round(sum(1 for r in obs_rows if r.features.get(c) in (None, (), [])) / n, 4) if n else None
                  for c in rs.feature_names()}
    elig = {h: A.eligible(day_rows, h) for h in A.HORIZONS}
    label_balance = {f"{h}d": {"n": len(v), "positive": sum(y for _, y in v),
                               "positive_rate": round(sum(y for _, y in v) / len(v), 4) if v else None}
                     for h, v in elig.items()}
    ticker_c = Counter(r.ticker for r, _ in elig[5])
    setup_c = Counter(r.setup or "UNKNOWN" for r, _ in elig[5])
    date_c = Counter(r.entry_day.isoformat() for r, _ in elig[5])
    n5 = len(elig[5])
    certified_bench = sum(1 for r in mat5 if r.labels.get("benchmark_return_5d") is not None)
    out = {
        "observation_unit": {"observations": n, "signal_days": len(day_rows)},
        "duplicate_observations": base["duplicate_observations"],
        "same_ticker_same_time_duplicates": sum(c - 1 for c in same_time.values() if c > 1),
        "same_ticker_same_day_repeated_snapshots": sum(c - 1 for c in same_day.values() if c > 1),
        "overlap": base["overlap"],
        "missing_timestamps": base["missing_timestamps"],
        "invalid_timestamps": sum(1 for r in raw_rows if (rd.to_dt(r.get("fired_at")) or _dt.datetime.max.replace(
            tzinfo=_dt.timezone.utc)) > _dt.datetime.now(_dt.timezone.utc)),
        "missing_prices": sum(1 for p in prices if p is None),
        "zero_or_negative_prices": sum(1 for p in prices if p is not None and p <= 0),
        "missing_scores": sum(1 for s in scores if s is None),
        "scores_outside_0_100": sum(1 for s in scores if s is not None and not 0 <= s <= 100),
        "feature_null_rates": null_rates,
        "missing_outcomes": base["missing_outcomes"],
        "matured_5d_observations": len(mat5),
        "missing_benchmark_among_matured_5d": len(mat5) - certified_bench,
        "missing_mfe_mae_among_matured_5d": sum(1 for r in mat5 if r.labels.get("mfe_5d") is None
                                                or r.labels.get("mae_5d") is None),
        "unknown_model_versions": sum(1 for r in records if not r["observation"]["provenance"].get("model_version")),
        "unknown_scoring_versions": base["unknown_scoring_version"],
        "scoring_versions": base["scoring_versions"],
        "temporal_join_status": base["scan_join_status"],
        "join_integrity": A.join_integrity(obs_rows),
        "label_balance_primary_label": label_balance,
        "concentration_matured_5d": {
            "rows": n5, "unique_tickers": len(ticker_c),
            "top_ticker_share": round(ticker_c.most_common(1)[0][1] / n5, 4) if n5 else None,
            "top10_ticker_share": round(sum(v for _, v in ticker_c.most_common(10)) / n5, 4) if n5 else None,
            "setups": dict(setup_c.most_common()),
            "entry_days": len(date_c),
            "max_entry_day_share": round(max(date_c.values()) / n5, 4) if n5 else None,
        },
        "observed_date_distribution": dict(sorted(Counter(r.observed_day for r in obs_rows).items())),
    }
    blockers, warnings = [], []
    ji = out["join_integrity"]
    if ji["negative_lag_violations"] or out["invalid_timestamps"] or out["scores_outside_0_100"] \
            or out["same_ticker_same_time_duplicates"]:
        blockers.append("integrity violation (look-ahead join, invalid timestamp, out-of-range score or duplicate)")
    if n5 < DEFINITIVE_MIN_ROWS:
        blockers.append(f"only {n5} matured, certified signal-days at the primary 5-day horizon "
                        f"(definitive validation needs >= {DEFINITIVE_MIN_ROWS})")
    if len(mat5) and certified_bench == 0:
        blockers.append("benchmark (SPY) coverage is 0% among matured rows, so excess-return metrics are unavailable")
    elif len(mat5) and certified_bench < len(mat5):
        warnings.append(f"benchmark missing on {len(mat5) - certified_bench} of {len(mat5)} matured 5-day rows")
    if out["unknown_model_versions"]:
        warnings.append(f"served model version unknown on all {out['unknown_model_versions']} observations")
    if base["missing_outcomes"]["unavailable"]:
        warnings.append(f"{base['missing_outcomes']['unavailable']} observations have unavailable outcomes "
                        "(mostly the 2026-09-12/13 burst); excluded by maturity, not deleted")
    if out["same_ticker_same_day_repeated_snapshots"]:
        warnings.append(f"{out['same_ticker_same_day_repeated_snapshots']} repeated same-day snapshots collapsed "
                        "to one signal-day each")
    if out["overlap"]["rows_with_overlapping_label_windows"]:
        warnings.append("overlapping label windows across entry days: handled by purge + embargo")
    out["verdict"] = "FAIL" if blockers else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    out["blockers"] = blockers
    out["warnings"] = warnings
    return out


# ----------------------------------------------------------------------------- models
def model_factories(leakage_classes: Mapping[str, str], *, include_ai: bool) -> Dict[str, Any]:
    f = {
        "majority_baseline": A.Majority,
        "random_baseline": A.RandomBaseline,
        "hsf_score_heuristic": lambda: A.ScoreModel("hsf_score_heuristic", "hsf_score"),
        "prebreakout_production_as_served": lambda: A.ScoreModel("prebreakout_production_as_served",
                                                                 "prebreakout_prob"),
        "logistic_regression": A.Logistic,
        "xgb_retrained_production_features": A.production_feature_xgb,
        "xgb_leakage_safe_subset": lambda: A.safe_subset_xgb(leakage_classes),
    }
    if include_ai:
        f["ai_confidence_production_pit"] = lambda: A.ScoreModel("ai_confidence_production_pit",
                                                                 "ai_confidence_raw", source="extra")
    return f


def _compact(res: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k != "predictions"}


# ----------------------------------------------------------------------------- main entry
def run(*, raw_rows: Sequence[Mapping[str, Any]], scan_rows: Sequence[Mapping[str, Any]],
        manifest: Dict[str, Any], members: Optional[Sequence[int]] = None,
        ai_model: Any = None, ai_meta: Optional[Mapping[str, Any]] = None,
        spy_closes: Optional[Mapping[_dt.date, float]] = None,
        legacy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run the full audit. ``raw_rows`` must already be the as-of-freeze view."""
    records = rd.build_records(raw_rows, scan_rows)
    if members is not None:
        keep = {int(m) for m in members}
        records = [r for r in records if r["observation"]["observation_id"] in keep]
    obs_rows = A.build_audit_rows(records, raw_rows, unit="observation")
    day_rows = A.build_audit_rows(records, raw_rows, unit="signal_day")

    # AI Confidence (active model) scored point-in-time: only rows observed after
    # it was trained and with all six of its inputs stored at observation time.
    ai_info: Dict[str, Any] = {"scored_rows": 0}
    if ai_model is not None:
        trained = rd.to_dt((ai_meta or {}).get("trained_at"))
        names = list((ai_meta or {}).get("feature_names") or A.AI_CONFIDENCE_FEATURE_MAP)
        for r in day_rows + obs_rows:
            vals = [r.features.get(A.AI_CONFIDENCE_FEATURE_MAP.get(n, n)) for n in names]
            if trained is not None and r.observed_at > trained and all(v is not None for v in vals):
                x = np.array([[float(v) for v in vals]])
                r.extra["ai_confidence_raw"] = float(ai_model.predict_proba(x)[:, 1][0])
        ai_info = {"scored_rows": sum(1 for r in day_rows if "ai_confidence_raw" in r.extra),
                   "trained_at": (ai_meta or {}).get("trained_at"), "feature_names": names,
                   "rule": "observed_at > trained_at and all six inputs stored at observation time"}

    leak = A.leakage_table(obs_rows)
    leak_cls = {x["feature"]: x["point_in_time_status"] for x in leak}
    quality = data_quality(raw_rows, records, obs_rows, day_rows)
    manifest = dict(manifest)
    manifest["audit_slice_fingerprint"] = A.slice_fingerprint(day_rows)
    make_factories = model_factories(leak_cls, include_ai=ai_info["scored_rows"] > 0)
    purge_cfg = {"rule": "drop training rows whose h-day label window ends on/after the validation start "
                         "(purge), plus an extra embargo of N trading days", "embargo_trading_days":
                 FOLD_KW["embargo"], **{k: v for k, v in FOLD_KW.items() if k != "embargo"}}

    # ---- per-horizon walk-forward with the primary label
    horizons: Dict[int, Dict[str, Any]] = {}
    for h in A.HORIZONS:
        data = A.eligible(day_rows, h)
        hold = A.final_holdout(data, h, **HOLDOUT_KW, **FOLD_KW)
        excl = hold["ids"] if hold.get("valid") else []
        folds = A.walk_forward_folds(data, h, exclude_ids=excl, **FOLD_KW)
        results = {name: A.run_walk_forward(data, folds, fac, h) for name, fac in make_factories.items()}
        horizons[h] = {"data": data, "folds": folds, "holdout": hold, "results": results}

    prim = horizons[PRIMARY_HORIZON]
    pdata, pfolds = prim["data"], prim["folds"]
    by_id = {r.observation_id: r for r, _ in pdata}

    # ---- legacy-validation row for the mandatory table (in-sample chrono split, no purge, all obs)
    legacy_like = None
    obs_data = A.eligible(obs_rows, PRIMARY_HORIZON)
    if len(obs_data) >= 50:
        ordered = sorted(obs_data, key=lambda t: (t[0].observed_at, t[0].observation_id))
        cut = int(len(ordered) * 0.8)
        tr, va = ordered[:cut], ordered[cut:]
        lf = A.Fold(fold=1, horizon=PRIMARY_HORIZON, train_ids=[r.observation_id for r, _ in tr],
                    val_ids=[r.observation_id for r, _ in va], train_start=None, train_end=None,
                    validation_start=va[0][0].entry_day.isoformat(), validation_end=va[-1][0].entry_day.isoformat(),
                    purged=0, embargoed=0, embargo_days=0)
        legacy_like = {name: _compact(A.run_walk_forward(obs_data, [lf], make_factories[name], PRIMARY_HORIZON))
                       for name in ("xgb_retrained_production_features", "xgb_leakage_safe_subset",
                                    "logistic_regression")}

    # ---- label audit (each candidate label, every horizon it supports)
    label_audit = []
    for key, ld in A.LABELS.items():
        for h in ld.horizons:
            data = A.eligible(day_rows, h, key)
            row = {"label": key, "description": ld.description, "horizon": h, "n": len(data),
                   "positive_rate": round(sum(y for _, y in data) / len(data), 4) if data else None}
            if len(data) >= FOLD_KW["min_train"] + FOLD_KW["min_val"]:
                folds = A.walk_forward_folds(data, h, **FOLD_KW)
                row["folds"] = len(folds)
                for name in ("hsf_score_heuristic", "logistic_regression", "xgb_leakage_safe_subset"):
                    res = A.run_walk_forward(data, folds, make_factories[name], h)
                    row[f"{name}_mean_auc"] = res["summary"]["roc_auc"]["mean"]
                    row[f"{name}_pooled_auc"] = (res["pooled"] or {}).get("roc_auc")
            else:
                row["folds"] = 0
                row["note"] = "too few labelled rows for walk-forward" if ld.horizons else ""
            label_audit.append(row)

    # ---- importance + ablation (primary horizon / label)
    importance = A.importance_report(pdata, pfolds, leak_cls, PRIMARY_HORIZON) if pfolds else []
    ablation = []
    if pfolds:
        full = {}
        for mname in ("xgb_leakage_safe_subset", "logistic_regression"):
            full[mname] = prim["results"][mname]
        bad = {f for f, c in leak_cls.items() if c in (A.LEAKAGE, A.REVIEW)}
        experiments = [("production_features (all schema features)", ())] + \
            [(f"minus {g}", (g,)) for g in A.FEATURE_GROUPS] + [("core_technical_only", None)]
        for label, groups in experiments:
            for mname in ("xgb_leakage_safe_subset", "logistic_regression"):
                if groups is None:
                    only = ("chg_pct", "scan_chg_pct", "trend_10d_pct", "trend_20d_pct", "rvol_20", "dollar_vol_20",
                            "volume", "volatility_20d_pct", "gap_pct", "scan_gap_pct", "breakout_pos_20d",
                            "ema_cross", "scanner_breakout_score", "is_breakout")
                    num, cat, sig = A.feature_set(only=only)
                else:
                    num, cat, sig = A.feature_set(drop_groups=groups)
                if mname == "xgb_leakage_safe_subset":
                    num = [c for c in num if c not in bad]
                    cat = [c for c in cat if c not in bad]
                    fac = (lambda num=num, cat=cat, sig=sig: A.XGB(
                        "xgb_ablation", numeric=num, categorical=cat, signals=sig,
                        params={**A.XGB_PRODUCTION_PARAMS, "n_estimators": 200, "max_depth": 3,
                                "min_child_weight": 5}))
                else:
                    fac = (lambda num=num, cat=cat, sig=sig: A.Logistic(numeric=num, categorical=cat, signals=sig))
                res = A.run_walk_forward(pdata, pfolds, fac, PRIMARY_HORIZON)
                pooled = res["pooled"] or {}
                fin = pooled.get("financial_top_tercile") or {}
                base_res = full[mname]
                bp = base_res["pooled"] or {}
                bfin = bp.get("financial_top_tercile") or {}

                def d(a, b):
                    return None if a is None or b is None else round(a - b, 4)

                ablation.append({
                    "experiment": label, "model": mname, "groups_removed": ",".join(groups or ()) if groups
                    else ("all but core technical" if groups is None else ""),
                    "skipped_groups_absent": ",".join(A.MISSING_GROUPS),
                    "mean_auc": res["summary"]["roc_auc"]["mean"], "pooled_auc": pooled.get("roc_auc"),
                    "pooled_pr_auc": pooled.get("pr_auc"), "pooled_brier": pooled.get("brier"),
                    "delta_mean_auc": d(res["summary"]["roc_auc"]["mean"], base_res["summary"]["roc_auc"]["mean"]),
                    "delta_pooled_auc": d(pooled.get("roc_auc"), bp.get("roc_auc")),
                    "delta_pr_auc": d(pooled.get("pr_auc"), bp.get("pr_auc")),
                    "delta_brier": d(pooled.get("brier"), bp.get("brier")),
                    "top_tercile_median_excess": fin.get("median_excess_return"),
                    "delta_median_excess": d(fin.get("median_excess_return"), bfin.get("median_excess_return")),
                    "top_tercile_beat_rate": fin.get("benchmark_beat_rate"),
                    "delta_beat_rate": d(fin.get("benchmark_beat_rate"), bfin.get("benchmark_beat_rate")),
                    "top_tercile_median_mfe": fin.get("median_mfe"),
                    "delta_median_mfe": d(fin.get("median_mfe"), bfin.get("median_mfe")),
                    "top_tercile_median_mae": fin.get("median_mae"),
                    "delta_median_mae": d(fin.get("median_mae"), bfin.get("median_mae")),
                })

    # ---- calibration (offline only)
    calibration = {}
    for name, res in prim["results"].items():
        pr = res["predictions"]
        if not pr["y"]:
            continue
        y, p = np.array(pr["y"]), np.array(pr["p"])
        calibration[name] = {"brier": (res["pooled"] or {}).get("brier"), "ece": A.ece(y, p),
                             "reliability": A.reliability_bins(y, p), "offline": A.offline_calibration(pr)}

    # ---- regimes
    regime = {"available": False}
    if spy_closes:
        days = sorted({r.entry_day for r, _ in pdata})
        reg = A.spy_regime_by_day(spy_closes, days)
        regime = {"available": True, "rule": "SPY 20-session return and annualized volatility from closes "
                                             "strictly before the entry day; bullish > +2%, bearish < -2%, "
                                             "high volatility >= 20%", "by_model": {},
                  "day_labels": {d.isoformat(): v for d, v in reg.items()}}
        for name in ("hsf_score_heuristic", "logistic_regression", "xgb_leakage_safe_subset",
                     "prebreakout_production_as_served"):
            pr = prim["results"][name]["predictions"]
            rows_out = []
            for dim in ("trend", "volatility"):
                values = sorted({v.get(dim) for v in reg.values() if v.get(dim)})
                for val in values:
                    ids = [oid for oid in pr["ids"] if reg.get(by_id[oid].entry_day, {}).get(dim) == val]
                    sub = A.subset_auc(pr, ids)
                    fin = A.financial_metrics([by_id[i] for i in ids], PRIMARY_HORIZON)
                    rows_out.append({"dimension": dim, "regime": val, **(sub or {}),
                                     "win_rate": fin["win_rate"], "median_return": fin["median_return"]})
            regime["by_model"][name] = rows_out

    # ---- concentration
    conc = A.concentration([r for r, _ in pdata])
    top10 = {t["ticker"] for t in conc["top10"]}
    conc["excluding_top10_tickers"] = {}
    for name in ("hsf_score_heuristic", "logistic_regression", "xgb_leakage_safe_subset"):
        pr = prim["results"][name]["predictions"]
        conc["excluding_top10_tickers"][name] = {
            "all": A.subset_auc(pr, pr["ids"]),
            "excluding_top10": A.subset_auc(pr, [i for i in pr["ids"] if by_id[i].ticker not in top10])}

    # ---- temporal stability
    temporal = {}
    for name in ("hsf_score_heuristic", "logistic_regression", "xgb_leakage_safe_subset",
                 "xgb_retrained_production_features", "prebreakout_production_as_served"):
        res = prim["results"][name]
        weeks = A.temporal_buckets(res["predictions"], by_id, PRIMARY_HORIZON)
        temporal[name] = {"weeks": weeks,
                          "folds": [{k: f.get(k) for k in ("fold", "validation_start", "validation_end",
                                                           "validation_n", "roc_auc", "pr_auc", "brier")}
                                    for f in res["folds"]],
                          "verdict_by_fold": A.stability_verdict([f.get("roc_auc") for f in res["folds"]])}

    # ---- path quality (MFE/MAE)
    mat5 = [r for r in day_rows if r.matured(PRIMARY_HORIZON) and r.certified]
    path = {"by_setup": {}, "by_prediction_tercile": {}}
    for s in sorted({r.setup or "UNKNOWN" for r in mat5}):
        grp = [r for r in mat5 if (r.setup or "UNKNOWN") == s]
        path["by_setup"][s] = A.financial_metrics(grp, PRIMARY_HORIZON)
    for name in ("hsf_score_heuristic", "xgb_leakage_safe_subset", "logistic_regression"):
        pr = prim["results"][name]["predictions"]
        if not pr["ids"]:
            continue
        order = np.argsort(np.array(pr["p"]))
        k = len(order)
        terc = {"low": order[: k // 3], "mid": order[k // 3: 2 * k // 3], "high": order[2 * k // 3:]}
        path["by_prediction_tercile"][name] = {
            t: A.financial_metrics([by_id[pr["ids"][i]] for i in idx], PRIMARY_HORIZON) for t, idx in terc.items()}

    # ---- final holdout (once, only if valid)
    holdout_result: Dict[str, Any] = dict(prim["holdout"])
    if prim["holdout"].get("valid"):
        cands = ("hsf_score_heuristic", "logistic_regression", "xgb_retrained_production_features",
                 "xgb_leakage_safe_subset")
        best = max(cands, key=lambda n: prim["results"][n]["summary"]["roc_auc"]["mean"] or -1)
        hstart = _dt.date.fromisoformat(prim["holdout"]["start"])
        rest = [r for r, _ in pdata if r.entry_day < hstart]
        tr, purged, emb = A.purge_train(rest, hstart, PRIMARY_HORIZON, FOLD_KW["embargo"])
        hf = A.Fold(fold=1, horizon=PRIMARY_HORIZON, train_ids=[r.observation_id for r in tr],
                    val_ids=prim["holdout"]["ids"], train_start=None, train_end=None,
                    validation_start=prim["holdout"]["start"], validation_end=max(
                        by_id[i].entry_day for i in prim["holdout"]["ids"]).isoformat(),
                    purged=purged, embargoed=emb, embargo_days=FOLD_KW["embargo"])
        holdout_result["selected_by_walk_forward"] = best
        holdout_result["result"] = _compact(A.run_walk_forward(pdata, [hf], make_factories[best], PRIMARY_HORIZON))

    # ---- experiment records (champion / challenger dry run)
    experiments = []
    for h, hv in horizons.items():
        for name, res in hv["results"].items():
            experiments.append(A.experiment_record(
                experiment_id=f"{A.AUDIT_VERSION}:{name}:A_return_gt_0:{h}d", manifest=manifest,
                model=make_factories[name](), label=A.PRIMARY_LABEL, h=h, folds=hv["folds"], purge=purge_cfg,
                result=_compact(res)))

    preds_for_drift = {n: prim["results"][n]["predictions"] for n in ("hsf_score_heuristic",
                                                                      "xgb_leakage_safe_subset",
                                                                      "logistic_regression")}
    return {
        "manifest": manifest,
        "quality": quality,
        "leakage": leak,
        "single_feature_auc": A.single_feature_auc_scan(pdata),
        "ai_confidence": ai_info,
        "purge_config": purge_cfg,
        "horizons": {h: {"n": len(v["data"]),
                         "positive_rate": round(sum(y for _, y in v["data"]) / len(v["data"]), 4) if v["data"] else None,
                         "folds": [vars(f) | {"train_n": len(f.train_ids), "validation_n": len(f.val_ids),
                                              "train_ids": None, "val_ids": None,
                                              "positive_rate": _pos_rate(v["data"], f.val_ids),
                                              "tickers": len({by(v, i).ticker for i in f.val_ids}),
                                              "setups": dict(Counter(by(v, i).setup or "UNKNOWN"
                                                                     for i in f.val_ids).most_common())}
                                   for f in v["folds"]],
                         "holdout": {k: x for k, x in v["holdout"].items() if k != "ids"},
                         "results": {n: _compact(r) for n, r in v["results"].items()}}
                     for h, v in horizons.items()},
        "legacy_like_split_on_research_data": legacy_like,
        "score_calibration": A.score_calibration(day_rows),
        "label_audit": label_audit,
        "importance": importance,
        "ablation": ablation,
        "calibration": calibration,
        "regime": regime,
        "concentration": conc,
        "temporal": temporal,
        "path_quality": path,
        "holdout": {k: v for k, v in holdout_result.items() if k != "ids"},
        "experiments": experiments,
        "drift_baseline": A.drift_baseline(day_rows, preds_for_drift),
        "legacy": dict(legacy or {}),
        "unsupported_horizons": {f"{h}_bar": "NOT AVAILABLE: no 10/15/20-bar outcome exists for HSF observations "
                                              "(label schema v1 = 1/3/5 trading days); not fabricated"
                                 for h in A.UNSUPPORTED_HORIZONS},
    }


def _pos_rate(data, ids) -> Optional[float]:
    lab = {r.observation_id: y for r, y in data}
    v = [lab[i] for i in ids]
    return round(sum(v) / len(v), 4) if v else None


def by(hv, oid):
    idx = hv.setdefault("_by_id", {r.observation_id: r for r, _ in hv["data"]})
    return idx[oid]
