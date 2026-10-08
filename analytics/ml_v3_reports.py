"""Render ML v3 audit results (``ml_v3_pipeline.run``) to ml/reports/ files.

Every number written here comes from the result dict; nothing is typed in.
Narrative judgement (production model write-up, recommendations, verdict)
lives in hand-written files next to these and cites them.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from analytics import ml_v3_audit as A

PRIMARY = 5

MODEL_LABELS = {
    "majority_baseline": "Majority-class baseline",
    "random_baseline": "Random probability baseline",
    "hsf_score_heuristic": "HSF Score (production heuristic)",
    "prebreakout_production_as_served": "PreBreakout % as served (production XGBoost)",
    "ai_confidence_production_pit": "AI Confidence (production XGBoost, scored point-in-time)",
    "logistic_regression": "Logistic regression (research)",
    "xgb_retrained_production_features": "XGBoost retrained, production features (research)",
    "xgb_leakage_safe_subset": "XGBoost, leakage-safe subset (research)",
}


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v: Any, nd: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def _csv(rows: List[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    buf = io.StringIO()
    cols = list(columns or (list(rows[0].keys()) if rows else []))
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(A.to_jsonable(v), sort_keys=True) if isinstance(v, (dict, list)) else v)
                    for k, v in r.items()})
    return buf.getvalue()


def _ci(d: Optional[Mapping[str, Any]], pct: bool = False) -> str:
    if not d:
        return ""
    lo, hi = d.get("low"), d.get("high")
    return f" [{_pct(lo) if pct else _f(lo)}, {_pct(hi) if pct else _f(hi)}]"


# ----------------------------------------------------------------------------- sections
def manifest_json(res: Mapping[str, Any]) -> str:
    return A.dumps(res["manifest"])


def data_quality_md(res: Mapping[str, Any]) -> str:
    q = res["quality"]
    m = res["manifest"]
    lines = ["# ML v3 data quality gate\n",
             f"Dataset `{m.get('dataset_version')}` · fingerprint `{m.get('dataset_fingerprint')}` · "
             f"audit slice `{m.get('audit_slice_fingerprint')}`\n",
             f"## Verdict: **{q['verdict']}**\n"]
    if q["blockers"]:
        lines.append("Blocks definitive validation (diagnostic analysis continues where safe):\n")
        lines += [f"- {b}" for b in q["blockers"]]
        lines.append("")
    if q["warnings"]:
        lines.append("Warnings:\n")
        lines += [f"- {w}" for w in q["warnings"]]
        lines.append("")
    ji = q["join_integrity"]
    lines.append("## Checks\n")
    rows = [
        ["observations / signal-days", f"{q['observation_unit']['observations']} / {q['observation_unit']['signal_days']}"],
        ["duplicate observations (same ticker + instant)", q["duplicate_observations"]],
        ["same-ticker same-time duplicates", q["same_ticker_same_time_duplicates"]],
        ["same-ticker same-day repeated snapshots", q["same_ticker_same_day_repeated_snapshots"]],
        ["rows in multi-row ticker-day groups", q["overlap"]["rows_in_multi_row_ticker_day_groups"]],
        ["rows with overlapping label windows", q["overlap"]["rows_with_overlapping_label_windows"]],
        ["missing timestamps", q["missing_timestamps"]],
        ["invalid (future) timestamps", q["invalid_timestamps"]],
        ["missing prices (no matched scan)", q["missing_prices"]],
        ["zero/negative prices", q["zero_or_negative_prices"]],
        ["missing scores", q["missing_scores"]],
        ["scores outside 0-100", q["scores_outside_0_100"]],
        ["5d outcome pending / unavailable / invalid",
         f"{q['missing_outcomes']['pending']} / {q['missing_outcomes']['unavailable']} / {q['missing_outcomes']['invalid']}"],
        ["matured 5d observations", q["matured_5d_observations"]],
        ["missing benchmark among matured 5d", q["missing_benchmark_among_matured_5d"]],
        ["missing MFE/MAE among matured 5d", q["missing_mfe_mae_among_matured_5d"]],
        ["unknown model versions", q["unknown_model_versions"]],
        ["unknown scoring versions", q["unknown_scoring_versions"]],
        ["scoring versions", json.dumps(q["scoring_versions"])],
        ["temporal join status", json.dumps(q["temporal_join_status"])],
        ["look-ahead join violations (scan written after observation)", ji["negative_lag_violations"]],
        ["as-of join lag seconds (min/p25/median/p75/p95/max)",
         " / ".join(_f(ji["lag_seconds"][k], 0) for k in ("min", "p25", "median", "p75", "p95", "max"))],
    ]
    lines.append(_table(["CHECK", "RESULT"], rows))
    lines.append("\n## Label balance (primary label: return > 0, matured + certified signal-days)\n")
    lines.append(_table(["HORIZON", "N", "POSITIVE", "RATE"],
                        [[h, v["n"], v["positive"], _pct(v["positive_rate"])]
                         for h, v in q["label_balance_primary_label"].items()]))
    c = q["concentration_matured_5d"]
    lines.append("\n## Concentration (matured 5d signal-days)\n")
    lines.append(_table(["MEASURE", "VALUE"], [
        ["rows", c["rows"]], ["unique tickers", c["unique_tickers"]], ["top ticker share", _pct(c["top_ticker_share"])],
        ["top-10 ticker share", _pct(c["top10_ticker_share"])], ["entry days", c["entry_days"]],
        ["largest single-day share", _pct(c["max_entry_day_share"])], ["setups", json.dumps(c["setups"])]]))
    lines.append("\n## Feature nullness (observation unit)\n")
    lines.append(_table(["FEATURE", "NULL RATE"], [[k, _pct(v)] for k, v in q["feature_null_rates"].items()]))
    lines.append("\n## Observations per UTC day\n")
    lines.append(_table(["DAY", "OBSERVATIONS"], list(q["observed_date_distribution"].items())))
    return "\n".join(lines) + "\n"


def leakage_md(res: Mapping[str, Any]) -> str:
    lines = ["# ML v3 point-in-time and leakage audit\n",
             "Features come only from the canonical FeatureSnapshot (research schema v1). Nothing is rebuilt from "
             "current scanner state. Classes: SAFE_STORED (frozen at observation), SAFE_ASOF_JOIN (backward join "
             "to a scan written before the observation), REVIEW (point-in-time but with a provenance gap), "
             "LEAKAGE (would carry post-observation information), MISSING (never stored).\n"]
    lines.append(_table(["FEATURE", "SOURCE", "SCHEMA", "COVERAGE", "POINT-IN-TIME", "LEAKAGE RISK", "NOTES"],
                        [[x["feature"], x["source"], x["schema_version"] or "-", _pct(x["coverage"]),
                          x["point_in_time_status"], x["leakage_risk"], x["notes"]] for x in res["leakage"]]))
    lines.append("\n## Automated checks\n")
    ji = res["quality"]["join_integrity"]
    lines.append(f"- Scan joins: {ji['matched']} matched, {ji['negative_lag_violations']} written after the "
                 f"observation (must be 0).")
    lines.append("- Outcome-like names in the feature vector: "
                 + (", ".join(x["feature"] for x in res["leakage"] if "outcome pattern" in x["notes"]) or "none") + ".")
    sus = [x for x in res["single_feature_auc"] if x["suspicious"]]
    lines.append("- Single-feature in-sample AUC >= 0.85 (leak symptom): "
                 + (", ".join(f"{x['feature']} {x['auc']}" for x in sus) or "none") + ".")
    lines.append("\n### Single-feature in-sample AUC (primary horizon, primary label)\n")
    lines.append(_table(["FEATURE", "N", "AUC"], [[x["feature"], x["n"], x["auc"]] for x in
                                                   sorted(res["single_feature_auc"], key=lambda x: -abs(x["auc"] - 0.5))]))
    lines.append("\n## Explicit inspection list\n")
    items = [
        ("future price / volume / returns", "Not in the snapshot. Labels live only in OutcomeRecord; FeatureSnapshot "
                                            "refuses outcome-like keys (tests/test_ml_v3_audit.py)."),
        ("MFE / MAE / outcome / maturity fields", "Labels only; never encoded as features."),
        ("future benchmark values", "Benchmark returns are labels; regime labels use SPY closes strictly before "
                                    "the entry day."),
        ("post-observation setup labels", "primary_setup/status/signals are the frozen fire-time values."),
        ("current HSF score / rank", "hsf_score and snapshot_rank are the frozen values at fire time; scanner_rank "
                                     "is the scan's own rank."),
        ("current sector metadata / universe / fundamentals", "Not joined (UNSAFE_CURRENT_VALUE)."),
        ("normalization across future samples", "Medians, scalers and category vocabularies are fitted on the "
                                                "training fold only."),
        ("target-derived variables", "None in research schema v1. The legacy July recipes used one "
                                     "(see ml_v3_walk_forward_summary.md, legacy section)."),
    ]
    lines.append(_table(["ITEM", "FINDING"], items))
    return "\n".join(lines) + "\n"


def temporal_md(res: Mapping[str, Any]) -> str:
    pc = res["purge_config"]
    lines = ["# ML v3 temporal validation design\n",
             "## Algorithm\n",
             "1. Unit: one row per (ticker, UTC observation day), the day's first frozen observation "
             "(Outcome Intelligence `signal_day`). Population: matured AND certified at horizon h.",
             "2. Each row gets `entry_day` (first trading day on/after the UTC fire date) and `window_end[h]` "
             "(the trading day h bars later: the last bar its label reads).",
             f"3. Validation blocks are consecutive entry days holding >= {pc['min_val']} rows and >= "
             f"{pc['min_val_class']} of each class. Blocks are chosen from actual coverage, not calendar quarters.",
             "4. Training rows for a block starting on day V: entry_day < V, and window_end[h] < V (purge), and "
             f"window_end[h] < V minus {pc['embargo_trading_days']} trading day(s) (embargo). Training needs >= "
             f"{pc['min_train']} rows with both classes; otherwise the block start moves forward a day.",
             "5. Expanding window: every later fold trains on all earlier, purged history.",
             "6. Final holdout: the most recent block of >= 100 rows over >= 5 days is reserved first, only if "
             "at least 3 walk-forward folds remain without it. Otherwise no holdout is created.",
             "7. Score inputs (HSF Score, stored model %) get probabilities from a train-fold logistic map.",
             "\nDeterministic tests: `tests/test_ml_v3_audit.py` (chronological order, purge, embargo, overlapping "
             "outcomes, no training label reading a validation-period bar, fold reproducibility).\n"]
    for h, v in res["horizons"].items():
        lines.append(f"## {h}-day horizon: {v['n']} rows, positive rate {_pct(v['positive_rate'])}\n")
        hold = v["holdout"]
        if hold.get("valid"):
            lines.append(f"Final holdout: reserved from {hold['start']} ({hold['rows']} rows).\n")
        else:
            lines.append(f"Final holdout: not created: {hold.get('reason', '')}\n")
        if not v["folds"]:
            lines.append("No fold could be formed with the minimums above.\n")
            continue
        lines.append(_table(["FOLD", "TRAIN START", "TRAIN END", "VAL START", "VAL END", "TRAIN N", "VAL N",
                             "VAL POS RATE", "PURGED", "EMBARGOED", "VAL TICKERS", "VAL SETUPS"],
                            [[f["fold"], f["train_start"], f["train_end"], f["validation_start"], f["validation_end"],
                              f["train_n"], f["validation_n"], _pct(f["positive_rate"]), f["purged"], f["embargoed"],
                              f["tickers"], json.dumps(f["setups"])] for f in v["folds"]]))
        lines.append("")
    reg = res["regime"]
    if reg.get("available"):
        lines.append("Market conditions per entry day (point-in-time SPY regime) are in ml_v3_market_context.md.\n")
    return "\n".join(lines) + "\n"


def _row_for(name: str, validation: str, n: Any, res_pooled: Mapping[str, Any]) -> List[Any]:
    fin = res_pooled.get("financial_top_tercile") or {}
    return [MODEL_LABELS.get(name, name), validation, n, _f(res_pooled.get("roc_auc")) + _ci(res_pooled.get("roc_auc_ci95")),
            _f(res_pooled.get("pr_auc")), _f(res_pooled.get("brier")), _pct(fin.get("win_rate")),
            _pct(fin.get("median_excess_return"), 2), _pct(fin.get("benchmark_beat_rate")),
            _pct(fin.get("median_mfe"), 2), _pct(fin.get("median_mae"), 2)]


def comparison_rows(res: Mapping[str, Any]) -> List[List[Any]]:
    prim = res["horizons"][PRIMARY]
    rows = []
    legacy = res.get("legacy") or {}
    v1 = (legacy.get("v1_recipes") or {}).get("prebreakout_v1") or {}
    reg = legacy.get("registry_legacy_auc") or {}
    if reg:
        rows.append(["PreBreakout v1 (production 2026-06-30), as reported", "random stratified 80/20, own label",
                     reg.get("rows") or "n/a", _f(reg.get("auc")), "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
    if v1:
        rows.append(["PreBreakout v1 recipe, reproduced on current runs", "random stratified 80/20",
                     legacy.get("v1_recipes", {}).get("rows"), _f(v1.get("reproduced_random_split_auc")),
                     "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
        rows.append(["PreBreakout v1 recipe, same rows", "chronological + purge, no snapshot copies",
                     "", _f(v1.get("chronological_purged_no_snapshot_copies_auc")), "n/a", "n/a", "n/a", "n/a",
                     "n/a", "n/a", "n/a"])
    ll = res.get("legacy_like_split_on_research_data") or {}
    for name, r in ll.items():
        rows.append(_row_for(name, "legacy-style: chrono 80/20, all observations, no purge/dedup",
                             (r.get("pooled") or {}).get("n"), r.get("pooled") or {}))
    for name, r in prim["results"].items():
        p = r.get("pooled") or {}
        rows.append(_row_for(name, f"walk-forward, purged, {len([f for f in r['folds'] if 'skipped' not in f])} folds",
                             p.get("n"), p))
    return rows


COMPARISON_HEADERS = ["MODEL", "VALIDATION", "N", "ROC-AUC [95% CI]", "PR-AUC", "BRIER", "WIN RATE",
                      "MEDIAN EXCESS RETURN", "BENCHMARK BEAT RATE", "MFE", "MAE"]


def horizon_rows(res: Mapping[str, Any], model: str) -> List[Dict[str, Any]]:
    out = []
    for h, v in res["horizons"].items():
        r = v["results"].get(model) or {}
        p = r.get("pooled") or {}
        fin = p.get("financial_top_tercile") or {}
        s = r.get("summary", {}).get("roc_auc", {})
        out.append({"horizon": f"{h}d", "model": model, "n": p.get("n"), "folds": s.get("folds"),
                    "roc_auc": p.get("roc_auc"), "roc_auc_ci95": p.get("roc_auc_ci95"),
                    "fold_auc_mean": s.get("mean"), "fold_auc_std": s.get("std"), "fold_auc_worst": s.get("worst"),
                    "pr_auc": p.get("pr_auc"), "precision": p.get("precision"), "recall": p.get("recall"),
                    "f1": p.get("f1"), "brier": p.get("brier"), "win_rate": fin.get("win_rate"),
                    "median_return": fin.get("median_return"), "median_excess_return": fin.get("median_excess_return"),
                    "benchmark_beat_rate": fin.get("benchmark_beat_rate"), "median_mfe": fin.get("median_mfe"),
                    "median_mae": fin.get("median_mae"),
                    "stability": A.stability_verdict([f.get("roc_auc") for f in r.get("folds", [])])})
    for h in A.UNSUPPORTED_HORIZONS:
        out.append({"horizon": f"{h}-bar", "model": model, "n": 0, "stability": "NOT AVAILABLE (no outcome exists)"})
    return out


def walk_forward_summary_md(res: Mapping[str, Any]) -> str:
    m = res["manifest"]
    lines = ["# ML v3 walk-forward summary\n",
             f"Dataset `{m.get('dataset_version')}` (`{m.get('dataset_fingerprint')}`), audit slice "
             f"`{m.get('audit_slice_fingerprint')}`, git `{m.get('git_revision')}`. Primary horizon {PRIMARY} trading "
             "days and primary label return > 0 were fixed before any result was seen (Outcome Intelligence "
             "defaults). Financial columns are Outcome Intelligence metrics on each model's top third of "
             "validation predictions per fold. Benchmark-based columns are n/a when SPY coverage is missing.\n",
             "## Mandatory comparison table (5-day horizon)\n",
             _table(COMPARISON_HEADERS, comparison_rows(res)), ""]
    ai = res.get("ai_confidence") or {}
    lines.append(f"AI Confidence rows scored point-in-time: {ai.get('scored_rows', 0)} "
                 f"({ai.get('rule', 'model unavailable in this run')}).\n")
    lines.append("## Fold-level ROC-AUC per model (5-day)\n")
    prim = res["horizons"][PRIMARY]
    rows = []
    for name, r in prim["results"].items():
        s = r["summary"]["roc_auc"]
        rows.append([MODEL_LABELS.get(name, name), s["folds"], _f(s["mean"]), _f(s["median"]), _f(s["std"]),
                     _f(s["worst"]), _f(s["best"]), r.get("excluded_unscorable_rows")])
    lines.append(_table(["MODEL", "FOLDS", "MEAN", "MEDIAN", "STD", "WORST", "BEST", "ROWS NOT SCORABLE"], rows))
    lines.append("\n## Horizon table (HSF Score heuristic and leakage-safe XGBoost)\n")
    for model in ("hsf_score_heuristic", "xgb_leakage_safe_subset", "logistic_regression"):
        lines.append(f"### {MODEL_LABELS[model]}\n")
        lines.append(_table(["HORIZON", "N", "ROC-AUC", "PR-AUC", "WIN RATE", "MEDIAN RETURN", "MEDIAN EXCESS",
                             "BEAT RATE", "MFE", "MAE", "STABILITY"],
                            [[r["horizon"], r.get("n"), _f(r.get("roc_auc")) + _ci(r.get("roc_auc_ci95")),
                              _f(r.get("pr_auc")), _pct(r.get("win_rate")), _pct(r.get("median_return"), 2),
                              _pct(r.get("median_excess_return"), 2), _pct(r.get("benchmark_beat_rate")),
                              _pct(r.get("median_mfe"), 2), _pct(r.get("median_mae"), 2), r["stability"]]
                             for r in horizon_rows(res, model)]))
        lines.append("")
    lines.append("## Score table (HSF Score buckets, 5-day, all matured signal-days)\n")
    lines.append(_table(["HSF SCORE", "N", "WIN RATE [95% CI]", "MEDIAN RETURN", "MEDIAN EXCESS", "BEAT RATE",
                         "MFE", "MAE"],
                        [[b["bucket"], b["matured_count"], _pct(b["win_rate"]) + _ci(b.get("win_rate_ci95"), True),
                          _pct(b["median_return"], 2), _pct(b["median_excess_return"], 2),
                          _pct(b["benchmark_beat_rate"]), _pct(b["median_mfe"], 2), _pct(b["median_mae"], 2)]
                         for b in res["score_calibration"] if b["horizon"] == PRIMARY]))
    inv = next((b for b in res["score_calibration"] if b["horizon"] == PRIMARY), {})
    lines.append(f"\nMonotonicity (buckets with >= 10 matured): {json.dumps(inv.get('monotonicity'))}. "
                 f"Inversions: {'; '.join(inv.get('inversions') or []) or 'none'}. "
                 f"Spearman(score, 5d return): {json.dumps(inv.get('spearman_score_vs_return'))}.\n")
    hold = res["holdout"]
    lines.append("## Final holdout\n")
    if hold.get("valid"):
        p = (hold.get("result") or {}).get("pooled") or {}
        lines.append(f"Reserved from {hold['start']} ({hold['rows']} rows). Configuration selected by walk-forward "
                     f"mean AUC: {hold['selected_by_walk_forward']}. Holdout ROC-AUC {_f(p.get('roc_auc'))}"
                     f"{_ci(p.get('roc_auc_ci95'))}, PR-AUC {_f(p.get('pr_auc'))}, Brier {_f(p.get('brier'))}.\n")
    else:
        lines.append(f"Not valid: {hold.get('reason')}. No holdout was created.\n")
    legacy = res.get("legacy") or {}
    lines.append("## Legacy metric reproduction\n")
    lines.append("```json\n" + A.dumps({k: v for k, v in legacy.items() if k != "frame_rows"}) + "\n```\n")
    return "\n".join(lines) + "\n"


def baselines_csv(res: Mapping[str, Any]) -> str:
    rows = []
    for h, v in res["horizons"].items():
        for name, r in v["results"].items():
            for f in r["folds"]:
                fin = f.get("financial_top_tercile") or {}
                rows.append({"horizon": h, "model": name, "fold": f["fold"], "validation_start": f["validation_start"],
                             "validation_end": f["validation_end"], "train_n": f["train_n"],
                             "validation_n": f["validation_n"], "skipped": f.get("skipped"),
                             **{k: f.get(k) for k in ("positive_rate", "roc_auc", "pr_auc", "precision", "recall",
                                                      "f1", "accuracy", "log_loss", "brier", "threshold")},
                             "top_tercile_win_rate": fin.get("win_rate"),
                             "top_tercile_median_return": fin.get("median_return"),
                             "top_tercile_median_excess": fin.get("median_excess_return"),
                             "top_tercile_beat_rate": fin.get("benchmark_beat_rate"),
                             "top_tercile_median_mfe": fin.get("median_mfe"),
                             "top_tercile_median_mae": fin.get("median_mae")})
            for stat in ("mean", "median", "std", "worst"):
                rows.append({"horizon": h, "model": name, "fold": stat,
                             **{k: r["summary"][k][stat] for k in r["summary"]}})
            p = r.get("pooled") or {}
            fin = p.get("financial_top_tercile") or {}
            rows.append({"horizon": h, "model": name, "fold": "pooled_oos", "validation_n": p.get("n"),
                         **{k: p.get(k) for k in ("positive_rate", "roc_auc", "pr_auc", "precision", "recall", "f1",
                                                  "accuracy", "log_loss", "brier", "threshold")},
                         "top_tercile_win_rate": fin.get("win_rate"),
                         "top_tercile_median_return": fin.get("median_return"),
                         "top_tercile_median_excess": fin.get("median_excess_return"),
                         "top_tercile_beat_rate": fin.get("benchmark_beat_rate"),
                         "top_tercile_median_mfe": fin.get("median_mfe"),
                         "top_tercile_median_mae": fin.get("median_mae")})
    cols = ["horizon", "model", "fold", "validation_start", "validation_end", "train_n", "validation_n", "skipped",
            "positive_rate", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "log_loss", "brier",
            "threshold", "top_tercile_win_rate", "top_tercile_median_return", "top_tercile_median_excess",
            "top_tercile_beat_rate", "top_tercile_median_mfe", "top_tercile_median_mae"]
    return _csv(rows, cols)


def score_calibration_csv(res: Mapping[str, Any]) -> str:
    cols = ["horizon", "bucket", "sample_size", "matured_count", "win_rate", "win_rate_ci95", "median_return",
            "average_return", "median_benchmark_return", "median_excess_return", "average_excess_return",
            "benchmark_beat_rate", "benchmark_count", "median_mfe", "median_mae", "mfe_count", "evidence_quality",
            "monotonicity", "inversions", "spearman_score_vs_return"]
    return _csv(res["score_calibration"], cols)


def horizon_csv(res: Mapping[str, Any]) -> str:
    rows = []
    for model in res["horizons"][PRIMARY]["results"]:
        rows += horizon_rows(res, model)
    cols = ["horizon", "model", "n", "folds", "roc_auc", "roc_auc_ci95", "fold_auc_mean", "fold_auc_std",
            "fold_auc_worst", "pr_auc", "precision", "recall", "f1", "brier", "win_rate", "median_return",
            "median_excess_return", "benchmark_beat_rate", "median_mfe", "median_mae", "stability"]
    return _csv(rows, cols)


def label_audit_md(res: Mapping[str, Any]) -> str:
    lines = ["# ML v3 label audit\n",
             "## Current production labels\n",
             _table(["SYSTEM", "LABEL", "SOURCE"], [
                 ["HSF Score", "none: a fixed heuristic (signals + max(BreakoutScore, PreBreakout %) + momentum - "
                               "fading), never trained", "ui/opportunities.py score_breakdown"],
                 ["PreBreakout (prebreakout-xgb-v9, active)", "FutureQualitySetupHit: a high-quality setup "
                  "(setup score >= 8) appears within 3 sessions for a below-20d-high candidate",
                  "ml_prebreakout.add_prebreakout_target_label"],
                 ["AI Confidence (ai-confidence-xgb-v1, active since 2026-09-09)", "ForwardReturnHit: +4% before "
                  "-2% within 5 trading days (fallback return_5d >= 4%)", "ml_prebreakout.add_forward_return_labels"],
                 ["Outcome Intelligence (what users see)", "win = return_h > 0; benchmark beat = excess_h > 0",
                  "analytics/outcome_intelligence.metrics"],
             ]),
             "\nNone of the production model labels is what Outcome Intelligence reports to users. PreBreakout "
             "predicts a future scanner state, not a price outcome.\n",
             "## Candidate targets on the frozen research dataset (walk-forward, purged)\n"]
    lines.append(_table(["LABEL", "H", "N", "POS RATE", "FOLDS", "HSF SCORE AUC (mean / pooled)",
                         "LOGISTIC AUC (mean / pooled)", "XGB SAFE AUC (mean / pooled)", "DEFINITION"],
                        [[r["label"], r["horizon"], r["n"], _pct(r["positive_rate"]), r.get("folds"),
                          f"{_f(r.get('hsf_score_heuristic_mean_auc'))} / {_f(r.get('hsf_score_heuristic_pooled_auc'))}",
                          f"{_f(r.get('logistic_regression_mean_auc'))} / {_f(r.get('logistic_regression_pooled_auc'))}",
                          f"{_f(r.get('xgb_leakage_safe_subset_mean_auc'))} / {_f(r.get('xgb_leakage_safe_subset_pooled_auc'))}",
                          r["description"]] for r in res["label_audit"]]))
    lines.append("\nLabels C, D and E need SPY benchmark returns, so they are evaluated only on rows that have one (smaller N). "
                 "No label was chosen using the holdout.\n")
    return "\n".join(lines) + "\n"


def importance_csv(res: Mapping[str, Any]) -> str:
    cols = ["horizon", "feature", "group", "folds", "gain_mean", "permutation_auc_drop_mean",
            "permutation_auc_drop_std", "permutation_positive_fold_share", "shap_mean_abs", "classification",
            "leakage_status"]
    return _csv(res["importance"], cols)


def ablation_csv(res: Mapping[str, Any]) -> str:
    cols = ["experiment", "model", "groups_removed", "skipped_groups_absent", "mean_auc", "pooled_auc",
            "pooled_pr_auc", "pooled_brier", "delta_mean_auc", "delta_pooled_auc", "delta_pr_auc", "delta_brier",
            "top_tercile_median_excess", "delta_median_excess", "top_tercile_beat_rate", "delta_beat_rate",
            "top_tercile_median_mfe", "delta_median_mfe", "top_tercile_median_mae", "delta_median_mae"]
    return _csv(res["ablation"], cols)


def market_context_md(res: Mapping[str, Any]) -> str:
    q = res["quality"]
    nulls = q["feature_null_rates"]
    lines = ["# ML v3 market context readiness and regime performance\n",
             "## Readiness (historically safe availability at observation time)\n",
             _table(["CONTEXT", "STATUS", "EVIDENCE"], [
                 ["SPY trend", "SAFE_TO_DERIVE", "Not stored on observations. Derivable from SPY daily closes "
                                                 "strictly before the entry day (used below for regime labels only)."],
                 ["QQQ trend", "SAFE_TO_DERIVE", "Same as SPY; not stored."],
                 ["market volatility", "SAFE_TO_DERIVE", "SPY realized volatility from prior closes; no VIX stored."],
                 ["breadth", "INSUFFICIENT_HISTORY", "Only derivable from the same scan's candidate rows; never "
                                                      "frozen (REGIME_CAPTURE_UNAVAILABLE)."],
                 ["sector performance", "UNSAFE", "Only today's sector map exists."],
                 ["sector relative strength", "UNSAFE", "Needs the historical sector map."],
                 ["stock-vs-SPY strength", "AVAILABLE", f"rs_vs_spy stored by Run 57+ scans; null on "
                                                        f"{_pct(nulls.get('rs_vs_spy'))} of observations."],
                 ["stock-vs-QQQ strength", "MISSING", "Not computed or stored."],
                 ["stock-vs-sector strength", "UNSAFE", "Needs the historical sector map."],
             ]), ""]
    reg = res["regime"]
    lines.append("## Regime performance (5-day, out-of-sample predictions)\n")
    if not reg.get("available"):
        lines.append("Not run: SPY closes were not available to label regimes point-in-time.\n")
        return "\n".join(lines) + "\n"
    lines.append(f"Rule: {reg['rule']}.\n")
    counts: Dict[str, int] = {}
    for v in reg["day_labels"].values():
        k = f"{v.get('trend')}/{v.get('volatility')}"
        counts[k] = counts.get(k, 0) + 1
    lines.append("Entry days per regime (trend/volatility): " + json.dumps(counts) + "\n")
    for name, rows in reg["by_model"].items():
        lines.append(f"### {MODEL_LABELS.get(name, name)}\n")
        lines.append(_table(["DIMENSION", "REGIME", "N", "ROC-AUC [95% CI]", "WIN RATE", "MEDIAN RETURN"],
                            [[r["dimension"], r["regime"], r.get("n"),
                              _f(r.get("roc_auc")) + _ci(r.get("roc_auc_ci95")), _pct(r.get("win_rate")),
                              _pct(r.get("median_return"), 2)] for r in rows]))
        lines.append("")
    lines.append("Regimes with N < 20 are not evaluable; none is called statistically meaningful.\n")
    return "\n".join(lines) + "\n"


def temporal_stability_csv(res: Mapping[str, Any]) -> str:
    rows = []
    for name, t in res["temporal"].items():
        for f in t["folds"]:
            rows.append({"model": name, "granularity": "fold", "period": f"fold {f['fold']} "
                         f"{f['validation_start']}..{f['validation_end']}", "n": f.get("validation_n"),
                         "roc_auc": f.get("roc_auc"), "pr_auc": f.get("pr_auc"), "brier": f.get("brier"),
                         "verdict": t["verdict_by_fold"]})
        for w in t["weeks"]:
            rows.append({"model": name, "granularity": "week", **w})
    cols = ["model", "granularity", "period", "n", "sufficient", "roc_auc", "pr_auc", "brier", "win_rate",
            "median_excess_return", "benchmark_beat_rate", "verdict"]
    return _csv(rows, cols)


def calibration_csv(res: Mapping[str, Any]) -> str:
    rows = []
    for name, c in res["calibration"].items():
        rows.append({"model": name, "method": "raw (pooled walk-forward)", "bin": "ALL", "brier": c["brier"],
                     "ece": c["ece"]})
        for b in c["reliability"]:
            rows.append({"model": name, "method": "raw (pooled walk-forward)", **b})
        off = c["offline"]
        for method in ("raw", "platt", "isotonic"):
            if method in off:
                rows.append({"model": name, "method": f"{method} (folds 2+, calibrator fit on earlier folds)",
                             "bin": "ALL", "brier": off[method]["brier"], "ece": off[method]["ece"],
                             "n": off.get("evaluated_rows")})
                for b in off[method]["reliability"]:
                    rows.append({"model": name, "method": f"{method} (folds 2+)", **b})
    cols = ["model", "method", "bin", "n", "mean_predicted", "observed_rate", "brier", "ece"]
    return _csv(rows, cols)


def walk_forward_json(res: Mapping[str, Any]) -> str:
    keep = {k: res[k] for k in ("manifest", "purge_config", "horizons", "legacy_like_split_on_research_data",
                                "holdout", "experiments", "concentration", "path_quality", "ai_confidence",
                                "unsupported_horizons", "legacy", "regime")}
    return A.dumps(keep)


FILES = {
    "ml_v3_dataset_manifest.json": manifest_json,
    "ml_v3_data_quality.md": data_quality_md,
    "ml_v3_leakage_audit.md": leakage_md,
    "ml_v3_temporal_validation.md": temporal_md,
    "ml_v3_walk_forward_results.json": walk_forward_json,
    "ml_v3_walk_forward_summary.md": walk_forward_summary_md,
    "ml_v3_model_baselines.csv": baselines_csv,
    "ml_v3_score_calibration.csv": score_calibration_csv,
    "ml_v3_horizon_comparison.csv": horizon_csv,
    "ml_v3_label_audit.md": label_audit_md,
    "ml_v3_feature_importance.csv": importance_csv,
    "ml_v3_feature_ablation.csv": ablation_csv,
    "ml_v3_market_context.md": market_context_md,
    "ml_v3_temporal_stability.csv": temporal_stability_csv,
    "ml_v3_calibration.csv": calibration_csv,
    "ml_v3_drift_baseline.json": lambda res: A.dumps(res["drift_baseline"]),
}


def write_all(res: Mapping[str, Any], outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in FILES.items():
        p = outdir / name
        p.write_text(fn(res), encoding="utf-8")
        written.append(p)
    return written
