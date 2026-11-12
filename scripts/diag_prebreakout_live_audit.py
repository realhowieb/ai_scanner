"""Read-only audit: how good are the PreBreakout % scores users actually saw?

Uses the scores saved in the canonical daily snapshots (what the PreBreakout
tab ranked), on rows old enough for the label to have resolved. Reports hit
rate by per-scan model rank (top 10 = the tab), calibration of the displayed %,
AUC per served model version and per week, and simple heuristic baselines.

Label = the production PreBreakout target (FutureQualitySetupHit): within 1-3
trading days the same symbol becomes a high-quality setup (breakout or
BreakoutScore >= 8) that then hits +4% before -2% within 5 trading days.
Computed for every row (not just training candidates) so rows the tab shows
that the model never trained on are measured too. Nothing is written.
Run via the Diagnostics workflow; the report prints at the end.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_prebreakout as mp  # noqa: E402

DAYS_BACK = int(os.environ.get("AUDIT_DAYS_BACK", "75"))
MATURITY_DAYS = 10


def _auc(y, s):
    # Mann-Whitney AUC (sklearn is not installed in the Diagnostics job).
    y = pd.Series(y).astype(float).reset_index(drop=True)
    s = pd.Series(s).astype(float).reset_index(drop=True)
    ok = y.notna() & s.notna()
    y, s = y[ok], s[ok]
    if len(y) < 50 or y.nunique() < 2:
        return None
    r = s.rank(method="average")
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def _model_version(cell):
    try:
        rec = json.loads(cell) if isinstance(cell, str) else (cell or {})
        pb = rec.get("prebreakout") or {}
        art = pb.get("loaded_artifact") or {}
        return str(art.get("version") or art.get("model_version") or pb.get("trained_at") or "unknown")
    except Exception:
        return "unknown"


def _future_setup_label(labeled: pd.DataFrame) -> pd.Series:
    """Production target logic, for every row (not only candidates)."""
    setup = mp._high_quality_setup_mask(labeled) & (labeled[mp.TARGET_COLUMN].astype(int) == 1)
    ts = pd.to_datetime(labeled["Timestamp"], errors="coerce", utc=True)
    out = pd.Series(0, index=labeled.index)
    for _, idx in labeled.groupby(labeled["Symbol"], sort=False).groups.items():
        idx = list(idx)
        setups = [i for i in idx if bool(setup.loc[i])]
        if not setups:
            continue
        for i in idx:
            start = ts.loc[i]
            if pd.isna(start):
                continue
            for j in setups:
                f = ts.loc[j]
                if pd.isna(f) or f <= start:
                    continue
                lead = int(np.busday_count(start.date(), f.date()))
                if lead > mp.PREBREAKOUT_LEAD_DAYS:
                    break
                if lead >= 1:
                    out.loc[i] = 1
                    break
    return out


def _bucket_table(frame, key, label="y"):
    rows = []
    for name, g in frame.groupby(key, observed=True, sort=True):
        rows.append({
            "bucket": str(name), "n": int(len(g)),
            "hit_pct": round(float(g[label].mean()) * 100, 1),
            "own_4up_first_pct": round(float(g["own_hit"].mean()) * 100, 1),
            "avg_5d_ret_pct": round(float(g["ret5"].mean()) * 100, 2),
            "candidate_share_pct": round(float(g["cand"].mean()) * 100, 1),
            "avg_shown_pct": round(float(g["shown"].mean()), 1) if g["shown"].notna().any() else None,
        })
    return rows


def main() -> None:
    df = mp.load_run_history(days_back=DAYS_BACK, max_runs=20000, canonical_only=True)
    if df.empty:
        print("no snapshot history")
        return
    print(f"[audit] snapshot rows loaded: {len(df)}")
    df["_src"] = "snapshot"
    # Stored near-setup candidates (the PreBreakout tab's population since it moved to them).
    try:
        from db.prebreakout_candidates import load_since

        crow = []
        for scan in load_since(DAYS_BACK):
            for r in scan["rows"]:
                rec = dict(r)
                rec["Symbol"] = str(rec.get("Ticker") or rec.get("Symbol") or "").upper()
                rec["Timestamp"] = rec["run_time"] = pd.Timestamp(scan["scanned_at"])
                rec["_src"] = "candidates"
                rec["_version"] = scan.get("model_version") or "unknown"
                crow.append(rec)
        if crow:
            df = pd.concat([df, pd.DataFrame(crow)], ignore_index=True)
        print(f"[audit] stored candidate rows loaded: {len(crow)}")
    except Exception as e:
        print(f"[audit] stored candidates unavailable: {e}")
    labeled = mp.add_forward_return_labels(df, lookback_days=DAYS_BACK)
    labeled = mp.sort_symbol_history(labeled).reset_index(drop=True)
    labeled["y"] = _future_setup_label(labeled)
    ts = pd.to_datetime(labeled["Timestamp"], utc=True, errors="coerce")
    labeled = labeled.loc[ts <= mp._utc_now() - timedelta(days=MATURITY_DAYS)].copy()
    labeled["_ts"] = pd.to_datetime(labeled["Timestamp"], utc=True, errors="coerce")
    labeled["raw"] = pd.to_numeric(labeled.get("PreBreakoutProbRaw"), errors="coerce")
    labeled["shown"] = pd.to_numeric(labeled.get("PreBreakoutProb%"), errors="coerce")
    labeled["cand"] = mp.prebreakout_candidate_mask(labeled).astype(int)
    labeled["own_hit"] = pd.to_numeric(labeled[mp.TARGET_COLUMN], errors="coerce")
    labeled["ret5"] = pd.to_numeric(labeled[mp.RETURN_COLUMN], errors="coerce")
    labeled["version"] = labeled["ModelProvenance"].map(_model_version) if "ModelProvenance" in labeled else "unknown"
    if "_version" in labeled:
        labeled["version"] = labeled["_version"].where(labeled["_src"] == "candidates", labeled["version"])
    scored_any = labeled.loc[labeled["raw"].notna()].copy()
    scored = scored_any.loc[scored_any["_src"] == "snapshot"].copy()
    scored["run_key"] = scored["run_time"].astype(str)
    scored["rank"] = scored.groupby("run_key")["raw"].rank(ascending=False, method="first")
    scored["scan_size"] = scored.groupby("run_key")["raw"].transform("size")
    scored["rank_bucket"] = pd.cut(scored["rank"], [0, 10, 25, 50, 100, 1e9],
                                   labels=["01-10 (tab)", "11-25", "26-50", "51-100", "101+"])
    scored_any["run_key"] = scored_any["run_time"].astype(str)
    cand = scored_any.loc[scored_any["cand"] == 1].copy()
    cand["rank"] = cand.groupby("run_key")["raw"].rank(ascending=False, method="first")
    stored = cand.loc[cand["_src"] == "candidates"]
    cand["crank"] = cand.groupby("run_key")["raw"].rank(ascending=False, method="first")
    cand["crank_bucket"] = pd.cut(cand["crank"], [0, 10, 25, 50, 100, 1e9],
                                  labels=["01-10", "11-25", "26-50", "51-100", "101+"])

    report = {
        "window": {"days_back": DAYS_BACK, "maturity_days": MATURITY_DAYS,
                   "first": str(scored["_ts"].min()), "last": str(scored["_ts"].max()),
                   "snapshots": int(scored["run_key"].nunique()),
                   "median_rows_per_snapshot": float(scored.groupby("run_key").size().median())},
        "rows_labeled": int(len(labeled)), "rows_scored": int(len(scored)),
        "score_coverage_pct": round(float(labeled["raw"].notna().mean()) * 100, 1),
        "base_rate_all_pct": round(float(scored["y"].mean()) * 100, 2),
        "base_rate_candidates_pct": round(float(cand["y"].mean()) * 100, 2) if len(cand) else None,
        "candidate_share_pct": round(float(scored["cand"].mean()) * 100, 1),
        "auc_all_rows": _auc(scored["y"], scored["raw"]),
        "auc_candidates": _auc(cand["y"], cand["raw"]),
        "auc_own_4up_first_all_rows": _auc(scored["own_hit"], scored["raw"]),
        "by_version": [], "by_week": [],
        "rank_all_rows": _bucket_table(scored, "rank_bucket"),
        "rank_candidates_only": _bucket_table(cand, "crank_bucket") if len(cand) else [],
    }
    for v, g in scored.groupby("version"):
        c = g.loc[g["cand"] == 1]
        report["by_version"].append({"version": v, "n": int(len(g)), "first": str(g["_ts"].min())[:10],
                                     "last": str(g["_ts"].max())[:10], "auc_all": _auc(g["y"], g["raw"]),
                                     "auc_candidates": _auc(c["y"], c["raw"]),
                                     "top10_hit_pct": round(float(g.loc[g["rank"] <= 10, "y"].mean()) * 100, 1)
                                     if (g["rank"] <= 10).any() else None,
                                     "base_pct": round(float(g["y"].mean()) * 100, 1)})
    scored["week"] = scored["_ts"].dt.tz_convert(None).dt.to_period("W").astype(str)
    for w, g in scored.groupby("week"):
        top = g.loc[g["rank"] <= 10]
        report["by_week"].append({"week": w, "n": int(len(g)), "base_pct": round(float(g["y"].mean()) * 100, 1),
                                  "top10_n": int(len(top)),
                                  "top10_hit_pct": round(float(top["y"].mean()) * 100, 1) if len(top) else None,
                                  "auc": _auc(g["y"], g["raw"])})
    # Calibration of the displayed % (candidates = the population it was calibrated on).
    for name, frame in (("calibration_candidates", cand), ("calibration_all_rows", scored)):
        f = frame.loc[frame["shown"].notna()].copy()
        f["shown_bucket"] = pd.cut(f["shown"], [-0.1, 14, 18, 22, 26, 30, 101],
                                   labels=["<=14", "14-18", "18-22", "22-26", "26-30", "30+"])
        report[name] = _bucket_table(f, "shown_bucket")
        if len(f):
            p = f["shown"] / 100.0
            report[name + "_brier"] = round(float(((p - f["y"]) ** 2).mean()), 4)
            report[name + "_brier_base_rate"] = round(float(((f["y"].mean() - f["y"]) ** 2).mean()), 4)
    # Simple heuristic baselines on the same rows (higher = more likely).
    base = {}
    for col, sign in (("BreakoutScore", 1), ("BreakoutPos20D", 1), ("VolRel20", 1), ("Trend10D%", 1),
                      ("Trend20D%", 1), ("HSFScore", 1), ("AIConfidence", 1), ("AI Confidence", 1)):
        if col in scored.columns:
            s = pd.to_numeric(scored[col], errors="coerce") * sign
            if s.notna().mean() > 0.5:
                base[col] = {"auc_all": _auc(scored["y"], s), "auc_candidates": _auc(cand["y"], pd.to_numeric(cand[col], errors="coerce") * sign) if len(cand) else None,
                             "coverage_pct": round(float(s.notna().mean()) * 100, 1)}
    report["heuristic_baselines"] = base
    # Why are shown rows (not) training candidates?
    sc = pd.to_numeric(scored["BreakoutScore"], errors="coerce")
    pos = pd.to_numeric(scored.get("BreakoutPos20D"), errors="coerce")
    report["population"] = {
        "is_breakout_pct": round(float(scored["IsBreakout"].apply(mp._as_bool).mean()) * 100, 1),
        "score_ge_8_pct": round(float((sc >= 8).mean()) * 100, 1),
        "below_20d_high_pct": round(float(mp._price_below_20d_high(scored).fillna(False).mean()) * 100, 1),
        "breakout_score_quantiles": [round(float(x), 2) for x in sc.quantile([0, .1, .25, .5, .75, .9, 1]).tolist()],
        "breakoutpos20d_quantiles": [round(float(x), 3) for x in pos.quantile([0, .1, .25, .5, .75, .9, 1]).tolist()] if pos.notna().any() else None,
    }
    # Same per-scan rank analysis for simple heuristics: top-10 hit rate if the tab ranked by them.
    alt = {}
    for col in ("BreakoutScore", "BreakoutPos20D", "VolRel20", "Trend10D%", "Trend20D%", "RSI14", "RSvsSPY", "Volatility20D%", "GapPct", "PctChange"):
        if col not in scored.columns:
            continue
        for sign, tag in ((1, "high"), (-1, "low")):
            v = pd.to_numeric(scored[col], errors="coerce") * sign
            if v.notna().mean() < 0.5:
                continue
            rk = v.groupby(scored["run_key"]).rank(ascending=False, method="first")
            top = scored.loc[rk <= 10]
            alt[f"{col}_{tag}"] = {"auc": _auc(scored["y"], v), "top10_hit_pct": round(float(top["y"].mean()) * 100, 1),
                                   "top10_own_4up_first_pct": round(float(top["own_hit"].mean()) * 100, 1),
                                   "top10_avg_5d_ret_pct": round(float(top["ret5"].mean()) * 100, 2)}
    report["rank_by_heuristic"] = alt
    report["numeric_columns_seen"] = sorted(c for c in scored.columns if pd.api.types.is_numeric_dtype(scored[c]))[:80]

    top = stored.loc[stored["rank"] <= 10]
    report["stored_candidates"] = {
        "rows": int(len(stored)), "scans": int(stored["run_key"].nunique()) if len(stored) else 0,
        "base_rate_pct": round(float(stored["y"].mean()) * 100, 2) if len(stored) else None,
        "auc": _auc(stored["y"], stored["raw"]) if len(stored) else None,
        "tab_top10_n": int(len(top)),
        "tab_top10_hit_pct": round(float(top["y"].mean()) * 100, 1) if len(top) else None,
        "tab_top10_avg_5d_ret_pct": round(float(top["ret5"].mean()) * 100, 2) if len(top) else None,
    }

    print("=== PREBREAKOUT LIVE AUDIT ===")
    print(json.dumps(report, indent=1, default=str))
    _write_summary(report)


def _write_summary(report: dict) -> None:
    """Headline table for the weekly run's GitHub job summary (no-op elsewhere)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    sc = report.get("stored_candidates") or {}
    tab = (report.get("rank_all_rows") or [{}])[0]
    lines = [
        "## PreBreakout weekly check",
        f"Matured window: {report['window']['first'][:10]} to {report['window']['last'][:10]} "
        f"(rows at least {MATURITY_DAYS} days old).",
        "",
        "| Population | Rows | Top-10 hit | Base rate | AUC |",
        "|---|---|---|---|---|",
        f"| Near-setup candidates (current tab) | {sc.get('rows', 0)} | {sc.get('tab_top10_hit_pct')}% "
        f"| {sc.get('base_rate_pct')}% | {sc.get('auc')} |",
        f"| Scan top 100 (old tab) | {report.get('rows_scored')} | {tab.get('hit_pct')}% "
        f"| {report.get('base_rate_all_pct')}% | {report.get('auc_all_rows')} |",
        "",
        "Hit = becomes a setup within 1-3 trading days that then gains 4% before falling 2% within 5 days.",
    ]
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
