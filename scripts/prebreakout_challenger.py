"""Report-only PreBreakout retrain: a fresh challenger against the live champion.

Trains a new XGBoost on matured near-setup candidates since 2026-09-14 (the
model's real population: no breakout, BreakoutScore below 8, below the 20-day
high) and compares it with the live champion on the same later holdout, under
two input pipelines:

  per_scan  - each scan's own rows only (no price-bar or SPY/QQQ downloads)
  contract  - the live contract: earlier snapshots + daily bars + SPY/QQQ

Chronological split with an 8-business-day purge (3-day setup lead + 5-day
outcome). Prints AUC, top-decile and per-scan top-10 hit rates, and a
recommendation. Saves nothing and never changes the production model; a
promotion is a separate, reviewed step.
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

import pandas as pd  # noqa: E402

import ml_prebreakout as mp  # noqa: E402

START = pd.Timestamp(os.environ.get("CHALLENGER_START", "2026-09-14"), tz="UTC")
MATURITY_DAYS = 10
HOLDOUT_FRACTION = float(os.environ.get("CHALLENGER_HOLDOUT", "0.3"))
PURGE_BDAYS = mp.PREBREAKOUT_LEAD_DAYS + mp.RETURN_HORIZON_DAYS
MIN_HOLDOUT_ROWS, MIN_HOLDOUT_POS, MIN_AUC_GAIN = 300, 30, 0.02
MIN_TRAIN_ROWS, MIN_TRAIN_POS = 500, 50


def _auc(y, s):
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    s = pd.Series(s, dtype=float).reset_index(drop=True)
    ok = y.notna() & s.notna()
    y, s = y[ok], s[ok]
    if len(y) < 50 or y.nunique() < 2:
        return None
    r = s.rank(method="average")
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def _metrics(frame: pd.DataFrame, score_col: str) -> dict:
    s = pd.to_numeric(frame[score_col], errors="coerce")
    y = frame["y"].astype(int)
    out = {"n": int(len(frame)), "positives": int(y.sum()), "auc": _auc(y, s)}
    if len(frame):
        cut = s.quantile(0.9)
        top = frame.loc[s >= cut]
        out["top_decile_hit_pct"] = round(float(top["y"].mean()) * 100, 1)
        rk = s.groupby(frame["run_key"]).rank(ascending=False, method="first")
        t10 = frame.loc[rk <= 10]
        out["per_scan_top10_hit_pct"] = round(float(t10["y"].mean()) * 100, 1) if len(t10) else None
        out["per_scan_top10_avg_5d_ret_pct"] = (round(float(pd.to_numeric(t10[mp.RETURN_COLUMN], errors="coerce").mean()) * 100, 2)
                                                if len(t10) and mp.RETURN_COLUMN in t10 else None)
    return out


def _load_history(days_back: int) -> pd.DataFrame:
    df = mp.load_run_history(days_back=days_back, max_runs=20000, canonical_only=True)
    df["_src"] = "snapshot"
    try:
        from db.prebreakout_candidates import load_since

        rows = []
        for scan in load_since(days_back):
            for r in scan["rows"]:
                rec = {k: v for k, v in r.items() if not str(k).startswith("PreBreakoutProb")}
                rec["Symbol"] = str(rec.get("Ticker") or "").upper()
                rec["Timestamp"] = rec["run_time"] = pd.Timestamp(scan["scanned_at"])
                rec["_src"] = "candidates"
                rows.append(rec)
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        print(f"[challenger] stored candidate rows: {len(rows)}")
    except Exception as e:
        print(f"[challenger] stored candidates unavailable: {e}")
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.loc[ts >= START].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    return df.drop_duplicates(["Symbol", "Timestamp"], keep="last").reset_index(drop=True)


def main() -> None:
    mp._load_ml_libs()
    days_back = int((mp._utc_now() - START.to_pydatetime()).days) + 5
    df = _load_history(days_back)
    print(f"[challenger] history rows since {START.date()}: {len(df)}")
    labeled = mp.add_prebreakout_target_label(df, lookback_days=days_back)
    labeled = labeled.rename(columns={mp.PREBREAKOUT_TARGET_COLUMN: "y"})
    ts = pd.to_datetime(labeled["Timestamp"], utc=True, errors="coerce")
    labeled = labeled.loc[ts <= mp._utc_now() - timedelta(days=MATURITY_DAYS)].copy()
    labeled["_ts"] = pd.to_datetime(labeled["Timestamp"], utc=True)
    labeled = labeled.sort_values("_ts").reset_index(drop=True)
    labeled["run_key"] = labeled["run_time"].astype(str)
    print(f"[challenger] matured candidates: {len(labeled)}, positives: {int(labeled['y'].sum())}")
    if len(labeled) < 200 or labeled["y"].nunique() < 2:
        print(json.dumps({"skipped": "not enough matured candidates", "rows": int(len(labeled))}))
        return

    # Chronological split with a purge gap so no training label overlaps the holdout.
    split_ts = labeled["_ts"].iloc[int(len(labeled) * (1 - HOLDOUT_FRACTION))]
    purge_start = (split_ts - pd.offsets.BDay(PURGE_BDAYS))
    train = labeled.loc[labeled["_ts"] < purge_start].index
    hold = labeled.loc[labeled["_ts"] >= split_ts].index

    bundle = mp.load_prebreakout_model()
    if not bundle or bundle.get("model") is None:
        raise SystemExit("no live champion")
    feats = list(bundle["features"])

    # Pipeline A: each scan's rows alone.
    per_scan = []
    for _, idx in labeled.groupby("run_key", sort=False).groups.items():
        per_scan.append(mp.add_prebreakout_features(labeled.loc[idx].copy()))
    X_scan = pd.concat(per_scan).reindex(labeled.index)
    # Pipeline B: the live contract (prior snapshots + bars + SPY/QQQ).
    bench = mp.load_benchmark_regime_context(days_back)
    full = mp.add_historical_ohlcv_context(df.copy(), days_back=days_back)
    replay = mp.snapshot_feature_frame(full, benchmark_context=bench)
    keys = ["Symbol", "Timestamp"]
    left = labeled[keys].copy()
    left["Timestamp"] = mp._asof_timestamps(left["Timestamp"])
    replay["Timestamp"] = mp._asof_timestamps(replay["Timestamp"])
    X_contract = left.merge(replay, on=keys, how="left", validate="one_to_one")
    X_contract.index = labeled.index

    def model_matrix(X):
        X = X.copy()
        for c in feats:
            if c not in X.columns:
                X[c] = 0.0
        X, cols = mp._apply_preprocessing_plan(X, feats, bundle.get("preprocessing"))
        return X[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    report = {"champion": bundle.get("model_version"), "champion_trained_at": str(bundle.get("trained_at")),
              "window_start": str(START.date()), "rows": int(len(labeled)),
              "base_rate_pct": round(float(labeled["y"].mean()) * 100, 2),
              "train_rows": int(len(train)), "holdout_rows": int(len(hold)),
              "holdout_from": str(split_ts)[:10], "purge_bdays": PURGE_BDAYS, "results": {}}
    best = None
    for name, X in (("per_scan", X_scan), ("contract", X_contract)):
        M = model_matrix(X)
        scored = labeled.loc[hold].copy()
        scored["champion"] = bundle["model"].predict_proba(M.loc[hold])[:, 1]
        report["results"][name] = {"champion": _metrics(scored, "champion")}
        y_train = labeled.loc[train, "y"].astype(int)
        if len(train) < MIN_TRAIN_ROWS or y_train.sum() < MIN_TRAIN_POS or y_train.nunique() < 2:
            report["results"][name]["challenger"] = {
                "skipped": f"only {len(train)} training rows / {int(y_train.sum())} hits before the holdout "
                           f"(need {MIN_TRAIN_ROWS} / {MIN_TRAIN_POS})"}
        else:
            challenger = mp._new_prebreakout_classifier()
            challenger.fit(M.loc[train], y_train)
            scored["challenger"] = challenger.predict_proba(M.loc[hold])[:, 1]
            report["results"][name]["challenger"] = _metrics(scored, "challenger")
            imp = sorted(zip(feats, challenger.feature_importances_), key=lambda kv: -kv[1])[:10]
            report["results"][name]["challenger_top_features"] = [[k, round(float(v), 4)] for k, v in imp]
        for who in ("champion", "challenger"):
            auc = report["results"][name][who].get("auc")
            if auc is not None and (best is None or auc > best[2]):
                best = (who, name, auc)

    champ_live = report["results"]["contract"]["champion"]["auc"]
    hold_y = labeled.loc[hold, "y"]
    enough = len(hold) >= MIN_HOLDOUT_ROWS and int(hold_y.sum()) >= MIN_HOLDOUT_POS
    if best and best[0] == "challenger" and champ_live is not None and best[2] - champ_live >= MIN_AUC_GAIN and enough:
        decision = f"PROMOTION_CANDIDATE: challenger on {best[1]} inputs (AUC {best[2]} vs live {champ_live})"
    elif best and best[0] == "champion" and best[1] == "per_scan" and champ_live is not None \
            and best[2] - champ_live >= MIN_AUC_GAIN and enough:
        decision = f"SWITCH_INPUTS: keep the champion, score it on per_scan inputs (AUC {best[2]} vs {champ_live})"
    else:
        decision = "KEEP: no challenger clears the bar (+0.02 AUC on a holdout of 300+ rows, 30+ hits)"
    report["holdout_large_enough"] = enough
    report["decision"] = decision
    print("=== PREBREAKOUT CHALLENGER REPORT ===")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
