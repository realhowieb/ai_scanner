"""ML v4 walk-forward evaluation and calibration (Milestone 3).

No database, no network: rows come from tests/research_fixtures through the
same builder the production run uses.
"""
from __future__ import annotations

import datetime as dt
import random
import unittest

try:
    import catboost  # noqa: F401
    import lightgbm  # noqa: F401
    import pytest
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401
except ImportError:
    raise unittest.SkipTest("needs pytest and requirements-ml-v4.txt")

import numpy as np
from research_fixtures import opp, scan

from analytics import market_calendar as mc
from analytics import ml_v3_audit as A
from analytics import ml_v4_calibration as C
from analytics import ml_v4_evaluation as E
from analytics import ml_v4_providers as P
from analytics import research_dataset as rd

SMALL = {"xgboost": {**P.DEFAULT_PARAMS["xgboost"], "n_estimators": 30},
         "lightgbm": {**P.DEFAULT_PARAMS["lightgbm"], "n_estimators": 30},
         "catboost": {**P.DEFAULT_PARAMS["catboost"], "iterations": 30}}
FOLD_KW = dict(min_train=40, min_val=25, min_val_class=5, embargo=1)
HOLD_KW = dict(min_rows=60, min_days=5, min_remaining_folds=2)


def _trading_days(start, n):
    out, d = [], start
    while len(out) < n:
        if mc.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _dataset(n_days=45, per_day=10, seed=5):
    rng = random.Random(seed)
    rows, scans, oid = [], [], 1
    for d in _trading_days(dt.date(2026, 5, 4), n_days):
        for k in range(per_day):
            t = f"T{rng.randrange(90):02d}"
            score = float(round(rng.uniform(20, 95)))
            r5 = rng.gauss((score - 55) / 600, 0.035)
            fired = f"{d.isoformat()}T{14 + k % 5:02d}:{(k * 7) % 60:02d}:00"
            rows.append(opp(oid, t, fired, score=score, scored=True, r1=r5 / 3, r3=r5 / 2, r5=r5,
                            mfe=max(r5, 0) + rng.uniform(0, 0.03), mae=-rng.uniform(0, 0.03),
                            prob=rng.uniform(5, 40), b1=0.001, b3=0.002, b5=0.003))
            sts = (dt.datetime.fromisoformat(fired) - dt.timedelta(minutes=20)).isoformat()
            scans.append(scan(t, sts, trend20=rng.gauss(0, 5)))
            oid += 1
    return rows, scans


@pytest.fixture(scope="module")
def rows():
    raw, scans = _dataset()
    return A.build_audit_rows(rd.build_records(raw, scans), raw, unit="signal_day")


@pytest.fixture(scope="module")
def result(rows):
    data = A.eligible(rows, 5, P.PRIMARY_TARGET.label)
    models = E.default_models(feature_sets=("production",), params=SMALL)
    return E.evaluate(data, models, fold_kw=FOLD_KW, holdout_kw=HOLD_KW)


# ----------------------------------------------------------------------------- calibration
def test_sigmoid_and_isotonic_fit_and_apply():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(size=400) < p ** 2).astype(int)  # raw p is over-confident
    for fit in (C.fit_sigmoid, C.fit_isotonic):
        cal = fit(y, p)
        q = C.apply(cal, p)
        assert np.all((q >= 0) & (q <= 1))
        assert C.brier(y, q) <= C.brier(y, p) + 1e-9
    iso = C.fit_isotonic(y, p)
    q = C.apply(iso, np.sort(p))
    assert np.all(np.diff(q) >= -1e-12)  # monotonic: never changes ranking


def test_calibrator_refuses_tiny_or_one_class_data():
    assert C.fit_sigmoid([1] * 100, [0.5] * 100) is None
    assert C.fit_isotonic([0, 1] * 10, [0.3, 0.6] * 10) is None
    assert C.choose_and_fit([]) is None


def test_method_is_chosen_on_the_latest_earlier_fold_not_the_target_fold():
    rng = np.random.default_rng(2)
    chunks = []
    for _ in range(3):
        p = rng.uniform(0.05, 0.95, 150)
        chunks.append(((rng.uniform(size=150) < p).astype(int).tolist(), p.tolist()))
    cal = C.choose_and_fit(chunks)
    assert cal["selection"]["rule"].startswith("brier on the latest earlier fold")
    assert cal["selection"]["selection_rows"] == 150
    assert cal["fitted_on_folds"] == 3 and cal["n"] == 450
    one = C.choose_and_fit(chunks[:1])
    assert one["selection"]["rule"].startswith("size rule") and one["method"] == "sigmoid"


def test_reliability_and_ece():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    assert C.ece(y, p) == pytest.approx(0.05)
    rel = C.reliability(y, p)
    assert sum(b["n"] for b in rel) == 4 and rel[0]["hit_rate"] == 0.0 and rel[9]["hit_rate"] == 1.0


# ----------------------------------------------------------------------------- engine
def test_every_model_is_scored_on_identical_rows(result):
    n = {m["walk_forward_raw"]["n"] for m in result["models"].values()}
    assert len(n) == 1 and n.pop() > 0
    assert result["sample"]["common_rows"] == result["sample"]["eligible_rows"]


def test_holdout_is_disjoint_from_folds_and_evaluated_once(rows, result):
    assert result["holdout"]["valid"]
    start = dt.date.fromisoformat(result["holdout"]["start"])
    for f in result["folds"]:
        assert dt.date.fromisoformat(f["validation_end"]) < start
        assert dt.date.fromisoformat(f["train_end"]) < start
    for m in result["models"].values():
        assert m["holdout"]["raw"]["n"] == result["holdout"]["rows"]


def test_folds_are_chronological_and_purged(result):
    prev = None
    for f in result["folds"]:
        assert f["train_end"] < f["validation_start"]
        if prev:
            assert f["validation_start"] > prev
        prev = f["validation_end"]
    assert sum(f["purged"] for f in result["folds"]) > 0


def test_first_fold_is_never_calibrated_and_later_folds_use_only_earlier_data(result):
    m = result["models"]["xgboost_production"]
    per = m["calibration_per_fold"]
    assert per[0]["method"] is None  # nothing earlier to fit on
    methods = [c["method"] for c in per]
    first = next(i for i, mth in enumerate(methods) if mth)  # once enough earlier rows exist...
    assert all(methods[first:])  # ...every later fold is calibrated
    assert m["walk_forward_calibrated"]["rows_with_calibration"] < m["walk_forward_raw"]["n"]


def test_served_probability_is_never_recalibrated(rows):
    data = A.eligible(rows, 5, P.PRIMARY_TARGET.label)
    served = {r.observation_id: 40.0 for r, _ in data[::2]}
    rows2 = [r for r, _ in data]
    assert E.attach_served(rows2, served) == len(served)
    models = {"ai_confidence_as_served": lambda: E.ServedProbability("ai_confidence_as_served", "ai_confidence_served"),
              "xgboost_production": lambda: P.ProviderModel("xgboost", "production", SMALL["xgboost"])}
    res = E.evaluate(data, models, fold_kw=dict(min_train=20, min_val=20, min_val_class=3, embargo=1),
                     holdout_kw=dict(min_rows=40, min_days=5, min_remaining_folds=1))
    assert res["sample"]["common_rows"] == len(served)  # only rows the champion can score
    champ = res["models"]["ai_confidence_as_served"]
    assert all(c["method"] is None for c in champ["calibration_per_fold"])
    assert "walk_forward_calibrated" not in champ
    assert res["comparison_reference"] == "ai_confidence_as_served"
    assert "xgboost_production" in res["comparisons"]
    assert any(w.startswith("COMMON_SAMPLE") for w in res["warnings"])
    for r in rows2:
        r.extra.pop("ai_confidence_served", None)


def test_metrics_and_breakdowns_are_reported(result):
    m = result["models"]["lightgbm_production"]
    raw = m["walk_forward_raw"]
    for k in ("roc_auc", "pr_auc", "precision", "recall", "brier", "log_loss", "ece", "reliability", "roc_auc_ci95"):
        assert k in raw
    assert set(m["precision_at_top"]) == {"top_10pct", "top_20pct", "top_33pct"}
    ret = m["returns_after_costs"]["top_33pct"]
    assert ret["mean_net"] == pytest.approx(ret["mean_gross"] - E.DEFAULT_COST_BPS / 10000, abs=1e-4)
    assert m["by_setup"] and all("n" in g for g in m["by_setup"])
    assert result["comparison_reference"] == "xgboost_production"


def test_regime_breakdown_uses_supplied_prior_day_labels(rows):
    data = A.eligible(rows, 5, P.PRIMARY_TARGET.label)
    regime = {r.entry_day: {"trend": "bullish" if r.entry_day.day % 2 else "bearish"} for r, _ in data}
    res = E.evaluate(data, {"hsf_score": lambda: A.ScoreModel("hsf_score", "hsf_score")},
                     fold_kw=FOLD_KW, holdout_kw=HOLD_KW, regime_by_day=regime)
    groups = {g["group"] for g in res["models"]["hsf_score"]["by_regime"]}
    assert groups <= {"bullish", "bearish"} and groups


def test_evaluation_is_deterministic(rows):
    data = A.eligible(rows, 5, P.PRIMARY_TARGET.label)
    models = {"lightgbm_production": lambda: P.ProviderModel("lightgbm", "production", SMALL["lightgbm"])}
    a = E.evaluate(data, models, fold_kw=FOLD_KW, holdout_kw=HOLD_KW)
    b = E.evaluate(data, models, fold_kw=FOLD_KW, holdout_kw=HOLD_KW)
    assert A.dumps(a) == A.dumps(b)


def test_paired_delta_is_zero_for_identical_predictions():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    days = [f"d{i % 20}" for i in range(200)]
    d = E.paired_delta_auc(y, p, p, days)
    assert d["delta"] == 0 and d["ci95"]["low"] == 0 and d["ci95"]["high"] == 0


def test_insufficient_data_warnings(rows):
    data = A.eligible(rows, 5, P.PRIMARY_TARGET.label)[:150]
    res = E.evaluate(data, {"hsf_score": lambda: A.ScoreModel("hsf_score", "hsf_score")},
                     fold_kw=FOLD_KW, holdout_kw=HOLD_KW)
    assert any(w.startswith("INSUFFICIENT_DATA") for w in res["warnings"])
    small = E.evaluate([], {"majority_baseline": A.Majority})
    assert any(w.startswith("NO_HOLDOUT") for w in small["warnings"])


def test_targets_by_horizon(rows):
    out = E.evaluate_targets(rows, [P.target_by_key("A_return_gt_0", h) for h in (1, 3, 5)],
                             {"hsf_score": lambda: A.ScoreModel("hsf_score", "hsf_score")},
                             fold_kw=FOLD_KW, holdout_kw=HOLD_KW)
    assert [o["horizon_days"] for o in out] == [1, 3, 5]
    assert all(o["rows"] > 0 and "hsf_score" in o["models"] for o in out)


def test_markdown_report_states_target_and_caveats(result):
    md = E.render_markdown(result)
    assert "F_clean_path_5d" in md and "not evidence of profitability" in md
    assert "xgboost_production" in md and "Final holdout" in md


# ----------------------------------------------------------------------------- read-only script
def test_evaluate_script_writes_reports_only(monkeypatch, tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    raw, scans = _dataset()
    audit_rows = A.build_audit_rows(rd.build_records(raw, scans), raw, unit="signal_day")
    spec = importlib.util.spec_from_file_location(
        "ml_v4_evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "ml_v4_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entry = {"dataset_version": "hsf-ml-test-v1", "fingerprint": "sha256:test"}
    monkeypatch.setattr(mod.T, "load_audit_rows", lambda v: {"entry": entry, "members": [1], "frozen_at": None,
                                                            "built": {}, "audit_rows": audit_rows})
    monkeypatch.setattr(mod, "_served_ai_confidence",
                        lambda ids: {i: 35.0 + (i % 7) for i in list(ids)[len(list(ids)) // 3:]})
    monkeypatch.setattr(mod, "_spy_closes", lambda: {})
    monkeypatch.setattr(P, "DEFAULT_PARAMS", SMALL)
    monkeypatch.setattr(E, "FOLD_KW", FOLD_KW)
    monkeypatch.setattr(E, "HOLDOUT_KW", HOLD_KW)
    monkeypatch.setattr(E, "BOOT", 200)
    out = tmp_path / "eval"
    assert mod.main(["--out", str(out), "--feature-sets", "production"]) == 0
    rep = json.loads((out / "ml_v4_evaluation.json").read_text())
    assert rep["mode"].startswith("read-only")
    assert rep["head_to_head_ai_confidence"]["comparison_reference"] == "ai_confidence_as_served"
    assert len(rep["secondary_targets"]) == 6
    md = (out / "ml_v4_evaluation.md").read_text()
    assert "head-to-head with AI Confidence" in md and "Secondary targets" in md
    with pytest.raises(SystemExit):
        mod.main(["--out", str(out)])
