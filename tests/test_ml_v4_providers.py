"""ML v4 providers: XGBoost, LightGBM, CatBoost on the shared research pipeline.

No database, no network: rows are built with tests/research_fixtures, like the
ML v3 audit tests. Skipped when the ML v4 libraries aren't installed (the core
smoke job); the ML v4 smoke job installs requirements-ml-v4.txt and runs them.
"""
from __future__ import annotations

import datetime as dt
import random
import unittest
from pathlib import Path

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
from analytics import ml_v4_providers as P
from analytics import research_dataset as rd

SMALL = {"n_estimators": 40}
SMALL_CB = {"iterations": 40}


def _small(provider):
    base = dict(P.DEFAULT_PARAMS[provider])
    base.update(SMALL_CB if provider == "catboost" else SMALL)
    return base


def _trading_days(start: dt.date, n: int):
    out, d = [], start
    while len(out) < n:
        if mc.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _dataset(n_days=40, per_day=10, seed=11, start=dt.date(2026, 6, 1), missing_every=0):
    rng = random.Random(seed)
    rows, scans, oid = [], [], 1
    for d in _trading_days(start, n_days):
        for k in range(per_day):
            t = f"T{rng.randrange(80):02d}"
            score = float(round(rng.uniform(20, 95)))
            r5 = rng.gauss((score - 55) / 800, 0.035)
            fired = f"{d.isoformat()}T{14 + k % 5:02d}:{(k * 7) % 60:02d}:00"
            rows.append(opp(oid, t, fired, score=score, scored=True, r1=r5 / 3, r3=r5 / 2, r5=r5,
                            mfe=max(r5, 0) + rng.uniform(0, 0.03), mae=-rng.uniform(0, 0.03),
                            prob=rng.uniform(5, 40), b1=0.001, b3=0.002, b5=0.003))
            sts = (dt.datetime.fromisoformat(fired) - dt.timedelta(minutes=20)).isoformat()
            if not (missing_every and oid % missing_every == 0):
                scans.append(scan(t, sts, trend20=rng.gauss(0, 5)))
            oid += 1
    return rows, scans


def _rows(**kw):
    raw, scans = _dataset(**kw)
    return A.build_audit_rows(rd.build_records(raw, scans), raw, unit="signal_day")


@pytest.fixture(scope="module")
def rows():
    return _rows()


@pytest.fixture(scope="module")
def split(rows):
    return P.training_split(rows)


# ----------------------------------------------------------------------------- target
def test_primary_target_is_the_production_ai_confidence_event():
    t = P.PRIMARY_TARGET
    assert (t.label, t.horizon) == ("F_clean_path_5d", 5)
    assert "+4% before -2%" in t.description
    lab = {"mfe_5d": 0.05, "mae_5d": -0.01}
    assert A.LABELS[t.label].fn(lab, 5) == 1
    assert A.LABELS[t.label].fn({"mfe_5d": 0.05, "mae_5d": -0.02}, 5) == 0  # touched -2%
    assert A.LABELS[t.label].fn({"mfe_5d": 0.039, "mae_5d": 0.0}, 5) == 0
    assert A.LABELS[t.label].fn({"mfe_5d": None, "mae_5d": -0.01}, 5) is None  # never filled


def test_secondary_targets_cover_configured_horizons():
    keys = {(t.label, t.horizon) for t in P.SECONDARY_TARGETS}
    assert keys == {(lab, h) for lab in ("C_excess_gt_0", "A_return_gt_0") for h in (1, 3, 5)}
    with pytest.raises(ValueError):
        P.target_by_key("F_clean_path_5d", 3)  # path label exists only at 5 days


# ----------------------------------------------------------------------------- split / leakage
def test_training_split_reserves_and_purges_the_holdout(rows, split):
    assert split["holdout"]["valid"], split["holdout"]
    start = dt.date.fromisoformat(split["holdout"]["start"])
    hold = set(split["holdout"]["ids"])
    h = P.PRIMARY_TARGET.horizon
    for r, _ in split["train"]:
        assert r.observation_id not in hold
        assert r.entry_day < start
        assert r.window_end[h] < start  # no training label reads a holdout-period bar
    assert split["purged"] > 0
    assert split["train_rows"] + split["purged"] + split["embargoed"] + split["holdout"]["rows"] <= split["eligible_rows"]


def test_training_split_without_holdout_says_so():
    s = P.training_split(_rows(n_days=8, per_day=6))
    assert not s["holdout"]["valid"] and s["holdout"]["reason"]
    assert s["train_rows"] == s["eligible_rows"]


def test_only_matured_certified_rows_train(rows, split):
    ids = {r.observation_id for r, _ in split["train"]}
    for r in rows:
        if r.observation_id in ids:
            assert r.certified and r.matured(5)


# ----------------------------------------------------------------------------- providers
@pytest.mark.parametrize("provider", P.PROVIDERS)
@pytest.mark.parametrize("feature_set", P.DEFAULT_FEATURE_SETS)
def test_each_provider_trains_and_predicts_probabilities(split, provider, feature_set):
    res = P.train_model(split["train"], provider, feature_set, params=_small(provider))
    p = res["model"].predict([r for r, _ in split["train"]])
    assert p.shape == (split["train_rows"],)
    assert np.all((p >= 0) & (p <= 1))
    rec = res["record"]
    assert rec["provider"] == provider and rec["feature_set"] == feature_set
    assert rec["target"]["label"] == "F_clean_path_5d" and rec["target"]["horizon_days"] == 5
    assert rec["seed"] == P.SEED and rec["training_seconds"] >= 0
    assert rec["library_versions"][provider]
    assert rec["feature_schema_version"].startswith("research-v")
    assert rec["model_version"].startswith(f"ml-v4-{provider}-{feature_set}-f5d-")


@pytest.mark.parametrize("provider", P.PROVIDERS)
def test_training_is_deterministic(split, provider):
    a = P.train_model(split["train"], provider, "leakage_safe", params=_small(provider))
    b = P.train_model(split["train"], provider, "leakage_safe", params=_small(provider))
    rows = [r for r, _ in split["train"]]
    np.testing.assert_array_equal(a["model"].predict(rows), b["model"].predict(rows))
    assert a["record"]["model_version"] == b["record"]["model_version"]


def test_feature_parity_across_providers(split):
    rows = [r for r, _ in split["train"]]
    fitted = [P.train_model(split["train"], p, "leakage_safe", params=_small(p))["model"] for p in P.PROVIDERS]
    cols = [m.columns for m in fitted]
    assert cols[0] == cols[1] == cols[2]
    mats = [m.matrix(rows) for m in fitted]
    np.testing.assert_array_equal(mats[0], mats[1])
    np.testing.assert_array_equal(mats[0], mats[2])
    for col in P.LEAKAGE_REVIEW_FEATURES:
        assert col not in cols[0]


def test_production_feature_set_matches_ai_confidence_inputs():
    fs = P.FEATURE_SETS["production"]
    assert set(fs.numeric) == set(A.AI_CONFIDENCE_FEATURE_MAP.values())
    assert fs.categorical == () and fs.signals is False


@pytest.mark.parametrize("provider", P.PROVIDERS)
def test_missing_values_stay_missing_and_still_score(provider):
    s = P.training_split(_rows(missing_every=3))
    m = P.train_model(s["train"], provider, "leakage_safe", params=_small(provider))["model"]
    rows = [r for r, _ in s["train"]]
    X = m.matrix(rows)
    assert np.isnan(X).any()  # not filled with 0 or a median
    p = m.predict(rows)
    assert np.isfinite(p).all()


def test_invalid_inputs_are_refused(split):
    rows = [r for r, _ in split["train"]][:20]
    with pytest.raises(ValueError):
        P.ProviderModel("sklearn_rf")
    with pytest.raises(ValueError):
        P.ProviderModel("xgboost", "everything")
    with pytest.raises(ValueError):
        P.ProviderModel("xgboost").fit([], [])
    with pytest.raises(ValueError):
        P.ProviderModel("xgboost").fit(rows, [1] * len(rows))  # one class
    with pytest.raises(ValueError):
        P.ProviderModel("xgboost").fit(rows, [0, 1] * (len(rows) // 2) + [0])  # length mismatch
    with pytest.raises(RuntimeError):
        P.ProviderModel("lightgbm").predict(rows)  # not fitted


def test_one_failing_provider_does_not_stop_the_others(split, monkeypatch):
    real = P._library

    def broken(provider):
        if provider == "catboost":
            raise ImportError("catboost missing")
        return real(provider)

    monkeypatch.setattr(P, "_library", broken)
    out = P.train_all(split["train"], feature_sets=("production",))
    by = {o["record"]["provider"]: o for o in out}
    assert by["catboost"]["model"] is None and "ImportError" in by["catboost"]["record"]["error"]
    assert by["xgboost"]["model"] is not None and by["lightgbm"]["model"] is not None


# ----------------------------------------------------------------------------- artifacts
@pytest.mark.parametrize("provider", P.PROVIDERS)
def test_artifact_round_trip_is_exact_and_checksummed(split, provider, tmp_path: Path):
    res = P.train_model(split["train"], provider, "production", params=_small(provider),
                        dataset={"dataset_version": "hsf-ml-test-v1", "dataset_fingerprint": "sha256:abc"})
    man = P.save_artifact(res["model"], res["record"], tmp_path)
    assert man["dataset_version"] == "hsf-ml-test-v1" and man["artifact_sha256"]
    art = tmp_path / man["artifact_file"]
    loaded = P.load_artifact(art, man["artifact_sha256"])
    rows = [r for r, _ in split["train"]]
    np.testing.assert_array_equal(loaded.predict(rows), res["model"].predict(rows))
    with pytest.raises(ValueError):
        P.load_artifact(art, "0" * 64)
    with pytest.raises(ValueError):
        P.load_artifact(art, "")


def test_artifacts_are_never_overwritten(split, tmp_path: Path):
    res = P.train_model(split["train"], "lightgbm", "production", params=_small("lightgbm"))
    man = P.save_artifact(res["model"], res["record"], tmp_path)
    before = (tmp_path / man["artifact_file"]).read_bytes()
    with pytest.raises(FileExistsError):
        P.save_artifact(res["model"], res["record"], tmp_path)
    assert (tmp_path / man["artifact_file"]).read_bytes() == before


def test_non_model_bytes_are_rejected():
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump({"not": "a model"}, buf)
    blob = buf.getvalue()
    with pytest.raises(TypeError):
        P.ProviderModel.from_bytes(blob, expected_sha256=P.sha256(blob))


# ----------------------------------------------------------------------------- production isolation
def test_engine_never_touches_production_scoring_or_the_database():
    import ast

    src = (Path(__file__).resolve().parents[1] / "analytics" / "ml_v4_providers.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for name in imported:
        assert not name.startswith(("db", "scan", "ml_prebreakout", "api", "psycopg")), name
    assert "get_neon_conn" not in src


# ----------------------------------------------------------------------------- dry-run script
def test_training_script_dry_run_writes_artifacts_only(monkeypatch, tmp_path: Path):
    import importlib.util

    import api.research
    from analytics.ml_v4_synthetic import dataset

    loaded = dataset()
    raw, scans = loaded["raw_rows"], loaded["scan_rows"]
    spec = importlib.util.spec_from_file_location(
        "ml_v4_train_script", Path(__file__).resolve().parents[1] / "scripts" / "ml_v4_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entry = loaded["entry"]
    monkeypatch.setattr(mod, "_pick_version", lambda requested: entry)
    monkeypatch.setattr(mod, "_benchmark_computed_at", lambda ids: {})
    monkeypatch.setattr("db.ml_forward_cohorts.assert_development_allowed", lambda ids: None)
    monkeypatch.setattr(api.research, "load_window", lambda a, b, **kw: {"rows": raw, "scans": scans})
    monkeypatch.setattr(P, "DEFAULT_PARAMS", {p: _small(p) for p in P.PROVIDERS})
    out = tmp_path / "run"
    assert mod.main(["--out", str(out), "--git-sha", "test-code"]) == 0
    import json as _json

    summary = _json.loads((out / "ml_v4_training_manifest.json").read_text())
    assert summary["mode"].startswith("dry_run")
    assert summary["dataset"]["dataset_version"] == entry["dataset_version"]
    assert summary["split"]["holdout"]["valid"] and "ids" not in summary["split"]["holdout"]
    assert len(summary["models"]) == 6 and not summary["failures"]
    assert len(list(out.glob("*.joblib"))) == 6
    with pytest.raises(SystemExit):
        mod.main(["--out", str(out)])  # never overwrites a used folder
