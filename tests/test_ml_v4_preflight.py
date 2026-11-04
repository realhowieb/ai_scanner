"""Preparation must block bad snapshots and validate the actual target, not a proxy label."""
import unittest
from dataclasses import replace

import pytest

try:
    import sklearn  # noqa: F401
except ImportError:
    raise unittest.SkipTest("needs ML libraries")

from analytics import ml_v4_preflight as PF
from analytics import ml_v4_providers as P
from analytics.ml_v4_synthetic import dataset


@pytest.fixture(scope="module")
def loaded():
    return dataset()


def test_valid_snapshot_has_a_human_report_and_reserved_test(loaded):
    report, split, binding = PF.prepare(loaded, ["xgboost"], ["production"], git_sha="abc")
    assert report["ready"], report["blockers"]
    PF.require_ready(report)
    assert report["positive"] and report["negative"]
    assert split["holdout"]["valid"] and report["walk_forward_folds"] >= 3
    assert binding["git_sha"] == "abc" and binding["train_ids_sha256"]
    assert "READY" in PF.markdown(report) and "Reserved final test" in PF.markdown(report)


def test_fingerprint_and_membership_must_match(loaded):
    with pytest.raises(PF.PreparationError, match="fingerprint"):
        PF.verify_dataset({**loaded["entry"], "fingerprint": "changed"}, loaded["built"])
    with pytest.raises(PF.PreparationError, match="membership"):
        PF.verify_dataset({**loaded["entry"], "observation_ids": [1]}, loaded["built"])


def test_healthy_general_readiness_cannot_hide_bad_primary_target(loaded):
    # Leave return_5d and raw-data readiness unchanged; make only F_clean_path one-class.
    changed = {**loaded, "audit_rows": [replace(r, labels={**r.labels, "mfe_5d": 0}) for r in loaded["audit_rows"]]}
    report, _, _ = PF.prepare(changed, ["xgboost"], ["production"], git_sha="abc")
    assert report["readiness_status"] == "READY"
    assert not report["ready"]
    assert any("Actual v4 target" in b for b in report["blockers"])
    with pytest.raises(PF.PreparationError, match="Training blocked"):
        PF.require_ready(report)


def test_missing_holdout_and_code_revision_block_training(loaded, monkeypatch):
    split = P.training_split(loaded["audit_rows"])
    split["holdout"] = {"valid": False, "start": None, "rows": 0, "ids": [], "reason": "too small"}
    monkeypatch.setattr(P, "training_split", lambda *a: split)
    report, _, _ = PF.prepare(loaded, ["xgboost"], ["production"])
    assert not report["ready"]
    assert any("no valid final holdout" in b for b in report["blockers"])
    assert any("code commit" in b for b in report["blockers"])


def test_development_never_reads_labels_or_predictions_from_final_test_period(loaded):
    from analytics import ml_v3_audit as A
    from analytics import ml_v4_evaluation as E

    split = P.training_split(loaded["audit_rows"])
    seen = []

    class Recorder(A.Majority):
        def fit(self, rows, y):
            seen.extend(rows)
            return super().fit(rows, y)

        def predict(self, rows):
            seen.extend(rows)
            return super().predict(rows)

    result = E.evaluate(A.eligible(loaded["audit_rows"], 5, P.PRIMARY_TARGET.label),
                        {"recorder": Recorder}, reserved_holdout=split["holdout"])
    assert seen
    assert not result["holdout"]["evaluated"]
    assert "holdout" not in result["models"]["recorder"]
    assert all(r.window_end[5].isoformat() < split["holdout"]["start"] for r in seen)


def test_training_cli_writes_blockers_without_fitting_or_creating_models(loaded, monkeypatch, tmp_path):
    from unittest.mock import Mock

    from scripts import ml_v4_train as T

    bad = {**loaded, "raw_rows": [{**r, "ai_confidence": None} for r in loaded["raw_rows"]]}
    monkeypatch.setattr(T, "load_audit_rows", lambda requested: bad)
    monkeypatch.setattr(P, "available_providers", lambda: ["xgboost"])
    fit = Mock(side_effect=AssertionError("Blocked training must not fit"))
    monkeypatch.setattr(P, "train_all", fit)
    out = tmp_path / "blocked"
    with pytest.raises(PF.PreparationError, match="ai_confidence_frozen"):
        T.main(["--out", str(out), "--providers", "xgboost", "--feature-sets", "production", "--git-sha", "abc"])
    fit.assert_not_called()
    assert (out / "ml_v4_preflight.md").is_file()
    assert not list(out.glob("*.joblib"))
