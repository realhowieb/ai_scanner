"""The freeze tool must write a version the ML v4 trainer reproduces, and nothing when a check fails."""
import datetime as dt
import unittest

try:
    import pytest
except ImportError:
    raise unittest.SkipTest("needs pytest")

from analytics import ml_v3_pipeline as V3
from analytics import research_dataset as rd
from analytics.ml_v4_preflight import verify_dataset
from analytics.ml_v4_synthetic import dataset
from scripts import ml_freeze_dataset as F


@pytest.fixture()
def world(monkeypatch):
    syn = dataset()
    frozen = syn["frozen_at"]
    state = {"rows": [dict(r) for r in syn["raw_rows"]], "scans": syn["scan_rows"], "now": frozen,
             "saved": []}
    monkeypatch.setattr(F, "DEFAULT_START", dt.date(2026, 2, 2))
    monkeypatch.setattr(F, "_now", lambda: state["now"])
    monkeypatch.setattr(F, "load_rows", lambda: ([dict(r) for r in state["rows"]], state["scans"]))

    import db.research_datasets as DB

    def save(entry):
        state["saved"].append({**entry, "created_at": state["now"]})
        return True

    monkeypatch.setattr(DB, "list_dataset_versions", lambda: [])
    monkeypatch.setattr(DB, "save_dataset_version", save)

    def trainer_reload(name):
        # Mirrors scripts/ml_v4_train.load_audit_rows on the saved row.
        entry = state["saved"][-1]
        rows_asof = V3.as_of(state["rows"], entry["created_at"])
        built = rd.build_dataset(rows_asof, state["scans"], filters=entry["metadata"]["filters"],
                                 members=entry["observation_ids"])
        verify_dataset(entry, built)
        return {"frozen_at": entry["created_at"], "audit_rows": built["records"]}

    import scripts.ml_v4_train as T

    monkeypatch.setattr(T, "load_audit_rows", trainer_reload)
    return state


def test_preview_writes_nothing(world):
    assert F.main([]) == 0
    assert world["saved"] == []


def test_freeze_writes_one_row_that_the_trainer_reproduces_later(world):
    assert F.main(["--freeze", "FREEZE"]) == 0
    assert len(world["saved"]) == 1
    entry = world["saved"][0]
    # Later: the outcome job fills a member that was still pending at the freeze.
    i = next(i for i, r in enumerate(world["rows"]) if r.get("outcome_computed_at") is None
             and int(r["id"]) in set(entry["observation_ids"]))
    world["rows"][i] = {**world["rows"][i], "return_1d": .01, "return_3d": .02, "return_5d": .03,
                        "mfe_5d": .05, "mae_5d": -.01,
                        "outcome_computed_at": entry["created_at"] + dt.timedelta(days=2)}
    assert _rebuild(world, entry)["metadata"]["fingerprint"] == entry["fingerprint"]


def test_rescoring_a_frozen_member_breaks_reproducibility(world):
    # The Oct 8 dataset broke this way: rows re-scored after its freeze.
    assert F.main(["--freeze", "FREEZE"]) == 0
    entry = world["saved"][0]
    i = next(i for i, r in enumerate(world["rows"]) if r.get("outcome_computed_at") is not None)
    world["rows"][i] = {**world["rows"][i], "outcome_computed_at": entry["created_at"] + dt.timedelta(days=2)}
    assert _rebuild(world, entry)["metadata"]["fingerprint"] != entry["fingerprint"]


def _rebuild(world, entry):
    rows_asof = V3.as_of(world["rows"], entry["created_at"])
    return rd.build_dataset(rows_asof, world["scans"], filters=entry["metadata"]["filters"],
                            members=entry["observation_ids"])


def test_wrong_confirmation_and_bootstrap_start_are_refused(world):
    with pytest.raises(SystemExit):
        F.main(["--freeze", "yes"])
    with pytest.raises(SystemExit):
        F.main(["--start", "2026-01-01"])
    assert world["saved"] == []


def test_not_ready_blocks_the_write(world, monkeypatch):
    monkeypatch.setattr(F, "readiness_at", lambda *a: {"status": "NOT_READY", "blocking_reasons": ["x"]})
    assert F.main(["--freeze", "FREEZE"]) == 1
    assert world["saved"] == []
