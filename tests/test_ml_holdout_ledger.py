"""SQLite transaction exercises of the same durable claim/reuse policy as Postgres."""
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    import pytest
except ImportError:
    raise unittest.SkipTest("needs pytest")

from db import ml_holdout_evaluations as L

BINDING = {"target": {"label": "F_clean_path_5d", "horizon_days": 5}, "git_sha": "abc", "dataset": "ds1"}


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "ledger.sqlite"
    # Production connections close; SQLite injected connections are owned by the test.
    connections = []

    def connect():
        c = sqlite3.connect(path, timeout=10, check_same_thread=False)
        connections.append(c)
        return c

    L.set_connection_factory(connect)
    yield
    L.set_connection_factory(None)
    for c in connections:
        c.close()


def test_completed_report_is_reused_exactly_and_verified(ledger):
    key, cached = L.reserve(BINDING, [1, 2, 3], run_id="run1")
    assert cached is None
    report = {"binding": BINDING, "holdout_claim": key, "result": .57}
    L.complete(key, BINDING, report)
    assert L.reserve(BINDING, [3, 2, 1], run_id="run2") == (key, report)
    L.verify_completed(key, BINDING, report)
    with pytest.raises(L.HoldoutAlreadyUsed):
        L.verify_completed(key, BINDING, {**report, "result": .99})


def test_interrupted_run_or_changed_settings_cannot_retest(ledger):
    key, _ = L.reserve(BINDING, [1, 2], run_id="run1")
    with pytest.raises(L.HoldoutAlreadyUsed):
        L.reserve(BINDING, [1, 2], run_id="retry")
    L.complete(key, BINDING, {"done": True})
    with pytest.raises(L.HoldoutAlreadyUsed):
        L.reserve({**BINDING, "git_sha": "new-code"}, [1, 2], run_id="new")
    with pytest.raises(L.HoldoutAlreadyUsed):
        L.reserve(BINDING, [2, 3], run_id="overlapping-new-dataset")
    assert L.reserve(BINDING, [4, 5], run_id="unseen")[1] is None


def test_concurrent_claims_have_only_one_winner(ledger):
    def attempt(n):
        try:
            L.reserve(BINDING, [1, 2], run_id=str(n))
            return True
        except L.HoldoutAlreadyUsed:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, [1, 2])) == 1
