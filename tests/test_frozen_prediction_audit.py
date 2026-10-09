from unittest.mock import MagicMock

from scripts.frozen_prediction_audit import blockers, inventory, read_rows, summarize


def row(**kwargs):
    return dict(id=1, ticker="ABC", source="opportunity", fired_at="2026-09-28T15:00:00Z", **kwargs)


def test_missing_provenance_never_certified():
    assert "MISSING_ARTIFACT_SHA256" in blockers(row())
    assert summarize([row()])["classification"]["VERIFIED_EVALUABLE"] == 0


def test_boundary_equal_excluded():
    assert "TRAINING_BOUNDARY_UNVERIFIED_OR_OVERLAPPING" in blockers(row(), "2026-09-28T15:00:00Z")
    assert "TRAINING_BOUNDARY_UNVERIFIED_OR_OVERLAPPING" not in blockers(row(), "2026-09-27T15:00:00Z")


def test_proxy_not_target():
    assert "TARGET_IDENTITY_UNVERIFIED" in blockers(row(provenance={"target_name": "return_5d"}))


def test_zero_prediction_is_present_nan_is_missing():
    assert inventory([row(prebreakout_prob=0)])["prebreakout_prediction"]["count"] == 1
    assert inventory([row(prebreakout_prob=float("nan"))])["prebreakout_prediction"]["count"] == 0


def test_selection_before_outcomes_and_order_independent():
    a = row()
    b = {**row(prebreakout_prob=13.1, return_5d=.1), "id": 2}
    assert summarize([b, a]) == summarize([a, b])
    assert summarize([b, a])["prior_audit_funnel"]["with_prediction"] == 0


def test_database_enforced_read_only_first():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    assert read_rows(conn) == []
    commands = [call.args[0] for call in conn.execute.call_args_list]
    assert commands[0] == "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert "statement_timeout" in commands[1]
    assert "LIMIT 10001" in commands[2]
