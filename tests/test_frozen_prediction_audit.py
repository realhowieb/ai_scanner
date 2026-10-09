import json
from unittest.mock import MagicMock

import pytest

from scripts.frozen_prediction_audit import blockers, inventory, read_rows, scan_inventory, summarize


def row(**kwargs):
    return dict(id=1, ticker="ABC", source="opportunity", fired_at="2026-09-28T15:00:00Z", **kwargs)


def test_missing_provenance_never_certified():
    assert "MISSING_ARTIFACT_SHA256" in blockers(row())
    assert summarize([row()])["classification"]["VERIFIED_EVALUABLE"] == 0
    assert inventory([row(provenance=None)])["model_version"]["count"] == 0


def test_boundary_equal_excluded():
    assert "TRAINING_BOUNDARY_UNVERIFIED_OR_OVERLAPPING" in blockers(row(), "2026-09-28T15:00:00Z")
    assert "TRAINING_BOUNDARY_UNVERIFIED_OR_OVERLAPPING" not in blockers(row(), "2026-09-27T15:00:00Z")


def test_proxy_not_target():
    assert "TARGET_IDENTITY_UNVERIFIED" in blockers(row(provenance={"target_name": "return_5d"}))


def test_code_version_tag_cannot_certify_served_artifact():
    sample = row(provenance={"model_version": "prebreakout-xgb-v16"})
    assert "MISSING_ARTIFACT_SHA256" in blockers(sample)
    assert summarize([sample])["classification"]["VERIFIED_EVALUABLE"] == 0


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
    conn.execute.return_value.fetchone.return_value = {"transaction_read_only": "on"}
    assert read_rows(conn) == []
    commands = [call.args[0] for call in conn.execute.call_args_list]
    assert commands[0] == "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert commands[1] == "SHOW transaction_read_only"
    assert "statement_timeout" in commands[2]
    assert "LIMIT 10001" in commands[3]


def test_refuses_read_when_database_is_not_read_only():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"transaction_read_only": "off"}
    with pytest.raises(RuntimeError, match="read-only mode"):
        read_rows(conn)
    assert len(conn.execute.call_args_list) == 2


def test_different_raw_predictions_can_share_calibrated_floor():
    records = [{"Ticker": "ABC", "F": 1, "PreBreakoutProbRaw": .01, "PreBreakoutProb%": 13.1},
               {"Ticker": "XYZ", "F": 2, "PreBreakoutProbRaw": .02, "PreBreakoutProb%": 13.1}]
    report = scan_inventory([{"created_at": "2026-09-28T15:00:00Z", "results_json": json.dumps(records)}],
                            ["F"], {"x": [.034, .14], "y": [.13126, .23]}, [row()])
    assert report["counts"]["pairs_match_current_calibration"] == 2
    assert report["floor_raw_summary"]["unique_values"] == 2
    assert report["distinct_current_schema_source_signatures"] == 2
    assert report["observations_with_exact_run_timestamp_candidate"] == 1
    assert "do not establish" in report["warning"]


def test_history_missing_source_fields_not_filled():
    report = scan_inventory([{"created_at": "2026-09-28", "results_json": '[{"Ticker":"ABC"}]'}],
                            ["F"], None, [])
    assert report["counts"]["complete_current_schema_source_rows"] == 0
    assert report["counts"].get("raw_display_pairs", 0) == 0
