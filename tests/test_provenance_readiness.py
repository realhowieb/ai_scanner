import copy
from unittest.mock import patch

import pandas as pd

from analytics.prediction_provenance import attach, digest, mark_recalculation, models_from_row


def test_recalculation_retains_actual_time_and_parent_identity():
    parent = {"status": "captured", "inferred_at": "2026-10-09T18:00:00Z", "source_scan_id": "scan-1"}
    models = {"prebreakout": {"status": "captured", "inferred_at": "2026-10-09T19:00:00Z", "raw_probability": 0.3},
              "ai_confidence": {"inferred_at": "unchanged"}}
    before = copy.deepcopy(models)
    result = mark_recalculation(models, {"prebreakout": parent}, role="prebreakout", context="brief_pick_recalculation")
    assert models == before
    assert result["prebreakout"]["inferred_at"] == "2026-10-09T19:00:00Z"
    assert result["prebreakout"]["raw_probability"] == 0.3
    assert result["prebreakout"]["parent_prediction"]["provenance_hash"] == digest(parent)
    assert result["ai_confidence"] == models["ai_confidence"]


def test_missing_parent_is_not_reconstructed():
    result = mark_recalculation({"prebreakout": {"status": "captured"}}, {}, role="prebreakout", context="brief_pick_recalculation")
    assert result["prebreakout"]["parent_prediction"] is None
    assert result["prebreakout"]["parent_unavailable_reason"]


def test_boundaries_are_never_inferred_from_training_timestamp():
    frame = pd.DataFrame({"Ticker": ["ABC"]})
    inputs = pd.DataFrame({"x": [0.]})
    attach(frame, "prebreakout", inputs, inputs.isna(), [0.3], [0.2], {"trained_at": "2026-10-09"})
    model = models_from_row(frame.iloc[0])["prebreakout"]
    for field in ("training_data_end", "calibration_data_end", "target_version"):
        assert model[field] is None
        assert model["metadata_availability"][field]["status"] == "unavailable"


def test_supplied_boundaries_are_preserved():
    frame = pd.DataFrame({"Ticker": ["ABC"]})
    inputs = pd.DataFrame({"x": [0.]})
    attach(frame, "prebreakout", inputs, inputs.isna(), [0.3], [0.2], {"training_data_end": "2026-01-01"})
    model = models_from_row(frame.iloc[0])["prebreakout"]
    assert model["training_data_end"] == "2026-01-01"
    assert model["metadata_availability"]["training_data_end"]["status"] == "supplied"


def test_real_pick_path_marks_single_inference_without_changing_score():
    from scheduler.morning_digest import _prebreakout_picks

    parent = {"status": "captured", "inferred_at": "old", "source_scan_id": "scan-1"}
    frame = pd.DataFrame({"Ticker": ["ABC"], "ModelProvenance": [{"prebreakout": parent}]})
    scored = pd.DataFrame({"Ticker": ["ABC"], "PreBreakoutProb%": [30.],
                           "ModelProvenance": [{"prebreakout": {"status": "captured", "inferred_at": "new"}}]})
    with patch("ml_prebreakout.score_prebreakout", return_value=scored) as inference:
        picks = _prebreakout_picks(frame)
    inference.assert_called_once_with(frame)
    assert picks[0]["prob"] == 30.
    assert picks[0]["models"]["prebreakout"]["inference_context"] == "brief_pick_recalculation"
    assert picks[0]["models"]["prebreakout"]["inferred_at"] == "new"
