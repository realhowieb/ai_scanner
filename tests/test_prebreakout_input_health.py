"""Missing input evidence must never turn into a green health badge."""
import copy

from db.ml_operations import prebreakout_input_health


def row(mask=None):
    return {'fired_at': '2026-11-07T07:00:00Z', 'names': ['Price', 'LowSlope'],
            'values': [0.0, 0.0], 'mask': [False, True] if mask is None else mask,
            'version': 'v9', 'sha': 'hash', 'contract': 'snapshot-v1'}


def test_real_zero_is_available_but_defaulted_zero_warns():
    result = prebreakout_input_health([row()])
    assert result['status'] == 'WATCH'
    assert (result['latest_real_inputs'], result['latest_defaulted_inputs']) == (1, 1)
    assert result['features'][0] == {'name': 'LowSlope', 'defaulted_predictions': 1,
                                     'predictions': 1, 'defaulted_pct': 100.0}
    assert 'values' not in result and 'mask' not in result


def test_complete_inputs_are_healthy_without_claiming_quality():
    result = prebreakout_input_health([row([False, False])])
    assert result['status'] == 'HEALTHY'
    assert result['latest_real_inputs'] == 2
    assert 'not accuracy' in result['note']


def test_missing_and_malformed_latest_vectors_are_unknown():
    assert prebreakout_input_health([])['status'] == 'UNKNOWN'
    for changes in [{'names': None}, {'mask': [False]}, {'values': [None, 0]},
                    {'mask': [0, 1]}, {'names': ['X', 'X']}, {'values': [float('nan'), 0]}]:
        item = {**row(), **changes}
        result = prebreakout_input_health([item, row([False, False])])
        assert result['status'] == 'UNKNOWN'
        assert result['latest_real_inputs'] is None


def test_different_artifact_or_schema_is_not_mixed_into_current_coverage():
    newest = row([False, False])
    previous_model = {**row(), 'sha': 'other'}
    previous_schema = {**row(), 'names': ['Other', 'Fields']}
    previous_contract = {**row(), 'contract': 'old-contract'}
    result = prebreakout_input_health([newest, previous_model, previous_schema, previous_contract])
    assert result['comparable_predictions'] == 1
    assert result['other_identity_predictions'] == 3
    assert result['latest_defaulted_inputs'] == 0


def test_recent_defaults_warn_even_when_latest_prediction_is_complete():
    recent = row()
    result = prebreakout_input_health([row([False, False]), recent])
    assert result['status'] == 'WATCH'
    assert result['features'][0]['defaulted_pct'] == 50
    assert result['latest_defaulted_inputs'] == 0
    assert recent == copy.deepcopy(row())


def test_missing_artifact_identity_cannot_claim_healthy():
    result = prebreakout_input_health([{**row([False, False]), 'sha': None}])
    assert result['status'] == 'UNKNOWN'
