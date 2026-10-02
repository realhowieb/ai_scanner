import datetime as dt
from pathlib import Path
from unittest import mock

from analytics import observation_explorer as oe
from db import observation_explorer as store

ROOT = Path(__file__).resolve().parents[1]


def _observation(symbol, cohort, *, scan_id="scan-1", outcome=None):
    row = {
        "observation_id": f"{scan_id}:{symbol}:{cohort}",
        "symbol": symbol,
        "timestamp": "2026-10-06T14:00:00+00:00",
        "context": "scheduled:us_market",
        "research_cohort": cohort,
        "market_context": {
            "scan_id": scan_id,
            "control_design": "run59b_liquidity_matched_disjoint_v2",
        },
        "scanners": [{"name": "breakout"}],
    }
    if outcome is not None:
        row["outcomes"] = {"+15m": outcome}
    return row


def test_run59b_integrity_fixture_and_overlap_detection():
    disjoint = [
        _observation("AAPL", "CANDIDATE"),
        _observation("MSFT", "NEAR_MISS"),
        _observation("NVDA", "CONTROL"),
    ]
    assert oe.cohort_overlap(disjoint) == {
        "candidate_near_miss": 0,
        "candidate_control": 0,
        "near_miss_control": 0,
        "status": "PASS",
    }
    overlapping = disjoint + [_observation("AAPL", "CONTROL")]
    result = oe.cohort_overlap(overlapping)
    assert result["candidate_control"] == 1
    assert result["status"] == "OVERLAP_DETECTED"


def test_epoch_boundaries_are_immutable_and_current_is_run59b():
    specs = oe.epoch_specs()
    assert specs[-1]["label"] == "Current — Run 59B"
    assert specs[-1]["control_design"] == "run59b_liquidity_matched_disjoint_v2"
    assert specs[-2]["end"] == specs[-1]["start"]
    assert oe.epoch_for_timestamp("2026-10-06T14:00:00+00:00")["current"] is True


def test_filter_values_are_parameterized_and_cover_core_filters():
    start = dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc)
    where, params = store.build_where({
        "dataset": "scanner",
        "start": start,
        "end": "2026-10-07T00:00:00+00:00",
        "cohorts": ["candidate", "near_miss", "control"],
        "symbols": ["AAPL", "MSFT"],
        "signal": "breakout",
        "horizon": "+15m",
        "scan_id": "scan'; DROP TABLE hsf_observations; --",
        "session": "regular",
        "control_design": "run59b_liquidity_matched_disjoint_v2",
        "score_min": 80,
        "score_max": 100,
        "status": "Matured",
    })
    assert "DROP TABLE" not in where
    assert "%s" in where
    assert start in params
    assert ["CANDIDATE", "NEAR_MISS", "CONTROL"] in params
    assert ["AAPL", "MSFT"] in params
    assert "scan'; DROP TABLE hsf_observations; --" in params
    assert "run59b_liquidity_matched_disjoint_v2" in params
    assert "+15m" in params


def test_pending_usable_excluded_and_control_design_filters():
    for status, expected in (("Pending", "NOT"), ("Usable", "directional_return"),
                             ("Excluded", "UNAVAILABLE")):
        where, params = store.build_where({
            "dataset": "scanner", "status": status, "horizon": "+30m",
            "compatible_control_design": "run59b_liquidity_matched_disjoint_v2",
        })
        assert expected in where
        assert "+30m" in params
        assert "run59b_liquidity_matched_disjoint_v2" in params


def test_scanner_and_stair_stepper_filters_are_separate():
    scanner_where, scanner_params = store.build_where({"dataset": "scanner"})
    stair_where, stair_params = store.build_where({"dataset": "stair_stepper"})
    assert "LIKE" in scanner_where
    assert scanner_params == ("scheduled:%",)
    assert "LIKE" not in stair_where
    assert stair_params == ("day_trader:stair_stepper",)


def test_null_and_unavailable_outcomes_are_not_usable():
    assert oe.derived_status({}) == "PENDING"
    assert oe.derived_status({"+15m": {"data_status": "MATURED"}}) == "MATURED"
    assert oe.derived_status({
        "+15m": {"data_status": "MATURED", "raw_return": 0.0},
    }) == "USABLE"
    assert oe.derived_status({
        "+15m": {"data_status": "UNAVAILABLE"},
    }) == "EXCLUDED"


def test_pagination_clamps_invalid_values():
    assert oe.clamp_page(0, 500) == (1, oe.MAX_PAGE_SIZE)
    assert oe.clamp_page("bad", "bad") == (1, oe.DEFAULT_PAGE_SIZE)


def test_query_page_is_bounded_and_handles_empty_results():
    empty = {"rows": [], "error": None, "duration_ms": 1.0}
    with mock.patch.object(store, "_execute", return_value=empty) as execute:
        result = store.query_page({"dataset": "scanner"}, page=2, page_size=500)
    assert result["rows"] == []
    assert result["page_size"] == oe.MAX_PAGE_SIZE
    sql, params = execute.call_args.args
    assert "LIMIT %s OFFSET %s" in sql
    assert params[-2:] == (oe.MAX_PAGE_SIZE, oe.MAX_PAGE_SIZE)


def test_integrity_does_not_report_pass_when_database_is_unavailable():
    unavailable = {"rows": [], "error": "database_unavailable", "duration_ms": None}
    with mock.patch.object(store, "_execute", return_value=unavailable):
        result = store.query_integrity({"dataset": "scanner"})
    assert result["status"] == "UNKNOWN"


def test_csv_export_is_bounded():
    result = oe.bounded_csv(({"symbol": f"S{i}"} for i in range(5)), maximum=3)
    assert result["rows"] == 3
    assert result["truncated"] is True
    assert "S3" not in result["csv"]


def test_admin_authorization_precedes_explorer_queries():
    source = (ROOT / "ui" / "observation_explorer.py").read_text()
    guard = source.index('if not bool(st.session_state.get("is_admin"))')
    first_query = source.index("options = _options(option_filters)")
    assert guard < first_query


def test_schema_indexes_are_not_created_by_explorer_rendering():
    source = (ROOT / "ui" / "observation_explorer.py").read_text().upper()
    db_source = (ROOT / "db" / "observation_explorer.py").read_text().upper()
    assert "CREATE INDEX" not in source
    assert "CREATE INDEX" not in db_source


def test_score_filter_does_not_substitute_breakout_score():
    assert "BREAKOUT" not in store.SCORE_EXPR.upper()
