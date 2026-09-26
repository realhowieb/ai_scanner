"""Run 79: HSF headline hierarchy, Today quality floor, and trust copy."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd

from ui import market_scans, today
from ui.headline_score import MODEL_DETAIL_LABELS, details_toggle_label
from ui.opportunities import HSF_STRONG_MIN, _status

ROOT = Path(__file__).resolve().parents[1]


def _scan() -> pd.DataFrame:
    return pd.DataFrame({"Ticker": ["STRONG", "BOUNDARY", "WATCH"]})


def _opportunities() -> list[dict]:
    return [
        {"ticker": "STRONG", "score": 88, "status": "STRONG"},
        {"ticker": "BOUNDARY", "score": HSF_STRONG_MIN, "status": "STRONG"},
        {"ticker": "WATCH", "score": HSF_STRONG_MIN - 1, "status": "WATCH"},
    ]


def test_canonical_strong_boundary_is_the_today_quality_floor():
    assert HSF_STRONG_MIN == 75
    assert _status(HSF_STRONG_MIN, False) == "STRONG"
    assert _status(HSF_STRONG_MIN - 1, False) == "WATCH"


def test_today_top_setups_keeps_only_qualifying_results_and_boundary():
    with mock.patch(
        "ui.results_intelligence.consolidate_scanner_results",
        return_value=_opportunities(),
    ):
        result = today.today_top_setups(_scan(), n=5)
    assert result["state"] == "qualifying"
    assert result["threshold"] == HSF_STRONG_MIN
    assert [item["ticker"] for item in result["setups"]] == ["STRONG", "BOUNDARY"]


def test_today_distinguishes_weak_market_from_empty_scan():
    weak = [{"ticker": "WATCH", "score": HSF_STRONG_MIN - 1, "status": "WATCH"}]
    with mock.patch(
        "ui.results_intelligence.consolidate_scanner_results",
        return_value=weak,
    ):
        result = today.today_top_setups(_scan())
    assert result["state"] == "no_qualifying"
    assert result["setups"] == []

    empty = today.today_top_setups(pd.DataFrame())
    assert empty["state"] == "empty_scan"
    assert empty["setups"] == []


def test_scanner_default_preserves_below_threshold_opportunities():
    with mock.patch(
        "ui.results_intelligence.consolidate_scanner_results",
        return_value=_opportunities(),
    ):
        scanner_results = market_scans.top_setups(_scan(), n=5)
        today_results = market_scans.top_setups(
            _scan(), n=5, minimum_score=HSF_STRONG_MIN
        )
    assert [item["ticker"] for item in scanner_results] == [
        "STRONG",
        "BOUNDARY",
        "WATCH",
    ]
    assert [item["ticker"] for item in today_results] == ["STRONG", "BOUNDARY"]


def test_secondary_metrics_are_named_as_model_details():
    label = details_toggle_label(
        ["HSF Score", "BreakoutScore", "PreBreakoutProb%", "AI Confidence"]
    )
    assert label.startswith("Show model details")
    assert MODEL_DETAIL_LABELS["AI Confidence"] == "5D outcome probability"
    assert MODEL_DETAIL_LABELS["PreBreakoutProb%"] == "PreBreakout setup probability"


def test_stock_intelligence_separates_hsf_reasoning_from_model_details():
    source = (ROOT / "ui" / "stock_intelligence.py").read_text()
    assert source.index('"Why HSF scored it this way"') < source.index('"Model details"')
    assert "Supporting model outputs are separate from the headline HSF Score" in source


def test_methodology_states_live_vs_historical_and_defined_model_outcome():
    source = (ROOT / "ui" / "methodology.py").read_text()
    assert "Live ranking asks what looks interesting now" in source
    assert "Historical Research separately asks" in source
    assert "reaching +4% before " in source
    assert "within five trading days" in source


def test_customer_copy_avoids_unsupported_certainty_claims():
    paths = (
        ROOT / "ui" / "today.py",
        ROOT / "ui" / "methodology.py",
        ROOT / "ui" / "headline_score.py",
        ROOT / "ui" / "stock_intelligence.py",
        ROOT / "ui" / "result_helpers.py",
    )
    text = "\n".join(path.read_text().lower() for path in paths)
    for phrase in ("guaranteed returns", "guaranteed profit", "sure winner", "likely winner"):
        assert phrase not in text
