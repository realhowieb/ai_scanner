"""Run 76: canonical HSF presentation ranking without raw-order mutation."""
from pathlib import Path
from unittest import mock

import pandas as pd

try:  # the Streamlit app was retired; its modules are gone
    from ui import discover
except ImportError:
    discover = None
from ui.headline_score import HSF_SCORE_COL, rank_hsf_opportunities
from ui.stock_handoff import handoff_state

ROOT = Path(__file__).resolve().parents[1]


def _ranked_input():
    return pd.DataFrame([
        {"Ticker": "A", HSF_SCORE_COL: 45, "BreakoutScore": 90, "Why": "a"},
        {"Ticker": "B", HSF_SCORE_COL: 71, "BreakoutScore": 20, "Why": "b"},
        {"Ticker": "C", HSF_SCORE_COL: 64, "BreakoutScore": 40, "Why": "c"},
    ], index=[10, 20, 30])


def test_basic_hsf_order_and_raw_frame_unchanged():
    raw = _ranked_input()
    before = raw.copy(deep=True)
    ranked = rank_hsf_opportunities(raw)
    assert list(ranked["Ticker"]) == ["B", "C", "A"]
    assert list(ranked.index) == [20, 30, 10]
    pd.testing.assert_frame_equal(raw, before)
    assert ranked is not raw


def test_ties_use_breakout_score_then_ticker():
    raw = pd.DataFrame([
        {"Ticker": "Z", HSF_SCORE_COL: 60, "BreakoutScore": 10},
        {"Ticker": "B", HSF_SCORE_COL: 60, "BreakoutScore": 20},
        {"Ticker": "A", HSF_SCORE_COL: 60, "BreakoutScore": 20},
    ])
    assert list(rank_hsf_opportunities(raw)["Ticker"]) == ["A", "B", "Z"]


def test_empty_missing_and_malformed_scores_are_safe():
    empty = pd.DataFrame()
    assert rank_hsf_opportunities(None) is None
    assert rank_hsf_opportunities(empty) is empty
    missing = pd.DataFrame({"Ticker": ["B", "A"]})
    assert rank_hsf_opportunities(missing) is missing
    malformed = pd.DataFrame([
        {"Ticker": "BAD", HSF_SCORE_COL: "not-a-number"},
        {"Ticker": "GOOD", HSF_SCORE_COL: 50},
        {"Ticker": None, HSF_SCORE_COL: None},
    ])
    assert list(rank_hsf_opportunities(malformed)["Ticker"])[0] == "GOOD"


def test_reordered_stock_handoff_keeps_ticker_and_row_context():
    ranked = rank_hsf_opportunities(_ranked_input())
    state = handoff_state("C", scan_df=ranked)
    assert state["hsf_stock_ticker"] == "C"
    assert state["hsf_stock_row"][HSF_SCORE_COL] == 64


def test_frozen_scanner_core_is_not_part_of_run76_diff_contract():
    helper = (ROOT / "ui" / "headline_score.py").read_text()
    assert "run_breakout_scan" not in helper
    assert "save_run" not in helper
