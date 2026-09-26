"""Run 76: canonical HSF presentation ranking without raw-order mutation."""
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import discover
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


def test_lenses_preserve_canonical_hsf_order():
    ranked = rank_hsf_opportunities(_ranked_input().assign(VolRel20=[2.0, 2.0, 1.0]))
    filtered = discover.apply_lens(ranked, "volume")
    assert list(filtered["Ticker"]) == ["B", "A"]


def test_cards_and_table_receive_identical_ranked_frame():
    ranked = rank_hsf_opportunities(_ranked_input())
    calls = []
    fake_st = mock.MagicMock()
    fake_st.session_state = {discover.VIEW_KEY: "Cards"}
    with mock.patch.object(discover, "st", fake_st), \
            mock.patch("ui.result_cards.render_result_cards",
                       side_effect=lambda df: calls.append(("cards", list(df["Ticker"])))):
        discover.with_card_view(lambda df, *a, **k: calls.append(("table", list(df["Ticker"]))))(ranked)
    fake_st.session_state[discover.VIEW_KEY] = "Table"
    with mock.patch.object(discover, "st", fake_st):
        discover.with_card_view(lambda df, *a, **k: calls.append(("table", list(df["Ticker"]))))(ranked)
    assert calls == [("cards", ["B", "C", "A"]), ("table", ["B", "C", "A"])]


def test_reordered_stock_handoff_keeps_ticker_and_row_context():
    ranked = rank_hsf_opportunities(_ranked_input())
    state = handoff_state("C", scan_df=ranked)
    assert state["hsf_stock_ticker"] == "C"
    assert state["hsf_stock_row"][HSF_SCORE_COL] == 64


def test_csv_and_history_use_the_hsf_ranked_presentation_frame():
    results = (ROOT / "ui" / "results.py").read_text()
    history = (ROOT / "ui" / "results_tabs.py").read_text()
    app = (ROOT / "app.py").read_text()
    assert "df.to_csv(index=False)" in results
    assert "rank_hsf_opportunities(add_hsf_score_column(run_df_norm))" in history
    assert app.index("df = rank_hsf_opportunities(df)") < app.index("df = render_discover_bar(df)")


def test_frozen_scanner_core_is_not_part_of_run76_diff_contract():
    helper = (ROOT / "ui" / "headline_score.py").read_text()
    assert "run_breakout_scan" not in helper
    assert "save_run" not in helper
