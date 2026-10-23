"""Market-wide earnings sections in the morning digest and evening wrap."""
import datetime as dt
from unittest import mock

from scheduler import earnings_email as ee

TODAY = dt.date(2026, 10, 22)   # a Thursday
FRI = dt.date(2026, 10, 23)


def _cal():
    return [
        {"ticker": "POOL", "date": TODAY, "time": "bmo"},
        {"ticker": "CBRE", "date": TODAY, "time": "bmo"},
        {"ticker": "SSNC", "date": TODAY, "time": "amc"},
        {"ticker": "DECK", "date": TODAY, "time": "amc"},
        {"ticker": "EFSI", "date": TODAY, "time": None},
        {"ticker": "AXP", "date": FRI, "time": "bmo"},
    ]


def _scorer(scores):
    def score(tickers, premarket=False):
        return {t: v for t, v in scores.items() if t in tickers}
    return score


def test_reporters_reads_calendar_and_normalizes():
    rows = [
        {"symbol": "pool", "earnings_date": TODAY, "earnings_time": "BMO"},
        {"symbol": "POOL", "earnings_date": TODAY, "earnings_time": "bmo"},       # duplicate
        {"symbol": "XYZ", "earnings_date": dt.date(2026, 11, 30), "earnings_time": "amc"},  # outside
        {"symbol": "ABC", "earnings_date": dt.datetime(2026, 10, 23, 0, 0), "earnings_time": "dmh"},
    ]
    with mock.patch("db.earnings.fetch_earnings_this_week", return_value=rows):
        out = ee.reporters(TODAY, FRI)
    assert out == [{"ticker": "POOL", "date": TODAY, "time": "bmo"},
                   {"ticker": "ABC", "date": FRI, "time": None}]


def test_reporters_empty_on_db_error():
    with mock.patch("db.earnings.fetch_earnings_this_week", side_effect=RuntimeError("db down")):
        assert ee.reporters(TODAY, FRI) == []


def test_build_rows_scores_every_reporter_and_sorts_by_score():
    scorer = _scorer({"POOL": {"score": 88, "last": 192.96, "chg_pct": 11.9},
                      "CBRE": {"score": 41, "last": 131.39, "chg_pct": -2.6},
                      "DECK": {"score": 70, "last": 77.5, "chg_pct": -9.4}})
    rows = ee.build_rows(_cal(), ranked=["POOL"], scorer=scorer, quotes=False)
    today = [r for r in rows if r["date"] == TODAY]
    assert [r["ticker"] for r in today][:3] == ["POOL", "DECK", "CBRE"]   # by score, unscored last
    assert next(r for r in rows if r["ticker"] == "POOL")["ranked"] is True
    assert next(r for r in rows if r["ticker"] == "CBRE")["ranked"] is False
    assert next(r for r in rows if r["ticker"] == "SSNC")["score"] is None
    assert rows[-1]["ticker"] == "AXP"   # later day after today's names


def test_build_rows_survives_scoring_failure():
    rows = ee.build_rows(_cal(), scorer=lambda t, premarket=False: {}, quotes=False)
    assert len(rows) == 6 and all(r["score"] is None for r in rows)


def test_score_tickers_empty_on_scan_error():
    with mock.patch("scan.execution.run_manual_scan_execution", side_effect=RuntimeError("feed")):
        assert ee.score_tickers(["POOL"]) == {}


def test_compose_section_groups_caps_and_links():
    rows = ee.build_rows(
        [{"ticker": f"B{i}", "date": TODAY, "time": "bmo"} for i in range(9)]
        + [{"ticker": "SSNC", "date": TODAY, "time": "amc"}],
        scorer=_scorer({"B0": {"score": 90, "last": 10.0, "chg_pct": 1.5}}), quotes=False)
    html, text = ee.compose_section(
        rows, title="Earnings today",
        order=[(TODAY, "bmo", "Before"), (TODAY, "amc", "After"), (TODAY, None, "TBA")],
        price_label="Last", pct_label="Chg %", link="https://x/scanner?signal=earnings&when=today&all=1",
        link_label="See them all", watched=["SSNC"])
    assert "Earnings today" in html and "Before" in html and "After" in html and "TBA" not in html
    assert html.count("<tr>") == 6 + 1 + 2 + 1   # 6 bmo shown (cap) + 1 amc, 2 header rows, the card
    assert "+ 3 more" in html and "3 more" in html and "10 companies report" in html and "when=today" in html
    assert "<strong>SSNC</strong> <span" in html and ">Watchlist</span>" in html
    assert "+1.50%" in html and "90" in html
    assert "SSNC *" in text and "See them all: https://x/" in text


def test_compose_section_empty_when_nobody_reports():
    assert ee.compose_section([], title="t", order=[(TODAY, "bmo", "B")], price_label="L",
                              pct_label="P", link="l", link_label="ll") == ("", "")


def test_watchlist_line_names_this_weeks_reporters_in_order():
    week = _cal() + [{"ticker": "NVDA", "date": dt.date(2026, 10, 28), "time": "amc"}]
    line = ee.watchlist_line(["nvda", "AXP", "DECK", "MSFT"], week, TODAY)
    assert line.startswith("⚠️ ")
    assert line.index("DECK") < line.index("AXP") < line.index("NVDA")
    assert "<b>DECK</b> today after the close" in line
    assert "<b>AXP</b> tomorrow before the open" in line
    assert "Wed Oct 28 after the close" in line
    assert ee.watchlist_line(["MSFT"], week, TODAY) == ""


def test_next_trading_day_skips_weekend():
    assert ee.next_trading_day(TODAY) == FRI
    assert ee.next_trading_day(FRI) == dt.date(2026, 10, 26)


def test_morning_compose_uses_section_over_old_watchlist_line():
    from scheduler.morning_digest import _compose

    with mock.patch("scheduler.morning_digest._track_record_line", return_value=("", "")):
        html, text = _compose("u@x.com", [], [], ["AAPL"], None,
                              earnings_section=("<h3>📅 Earnings today</h3>", "Earnings today:"))
    assert "📅 Earnings today" in html and "your watchlist)" not in html
    with mock.patch("scheduler.morning_digest._track_record_line", return_value=("", "")):
        html, _ = _compose("u@x.com", [], [], ["AAPL"], None, earnings_section=("", ""))
    assert "Earnings today (your watchlist)" in html   # falls back when the section is empty


def test_morning_section_labels_preopen():
    from scheduler import morning_digest as md

    rows = ee.build_rows(_cal(), scorer=_scorer({"POOL": {"score": 88, "last": 1.0, "chg_pct": 2.0}}),
                         quotes=False)
    html, _ = md._earnings_section(rows, _cal(), TODAY, ["DECK"], preopen=True)
    assert "Pre-mkt %" in html and "Reported before the open" in html and "Reports after the close" in html
    assert "<b>DECK</b> today after the close" in html
    html, _ = md._earnings_section(rows, _cal(), TODAY, [], preopen=False)
    assert "Chg %" in html


def test_evening_window_is_tonight_amc_and_next_morning():
    from scheduler import evening_wrap as ew

    cal = _cal()
    with mock.patch("scheduler.morning_digest.et_now",
                    return_value=dt.datetime(2026, 10, 22, 17, 0)), \
            mock.patch.object(ee, "reporters", return_value=cal), \
            mock.patch.object(ee, "score_tickers", return_value={}), \
            mock.patch.object(ee, "_fill_quotes"):
        earn = ew._earnings_calendar_rows(None)
    today, nxt, week, rows = earn
    assert nxt == FRI
    assert sorted(r["ticker"] for r in rows) == ["AXP", "DECK", "SSNC"]   # not today's bmo/TBA
    html, text = ew._earnings_section(earn, ["AXP"])
    assert "Earnings tonight &amp; tomorrow morning" in html or "Earnings tonight & tomorrow morning" in html
    assert "Tonight after the close" in html and "Tomorrow before the open (Fri)" in html
    assert "<strong>AXP</strong> <span" in html and "when=tomorrow" in html


def test_evening_friday_points_at_monday_and_week():
    from scheduler import evening_wrap as ew

    mon = dt.date(2026, 10, 26)
    cal = [{"ticker": "SLB", "date": FRI, "time": "amc"}, {"ticker": "KO", "date": mon, "time": "bmo"}]
    with mock.patch("scheduler.morning_digest.et_now",
                    return_value=dt.datetime(2026, 10, 23, 17, 0)), \
            mock.patch.object(ee, "reporters", return_value=cal), \
            mock.patch.object(ee, "score_tickers", return_value={}), \
            mock.patch.object(ee, "_fill_quotes"):
        earn = ew._earnings_calendar_rows(None)
    html, _ = ew._earnings_section(earn, [])
    assert "Monday before the open (Mon)" in html and "Monday morning" in html
    assert "when=week" in html


def test_evening_compose_uses_section():
    from scheduler.evening_wrap import _compose_wrap

    html, _ = _compose_wrap([], [], ["AAPL"], earnings_section=("<h3>📅 Earnings tonight</h3>", "x"))
    assert "📅 Earnings tonight" in html and "Earnings tomorrow (your watchlist)" not in html


def test_penny_stocks_counted_but_not_listed():
    rows = ee.build_rows(
        [{"ticker": "POOL", "date": TODAY, "time": "bmo"}, {"ticker": "PENY", "date": TODAY, "time": "bmo"},
         {"ticker": "TINY", "date": TODAY, "time": None}],
        scorer=_scorer({"POOL": {"score": 50, "last": 190.0, "chg_pct": 1.0},
                        "PENY": {"score": 95, "last": 0.55, "chg_pct": 30.0},
                        "TINY": {"score": 60, "last": 2.0, "chg_pct": 1.0}}), quotes=False)
    html, _ = ee.compose_section(rows, title="E", order=[(TODAY, "bmo", "B"), (TODAY, None, "TBA")],
                                 price_label="L", pct_label="P", link="l", link_label="ll")
    assert "POOL" in html and "PENY" not in html and "TBA" not in html
    assert "+ 2 more" in html


def test_duplicates_dropped():
    cal = _cal() + [{"ticker": "POOL", "date": TODAY, "time": "bmo"}]
    rows = ee.build_rows(cal, scorer=lambda t, premarket=False: {}, quotes=False)
    assert [r["ticker"] for r in rows].count("POOL") == 1
    assert ee.watchlist_line(["POOL"], cal, TODAY).count("POOL") == 1


def test_preopen_zero_move_shows_dash():
    rows = ee.build_rows([{"ticker": "RCBC", "date": TODAY, "time": "amc"}],
                         scorer=_scorer({"RCBC": {"score": 6, "last": 55.62, "chg_pct": 0.0}}), quotes=False)
    html, text = ee.compose_section(rows, title="E", order=[(TODAY, "amc", "After")], price_label="Last",
                                    pct_label="Pre-mkt %", link="l", link_label="ll", zero_is_missing=True)
    assert "+0.00%" not in html and "+0.00%" not in text
    html, _ = ee.compose_section(rows, title="E", order=[(TODAY, "amc", "After")], price_label="Last",
                                 pct_label="Chg %", link="l", link_label="ll")
    assert "+0.00%" in html
