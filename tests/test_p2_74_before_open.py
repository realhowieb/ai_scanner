"""P2-74 — Today's "Before the open" card: pre-market movers from this morning's
premarket scan, Pro+, until 10:30 ET ("This morning's pre-market movers" after the open)."""
import datetime as dt
import importlib.util
import math
import unittest
from unittest import mock

import pandas as pd

from scan import headless_common as hc
from ui import before_open as bo
from ui import scan_providers as sp

HAS_ST = True
UTC = dt.timezone.utc
TUE = dt.date(2026, 9, 29)
TUE_840_ET = dt.datetime(2026, 9, 29, 12, 40, tzinfo=UTC)     # after the 8:35 scan, before the open
TUE_830_ET = dt.datetime(2026, 9, 29, 12, 30, tzinfo=UTC)     # before the 8:35 scan
TUE_NOON_ET = dt.datetime(2026, 9, 29, 16, 0, tzinfo=UTC)     # market open
TUE_1000_ET = dt.datetime(2026, 9, 29, 14, 0, tzinfo=UTC)     # first hour of trading: still shown
TUE_1030_ET = dt.datetime(2026, 9, 29, 14, 30, tzinfo=UTC)    # window closed
MON_8PM_ET = dt.datetime(2026, 9, 29, 0, 0, tzinfo=UTC)
SAT = dt.datetime(2026, 10, 3, 13, 0, tzinfo=UTC)


def snap(trade_t, daily_t, daily_c, prev_t="2026-09-25T04:00:00Z", prev_c=90.0, price=110.0):
    return {"latestTrade": {"p": price, "t": trade_t},
            "dailyBar": {"t": daily_t, "c": daily_c},
            "prevDailyBar": {"t": prev_t, "c": prev_c}}


class SnapshotTests(unittest.TestCase):
    def test_previous_close_is_yesterdays_daily_bar(self):
        s = snap("2026-09-29T12:30:00Z", "2026-09-28T04:00:00Z", 100.0)
        self.assertEqual(sp.premarket_quote_from_snapshot(s, TUE), (110.0, 100.0))

    def test_todays_daily_bar_falls_back_to_prev_daily_bar(self):
        s = snap("2026-09-29T12:30:00Z", "2026-09-29T04:00:00Z", 105.0,
                 prev_t="2026-09-28T04:00:00Z", prev_c=100.0)
        self.assertEqual(sp.premarket_quote_from_snapshot(s, TUE), (110.0, 100.0))

    def test_last_nights_after_hours_trade_is_not_pre_market(self):
        s = snap("2026-09-28T23:30:00Z", "2026-09-28T04:00:00Z", 100.0)   # Mon 7:30 PM ET
        self.assertIsNone(sp.premarket_quote_from_snapshot(s, TUE))

    def test_bad_or_missing_data(self):
        self.assertIsNone(sp.premarket_quote_from_snapshot(None, TUE))
        self.assertIsNone(sp.premarket_quote_from_snapshot({"latestTrade": {"p": 1.0}}, TUE))
        s = snap("2026-09-29T12:30:00Z", "2026-09-28T04:00:00Z", 0.0)
        self.assertIsNone(sp.premarket_quote_from_snapshot(s, TUE))


def _results():
    return pd.DataFrame({"Ticker": ["NVDA", "AMD", "EA"], "BreakoutScore": [90.0, 80.0, 70.0],
                         "Last": [100.0, 50.0, 10.0]})


class AddPremarketColumnsTests(unittest.TestCase):
    def _run(self, quotes):
        with mock.patch("ui.scan_providers.get_alpaca_premarket_quotes", return_value=quotes) as q:
            out = hc.add_premarket_columns(_results(), today=TUE)
        q.assert_called_once_with(["NVDA", "AMD", "EA"], TUE)
        return out

    def test_adds_pm_last_and_pct_vs_previous_close(self):
        out = self._run({"NVDA": (105.0, 100.0), "AMD": (48.0, 50.0)})
        self.assertEqual(out["PMLast"].tolist()[:2], [105.0, 48.0])
        self.assertEqual(out["PMPctChange"].tolist()[:2], [5.0, -4.0])
        self.assertTrue(math.isnan(out.loc[2, "PMPctChange"]))            # EA: no trade yet today

    def test_scores_and_order_unchanged(self):
        out = self._run({"NVDA": (105.0, 100.0)})
        self.assertEqual(out["Ticker"].tolist(), ["NVDA", "AMD", "EA"])
        self.assertEqual(out["BreakoutScore"].tolist(), [90.0, 80.0, 70.0])
        self.assertEqual(out["Last"].tolist(), [100.0, 50.0, 10.0])

    def test_provider_error_fails_open(self):
        with mock.patch("ui.scan_providers.get_alpaca_premarket_quotes", side_effect=OSError("down")):
            out = hc.add_premarket_columns(_results(), today=TUE)
        self.assertTrue(out["PMLast"].isna().all())

    def test_empty_frame_passes_through(self):
        empty = pd.DataFrame()
        self.assertIs(hc.add_premarket_columns(empty, today=TUE), empty)


class PipelineTests(unittest.TestCase):
    def _pipeline(self, session):
        with (
            mock.patch("data.tradability.filter_tradable_tickers", side_effect=lambda s: s),
            mock.patch.object(hc, "fetch_headless_prices", return_value=({}, [], 0.0)),
            mock.patch.object(hc, "build_filtered_price_data", return_value={}),
            mock.patch.object(hc, "maybe_run_gap_filter"),
            mock.patch.object(hc, "run_headless_breakout", return_value=_results()),
            mock.patch("ui.scan_providers.get_alpaca_premarket_quotes", return_value={"NVDA": (101.0, 100.0)}) as pm,
            mock.patch("ui.scan_providers.get_alpaca_extended_last_prices", return_value={}) as ah,
        ):
            df, _ = hc.run_headless_pipeline(
                session, ["NVDA", "AMD", "EA"], min_price=5, max_price=1000, min_dollar_vol=1,
                use_parallel=False, parallel_workers=1, parallel_chunk=10, apply_gap_filter=False,
                top_n=10, session_label=session)
        return df, pm, ah

    def test_only_premarket_gets_pm_columns(self):
        df, pm, ah = self._pipeline("premarket")
        pm.assert_called_once()
        ah.assert_not_called()
        self.assertEqual(df.loc[0, "PMPctChange"], 1.0)
        for session in ("regular", "postmarket"):
            df, pm, _ = self._pipeline(session)
            pm.assert_not_called()
            self.assertNotIn("PMLast", df.columns)


def run(rid, label, created):
    return {"id": rid, "label": label, "created_at": created}


RUNS = [run(1, "premarket", "2026-09-29T12:35:00+00:00"),        # Tue 8:35 ET
        run(2, "postmarket", "2026-09-28T21:35:00+00:00"),
        run(3, "premarket", "2026-09-28T12:35:00+00:00")]        # Monday's


class WindowTests(unittest.TestCase):
    def test_todays_premarket_run_before_the_open(self):
        self.assertEqual(bo.current_premarket_run(RUNS, TUE_840_ET)["id"], 1)

    def test_still_shown_in_the_first_hour_of_trading(self):
        self.assertEqual(bo.current_premarket_run(RUNS, TUE_1000_ET)["id"], 1)

    def test_nothing_before_the_scan_after_1030_or_off_days(self):
        self.assertIsNone(bo.current_premarket_run(RUNS, TUE_830_ET))   # Monday's run is stale
        self.assertIsNone(bo.current_premarket_run(RUNS, TUE_1030_ET))
        self.assertIsNone(bo.current_premarket_run(RUNS, TUE_NOON_ET))
        self.assertIsNone(bo.current_premarket_run(RUNS, MON_8PM_ET))
        self.assertIsNone(bo.current_premarket_run(RUNS, SAT))


DF = pd.DataFrame([
    {"Ticker": "AAA", "Last": 10.0, "PMLast": 10.6, "PMPctChange": 6.0},
    {"Ticker": "bbb", "Last": 20.0, "PMLast": 18.0, "PMPctChange": -10.0},
    {"Ticker": "CCC", "Last": 30.0, "PMLast": None, "PMPctChange": None},
    {"Ticker": "DDD", "Last": 40.0, "PMLast": 40.1, "PMPctChange": 0.25},
])


class MoversTests(unittest.TestCase):
    def test_biggest_moves_with_scores(self):
        with mock.patch("ui.headline_score.hsf_scores_by_ticker", return_value={"BBB": 61}):
            movers = bo.premarket_movers(DF)
        self.assertEqual(movers, [{"ticker": "BBB", "pct": -10.0, "last": 18.0, "score": 61},
                                  {"ticker": "AAA", "pct": 6.0, "last": 10.6, "score": None}])

    def test_no_premarket_columns(self):
        self.assertEqual(bo.premarket_movers(pd.DataFrame([{"Ticker": "A", "Last": 1}])), [])


SCRIPT = '''
import datetime as dt
from utils.st_compat import st
from ui.before_open import render_before_open
render_before_open(now=dt.datetime.fromisoformat(st.session_state["now"]))
'''


class WiringTests(unittest.TestCase):
    def test_today_shows_before_open_between_top_setups_and_after_close(self):
        src = open("ui/today.py").read()
        top = src.index('("top setups"')
        before = src.index('("before the open", _section_before_open)')
        after = src.index('("after the close", _section_after_close)')
        self.assertLess(top, before)
        self.assertLess(before, after)
        self.assertIn("from ui.before_open import render_before_open", src)


if __name__ == "__main__":
    unittest.main()
