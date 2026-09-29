"""P2-45 — Today's "After the close" card: after-hours movers from the latest
postmarket scan, Pro+, between the close and the next open."""
import datetime as dt
import importlib.util
import unittest
from unittest import mock

import pandas as pd

from ui import after_close as ac

HAS_ST = importlib.util.find_spec("streamlit") is not None
UTC = dt.timezone.utc
MON_8PM_ET = dt.datetime(2026, 9, 29, 0, 0, tzinfo=UTC)       # Mon 20:00 ET
TUE_8AM_ET = dt.datetime(2026, 9, 29, 12, 0, tzinfo=UTC)      # Tue 08:00 ET (premarket)
TUE_NOON_ET = dt.datetime(2026, 9, 29, 16, 0, tzinfo=UTC)     # market open
SAT = dt.datetime(2026, 10, 3, 15, 0, tzinfo=UTC)


def run(rid, label, created):
    return {"id": rid, "label": label, "created_at": created}


RUNS = [run(1, "postmarket", "2026-09-28T20:35:00+00:00"),      # Mon 16:35 ET
        run(2, "postmarket", "2026-09-28T21:35:00+00:00"),      # Mon 17:35 ET (latest)
        run(3, "US_MARKET", "2026-09-28T22:00:00+00:00"),
        run(4, "postmarket", "2026-09-25T20:35:00+00:00")]      # last Friday


class WindowTests(unittest.TestCase):
    def test_last_close(self):
        self.assertEqual(ac.last_close(MON_8PM_ET).isoformat(), "2026-09-28T16:00:00-04:00")
        self.assertEqual(ac.last_close(TUE_8AM_ET).isoformat(), "2026-09-28T16:00:00-04:00")
        self.assertEqual(ac.last_close(SAT).isoformat(), "2026-10-02T16:00:00-04:00")

    def test_picks_latest_postmarket_run_of_this_evening(self):
        self.assertEqual(ac.current_postmarket_run(RUNS, MON_8PM_ET)["id"], 2)
        self.assertEqual(ac.current_postmarket_run(RUNS, TUE_8AM_ET)["id"], 2)   # still shown before the open

    def test_nothing_during_market_hours_or_when_stale(self):
        self.assertIsNone(ac.current_postmarket_run(RUNS, TUE_NOON_ET))
        self.assertIsNone(ac.current_postmarket_run([RUNS[3]], MON_8PM_ET))       # Friday's run is stale


class MoversTests(unittest.TestCase):
    DF = pd.DataFrame([
        {"Ticker": "AAA", "Last": 10.0, "AHLast": 10.5, "AHPctChange": 5.0},
        {"Ticker": "bbb", "Last": 20.0, "AHLast": 18.0, "AHPctChange": -10.0},
        {"Ticker": "CCC", "Last": 30.0, "AHLast": None, "AHPctChange": None},
        {"Ticker": "DDD", "Last": 40.0, "AHLast": 40.4, "AHPctChange": 1.0},
    ])

    def test_biggest_absolute_moves_with_scores(self):
        with mock.patch("ui.headline_score.hsf_scores_by_ticker", return_value={"BBB": 61}):
            movers = ac.after_hours_movers(self.DF, n=2)
        self.assertEqual([m["ticker"] for m in movers], ["BBB", "AAA"])
        self.assertEqual(movers[0], {"ticker": "BBB", "ah_pct": -10.0, "ah_last": 18.0, "score": 61})
        self.assertIsNone(movers[1]["score"])                                     # not a qualifying HSF row

    def test_no_after_hours_columns(self):
        self.assertEqual(ac.after_hours_movers(pd.DataFrame([{"Ticker": "A", "Last": 1}])), [])
        self.assertEqual(ac.after_hours_movers(None), [])


SCRIPT = '''
import datetime as dt
import streamlit as st
from ui.after_close import render_after_close
render_after_close(now=dt.datetime.fromisoformat(st.session_state["now"]))
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CardTests(unittest.TestCase):
    def render(self, tier, now=MON_8PM_ET, df=MoversTests.DF):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["now"] = now.isoformat()
        at.session_state["tier_key"] = tier
        at.session_state["entitlements"] = {"can_day_trader": tier in ("pro", "premium")}
        patches = [mock.patch("db.runs.list_runs", return_value=RUNS),
                   mock.patch("ui.market_scans.safe_run_df", return_value=df),
                   mock.patch("ui.headline_score.hsf_scores_by_ticker", return_value={"BBB": 61}),
                   mock.patch("streamlit.page_link")]
        started = [p.start() for p in patches]
        self.addCleanup(mock.patch.stopall)
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at, started[0]

    def text(self, at):
        return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)

    def test_pro_sees_movers_as_of_the_scan(self):
        at, _ = self.render("pro")
        t = self.text(at)
        self.assertIn("After the close", t)
        self.assertIn("**BBB** -10.00% after hours ($18.00) · HSF Score **61**", t)
        self.assertIn("**AAA** +5.00% after hours ($10.50)", t)
        self.assertIn("As of the 5:35 PM ET postmarket scan", t)
        self.assertNotIn("CCC", t)

    def test_free_sees_a_pro_note_and_no_data_is_loaded(self):
        at, list_runs = self.render("basic")
        self.assertIn("Pro shows tonight's biggest after-hours movers", self.text(at))
        self.assertNotIn("BBB", self.text(at))
        list_runs.assert_not_called()

    def test_hidden_during_market_hours(self):
        at, _ = self.render("pro", now=TUE_NOON_ET)
        self.assertNotIn("After the close", self.text(at))

    def test_scan_without_after_hours_columns(self):
        at, _ = self.render("pro", df=pd.DataFrame([{"Ticker": "A", "Last": 1.0}]))
        self.assertIn("No after-hours moves recorded in the 5:35 PM ET postmarket scan", self.text(at))


class WiringTests(unittest.TestCase):
    def test_today_shows_the_card_after_top_setups(self):
        src = open("ui/today.py").read()
        self.assertLess(src.index('("top setups"'), src.index('("after the close", _section_after_close)'))
        self.assertIn("from ui.after_close import render_after_close", src)


if __name__ == "__main__":
    unittest.main()
