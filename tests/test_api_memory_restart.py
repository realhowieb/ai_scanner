import datetime as dt
import os
import unittest
from unittest import mock

from api import memory

UTC = dt.timezone.utc
TUE_0330_ET = dt.datetime(2026, 11, 10, 8, 30, tzinfo=UTC)  # 03:30 EST


class RestartDueTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("HSF_MEMORY_RESTART_MB", None)

    def test_high_memory_in_the_window_restarts(self):
        self.assertTrue(memory.restart_due(TUE_0330_ET, 450.0, 7200, 0))

    def test_below_the_limit_does_nothing(self):
        self.assertFalse(memory.restart_due(TUE_0330_ET, 400.0, 7200, 0))

    def test_outside_the_window_does_nothing(self):
        self.assertFalse(memory.restart_due(TUE_0330_ET + dt.timedelta(hours=1), 450.0, 7200, 0))
        self.assertFalse(memory.restart_due(TUE_0330_ET - dt.timedelta(hours=1), 450.0, 7200, 0))

    def test_weekends_are_left_alone(self):
        sat = TUE_0330_ET + dt.timedelta(days=4)
        self.assertFalse(memory.restart_due(sat, 450.0, 7200, 0))

    def test_a_fresh_process_or_a_running_scan_waits(self):
        self.assertFalse(memory.restart_due(TUE_0330_ET, 450.0, 600, 0))
        self.assertFalse(memory.restart_due(TUE_0330_ET, 450.0, 7200, 1))

    def test_zero_turns_it_off_and_the_limit_is_configurable(self):
        with mock.patch.dict(os.environ, {"HSF_MEMORY_RESTART_MB": "0"}):
            self.assertFalse(memory.restart_due(TUE_0330_ET, 500.0, 7200, 0))
        with mock.patch.dict(os.environ, {"HSF_MEMORY_RESTART_MB": "460"}):
            self.assertFalse(memory.restart_due(TUE_0330_ET, 450.0, 7200, 0))

    def test_unknown_memory_does_nothing(self):
        self.assertFalse(memory.restart_due(TUE_0330_ET, None, 7200, 0))

    def test_hook_signals_itself_only_when_due(self):
        with mock.patch.object(memory, "container_mb", return_value=450.0), \
             mock.patch.object(memory, "_STARTED", memory.time.monotonic() - 7200), \
             mock.patch.object(memory.os, "kill") as kill:
            self.assertTrue(memory.maybe_restart_for_memory(TUE_0330_ET))
            kill.assert_called_once()
            kill.reset_mock()
            self.assertFalse(memory.maybe_restart_for_memory(TUE_0330_ET + dt.timedelta(hours=3)))
            kill.assert_not_called()


try:
    import fastapi  # noqa: F401
    HAVE_FASTAPI = True
except ImportError:  # the core-dependency job runs without the API's packages
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "needs the API's dependencies")
class MemoryLogTests(unittest.TestCase):
    def test_memory_lines_reach_stdout(self):
        import logging

        import api.main  # noqa: F401  (configures the logger)

        lg = logging.getLogger("hsf.memory")
        self.assertTrue(lg.handlers)
        self.assertTrue(lg.isEnabledFor(logging.INFO))


class BriefPicksMemoTests(unittest.TestCase):
    def setUp(self):
        from ui import market_brief

        self.mb = market_brief
        market_brief._PICKS_MEMO.update(key=None, picks=[])

    def _df(self, run_id, n=3):
        import pandas as pd

        df = pd.DataFrame({"Ticker": ["A", "B", "C"][:n]})
        df.attrs.update(hsf_scan_run_id=run_id, hsf_scan_saved_at="2026-11-10T15:00:00Z")
        return df

    def test_scores_once_per_scan_and_returns_copies(self):
        score = mock.Mock(return_value=[{"symbol": "A", "prob": 9.0}])
        first = self.mb._cached_prebreakout_picks(self._df(1), score)
        first[0]["symbol"] = "A ⚠️E1d"  # the Brief edits picks in place
        second = self.mb._cached_prebreakout_picks(self._df(1), score)
        self.assertEqual(score.call_count, 1)
        self.assertEqual(second[0]["symbol"], "A")

    def test_new_scan_rescores(self):
        score = mock.Mock(return_value=[{"symbol": "A", "prob": 9.0}])
        self.mb._cached_prebreakout_picks(self._df(1), score)
        self.mb._cached_prebreakout_picks(self._df(2), score)
        self.assertEqual(score.call_count, 2)

    def test_empty_answers_are_not_remembered(self):
        score = mock.Mock(return_value=[])
        self.mb._cached_prebreakout_picks(self._df(1), score)
        self.mb._cached_prebreakout_picks(self._df(1), score)
        self.assertEqual(score.call_count, 2)

    def test_unknown_run_is_never_cached(self):
        score = mock.Mock(return_value=[{"symbol": "A", "prob": 9.0}])
        self.mb._cached_prebreakout_picks(self._df(None), score)
        self.mb._cached_prebreakout_picks(self._df(None), score)
        self.assertEqual(score.call_count, 2)


if __name__ == "__main__":
    unittest.main()
