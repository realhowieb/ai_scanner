"""Scheduled scans save PreBreakoutProb% so the API can show `prob` to Premium.

The API Scanner and stock page read the saved cron run; before this, the cron
saved raw engine output and every scheduled scan had prob = null.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None


def _frame():
    import pandas as pd

    return pd.DataFrame([
        {"Ticker": "AAA", "Price": 10.0, "BreakoutScore": 80.0},
        {"Ticker": "BBB", "Price": 20.0, "BreakoutScore": 60.0},
        {"Ticker": "CCC", "Price": 30.0, "BreakoutScore": 40.0},
    ])


def _fake_score(df):
    # Probabilities in the opposite order to the rows: a re-rank would show.
    df = df.copy()
    df["PreBreakoutProb"] = [0.1, 0.5, 0.9]
    df["PreBreakoutProb%"] = [10.0, 50.0, 90.0]
    return df


@unittest.skipUnless(_PANDAS, "pandas not installed")
class ScorePrebreakoutHelperTests(unittest.TestCase):
    def _run(self, *, model, score=_fake_score):
        from scheduler import cron_runner

        with mock.patch("ml_prebreakout.load_prebreakout_model", return_value=model), \
                mock.patch("ml_prebreakout.score_prebreakout", side_effect=score):
            return cron_runner._score_prebreakout(_frame())

    def test_adds_probability_and_keeps_row_order(self):
        out = self._run(model={"model": object()})
        self.assertEqual(list(out["Ticker"]), ["AAA", "BBB", "CCC"])
        self.assertEqual(list(out["PreBreakoutProb%"]), [10.0, 50.0, 90.0])
        self.assertEqual(list(out["BreakoutScore"]), [80.0, 60.0, 40.0])

    def test_no_model_saves_unscored_not_zero(self):
        out = self._run(model=None)
        self.assertNotIn("PreBreakoutProb%", out.columns)

    def test_scoring_error_saves_unscored(self):
        def boom(df):
            raise ValueError("feature mismatch")

        out = self._run(model={"model": object()}, score=boom)
        self.assertNotIn("PreBreakoutProb%", out.columns)
        self.assertEqual(list(out["Ticker"]), ["AAA", "BBB", "CCC"])

    def test_empty_and_non_frame_results_pass_through(self):
        import pandas as pd

        from scheduler import cron_runner

        empty = pd.DataFrame()
        self.assertIs(cron_runner._score_prebreakout(empty), empty)
        rows = [{"Ticker": "AAA"}]
        self.assertIs(cron_runner._score_prebreakout(rows), rows)


@unittest.skipUnless(_PANDAS, "pandas not installed")
class RunAndSaveScoresBeforeSavingTests(unittest.TestCase):
    def test_saved_run_carries_prebreakout_probability(self):
        from scheduler import cron_runner

        saved = {}

        def fake_save_run(**kwargs):
            saved.update(kwargs)

        env = {"HSF_OBSERVATION_CAPTURE": "0", "HSF_RESEARCH_CAPTURE": "0"}
        with mock.patch.dict("os.environ", env), \
                mock.patch.object(cron_runner, "_load_universe_result",
                                  return_value=(["AAA", "BBB", "CCC"], {"universe_source": "live"})), \
                mock.patch("data.tradability.filter_tradable_tickers", side_effect=lambda t: t), \
                mock.patch("scan.engine.run_breakout_scan", return_value=_frame()), \
                mock.patch("db.runs.save_run", side_effect=fake_save_run), \
                mock.patch("db.runs.save_daily_snapshot"), \
                mock.patch("integrations.automation_export.publish_scan_results", return_value={}), \
                mock.patch.object(cron_runner, "_write_coverage_artifact"), \
                mock.patch.object(cron_runner, "_append_perf_history"), \
                mock.patch("ml_prebreakout.load_prebreakout_model", return_value={"model": object()}), \
                mock.patch("ml_prebreakout.score_prebreakout", side_effect=_fake_score):
            summary = cron_runner.run_and_save("US_MARKET")

        self.assertTrue(summary.ok, summary.error)
        rows = json.loads(saved["results_json"])
        self.assertEqual([r["Ticker"] for r in rows], ["AAA", "BBB", "CCC"])
        self.assertEqual([r["PreBreakoutProb%"] for r in rows], [10.0, 50.0, 90.0])
        self.assertEqual(saved["row_count"], 3)


if __name__ == "__main__":
    unittest.main()
