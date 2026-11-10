"""Hourly refresh scans (CRON_RECORD_RESEARCH=0) update the Scanner but record no research data.

The ML/research data is built on the six scheduled slots (Run 61 buckets, the
duplicate-rate readiness check, the collection audit), so the extra 10:35-14:35 ET
scans must save the Scanner run without observations, cohorts or freezes."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None
ROOT = Path(__file__).resolve().parents[1]


def _frame():
    import pandas as pd

    return pd.DataFrame([{"Ticker": "AAA", "Price": 10.0, "BreakoutScore": 80.0}])


@unittest.skipUnless(_PANDAS, "pandas not installed")
class RefreshScanTests(unittest.TestCase):
    def _run(self, env):
        from scheduler import cron_runner

        saved = {}
        with mock.patch.dict("os.environ", env), \
                mock.patch.object(cron_runner, "_load_universe_result",
                                  return_value=(["AAA"], {"universe_source": "live"})), \
                mock.patch("data.tradability.filter_tradable_tickers", side_effect=lambda t: t), \
                mock.patch("scan.engine.run_breakout_scan", return_value=_frame()), \
                mock.patch("db.runs.save_run", side_effect=lambda **kw: saved.update(kw)), \
                mock.patch("db.runs.save_daily_snapshot"), \
                mock.patch("integrations.automation_export.publish_scan_results", return_value={}), \
                mock.patch.object(cron_runner, "_write_coverage_artifact"), \
                mock.patch.object(cron_runner, "_append_perf_history"), \
                mock.patch.object(cron_runner, "_score_prebreakout", side_effect=lambda df: df), \
                mock.patch("analytics.observation_capture.capture_scan_observations") as capture, \
                mock.patch.object(cron_runner, "_capture_research_cohorts") as cohorts:
            summary = cron_runner.run_and_save("US_MARKET")
        self.assertTrue(summary.ok, summary.error)
        return saved, capture, cohorts

    def test_refresh_scan_saves_the_run_but_records_no_observations(self):
        saved, capture, cohorts = self._run({"CRON_RECORD_RESEARCH": "0"})
        self.assertEqual([r["Ticker"] for r in json.loads(saved["results_json"])], ["AAA"])
        self.assertTrue(saved["name"].endswith("| refresh"))
        capture.assert_not_called()
        cohorts.assert_not_called()

    def test_scheduled_slot_still_records_observations(self):
        saved, capture, _ = self._run({"CRON_RECORD_RESEARCH": "1", "HSF_OBSERVATION_CAPTURE": "1"})
        self.assertNotIn("refresh", saved["name"])
        capture.assert_called_once()


class RefreshFlagTests(unittest.TestCase):
    def test_flag_defaults_to_recording(self):
        from scheduler import cron_runner

        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CRON_RECORD_RESEARCH", None)
            self.assertTrue(cron_runner._records_research())
        with mock.patch.dict("os.environ", {"CRON_RECORD_RESEARCH": "0"}):
            self.assertFalse(cron_runner._records_research())

    def test_refresh_scan_skips_the_opportunity_freeze(self):
        src = (ROOT / "scheduler" / "cron_runner.py").read_text()
        freeze = src.index("frozen = freeze_latest_opportunities()")
        self.assertIn("if not _records_research():", src[freeze - 300:freeze])

    def test_workflow_maps_the_input_to_the_flag(self):
        wf = (ROOT / ".github" / "workflows" / "scheduled-scans.yml").read_text()
        self.assertIn("record_research:", wf)
        self.assertIn("CRON_RECORD_RESEARCH: ${{ (github.event_name == 'workflow_dispatch' && "
                      "inputs.record_research == false) && '0' || '1' }}", wf)


if __name__ == "__main__":
    unittest.main()
