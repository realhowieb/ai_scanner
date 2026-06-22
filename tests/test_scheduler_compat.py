from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scheduler.cron_runner import _load_universe
from scheduler import jobs


ROOT = Path(__file__).resolve().parents[1]


class SchedulerCompatTests(unittest.TestCase):
    def test_loads_file_backed_universes(self) -> None:
        self.assertGreater(len(_load_universe("SP500")), 0)
        self.assertGreater(len(_load_universe("NASDAQ")), 0)
        self.assertGreater(len(_load_universe("COMBO")), 0)

    def test_manual_job_exports_route_to_cron_runner(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_run_and_save(*args: object, **kwargs: object) -> bool:
            calls.append((args, kwargs))
            return True

        with patch.object(jobs, "run_and_save", side_effect=fake_run_and_save):
            self.assertEqual(jobs.run_sp500_now(), 0)
            self.assertEqual(jobs.run_nasdaq_now(), 0)
            self.assertEqual(jobs.run_premarket_now(universe="Nasdaq 100"), 0)
            self.assertEqual(jobs.run_postmarket({"universe_name": "sp500"}), 0)

        self.assertEqual(calls[0][0], ("SP500",))
        self.assertEqual(calls[1][0], ("NASDAQ",))
        self.assertEqual(calls[2][0], ("NASDAQ",))
        self.assertTrue(calls[2][1]["premarket"])
        self.assertEqual(calls[2][1]["profile"], "premarket")
        self.assertEqual(calls[3][0], ("SP500",))
        self.assertTrue(calls[3][1]["afterhours"])
        self.assertEqual(calls[3][1]["profile"], "postmarket")

    def test_manual_job_failure_returns_negative_one(self) -> None:
        with patch.object(jobs, "run_and_save", return_value=False):
            self.assertEqual(jobs.run_sp500_now(), -1)


class ScannerEntrypointSourceTests(unittest.TestCase):
    def test_ui_no_longer_discovers_removed_scan_runners(self) -> None:
        source = (ROOT / "ui" / "pages_main.py").read_text(encoding="utf-8")

        self.assertNotIn("run_sp500_scan", source)
        self.assertNotIn("run_nasdaq_scan", source)

    def test_breakout_keeps_legacy_adapter_to_current_scanner(self) -> None:
        source = (ROOT / "scan" / "breakout.py").read_text(encoding="utf-8")

        self.assertIn("def run_breakout_scan", source)
        self.assertIn("def breakout_scanner", source)
        self.assertIn("return run_breakout_scan(", source)


if __name__ == "__main__":
    unittest.main()
