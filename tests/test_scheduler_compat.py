from __future__ import annotations

import unittest
import importlib
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scheduler.cron_runner import _load_universe, _skip_reason
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

    def test_market_time_gate_uses_new_york_dst(self) -> None:
        # June is EDT (UTC-4). A fixed UTC-5 conversion would incorrectly
        # treat this as 5 AM ET and skip.
        june_six_am_et = datetime(2026, 6, 22, 10, 0, tzinfo=timezone.utc)
        self.assertIsNone(_skip_reason(june_six_am_et))

    def test_market_time_gate_skips_weekends(self) -> None:
        saturday_midday_et = datetime(2026, 6, 20, 16, 0, tzinfo=timezone.utc)
        self.assertIn("Weekend", _skip_reason(saturday_midday_et) or "")

    def test_market_time_gate_skips_before_six_eastern(self) -> None:
        pre_six_am_et = datetime(2026, 6, 22, 9, 30, tzinfo=timezone.utc)
        self.assertIn("Too early", _skip_reason(pre_six_am_et) or "")


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

    def test_scan_ui_uses_eastern_time_for_market_diagnostics(self) -> None:
        source = (ROOT / "ui" / "scans.py").read_text(encoding="utf-8")

        self.assertNotIn("utcnow()", source)
        self.assertIn('ZoneInfo("America/New_York")', source)

    def test_project_uses_timezone_aware_utc_datetimes(self) -> None:
        forbidden = "datetime." + "utcnow()"
        for path in ROOT.rglob("*.py"):
            if any(part in {"__pycache__", ".venv", "venv"} for part in path.parts):
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(forbidden, source, str(path.relative_to(ROOT)))


class ConfigCompatTests(unittest.TestCase):
    def test_config_db_url_matches_database_url_env(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://example/db"}, clear=False):
            os.environ.pop("NEON_DATABASE_URL", None)
            os.environ.pop("DB_URL", None)
            import config

            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.SETTINGS.db_url, "postgresql://example/db")

    def test_config_db_url_prefers_neon_database_url_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NEON_DATABASE_URL": "postgresql://neon/db",
                "DATABASE_URL": "postgresql://generic/db",
            },
            clear=False,
        ):
            import config

            reloaded = importlib.reload(config)

        self.assertEqual(reloaded.SETTINGS.db_url, "postgresql://neon/db")

    def test_db_core_url_prefers_neon_database_url_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NEON_DATABASE_URL": "postgresql://core-neon/db",
                "DATABASE_URL": "postgresql://core-generic/db",
            },
            clear=False,
        ):
            spec = importlib.util.spec_from_file_location("db_core_under_test", ROOT / "db" / "core.py")
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            self.assertEqual(module._get_database_url(), "postgresql://core-neon/db")

    def test_app_db_helper_mentions_neon_database_url(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("NEON_DATABASE_URL", source)
        self.assertIn('st.secrets["neon"]["database_url"]', source)


if __name__ == "__main__":
    unittest.main()
