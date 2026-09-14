from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from integrations import automation_export as ae


class AutomationExportTests(unittest.TestCase):
    def _started(self) -> dt.datetime:
        return dt.datetime(2026, 9, 14, 14, 35, tzinfo=dt.timezone.utc)

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Last": 12.5,
                    "PctChange": 2.3,
                    "VolRel20": 1.7,
                    "Volume": 1200000,
                    "VolAvg20": 900000,
                    "BreakoutScore": 42.0,
                    "PreBreakoutProb": 0.62,
                    "AI Confidence": 55.0,
                    "PatternTag": "NearHigh",
                    "RSI14": 64.0,
                }
            ]
        )

    def test_snapshot_serialization_maps_known_candidate_fields(self) -> None:
        snapshot = ae.build_snapshot(
            self._frame(),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            completed_at_utc=self._started() + dt.timedelta(seconds=3),
            duration_seconds=3.2,
            symbols_requested=500,
            symbols_processed=498,
            symbols_skipped=2,
            model_metadata={"prebreakout": {"version": "v1"}, "ai_confidence": {"version": "v2"}},
            env={"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_SHA": "abc"},
        )

        self.assertEqual(snapshot["schema_version"], "1.0")
        self.assertEqual(snapshot["scan"]["run_id"], "github-123-2-scheduled-sp500")
        self.assertEqual(snapshot["summary"]["candidates_found"], 1)
        candidate = snapshot["candidates"][0]
        self.assertEqual(candidate["symbol"], "AAA")
        self.assertEqual(candidate["rank"], 1)
        self.assertEqual(candidate["prebreakout_ml_probability"], 0.62)
        self.assertEqual(candidate["breakout_ml_probability"], 0.55)
        json.dumps(snapshot, allow_nan=False)

    def test_missing_optional_values_serialize_as_null(self) -> None:
        snapshot = ae.build_snapshot(
            pd.DataFrame([{"Ticker": "BBB", "BreakoutScore": 10.0}]),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={},
        )

        candidate = snapshot["candidates"][0]
        self.assertIsNone(candidate["prebreakout_ml_probability"])
        self.assertIsNone(candidate["ema9"])

    def test_invalid_numbers_do_not_create_invalid_json(self) -> None:
        snapshot = ae.build_snapshot(
            pd.DataFrame([{"Ticker": "BAD", "Last": math.inf, "PctChange": math.nan, "Volume": -5}]),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={},
        )

        candidate = snapshot["candidates"][0]
        self.assertIsNone(candidate["price"])
        self.assertIsNone(candidate["percent_change"])
        self.assertIn("negative_volume", candidate["data_quality"]["warnings"])
        json.dumps(snapshot, allow_nan=False)

    def test_duplicate_symbols_are_diagnosed(self) -> None:
        snapshot = ae.build_snapshot(
            pd.DataFrame([{"Ticker": "DUP"}, {"Ticker": "DUP"}]),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={},
        )

        self.assertEqual(snapshot["diagnostics"]["duplicate_symbols"], ["DUP"])
        self.assertIn("duplicate_symbols_detected", snapshot["warnings"])

    def test_probability_and_rsi_validation(self) -> None:
        candidate = ae.candidate_from_row(
            {"Ticker": "XYZ", "PreBreakoutProb": 1.2, "AI Confidence": 140, "RSI14": 140},
            rank=1,
        )

        warnings = candidate["data_quality"]["warnings"]
        self.assertIn("prebreakout_ml_probability_out_of_range", warnings)
        self.assertIn("breakout_ml_probability_out_of_range", warnings)
        self.assertIn("rsi_out_of_range", warnings)

    def test_prebreakout_score_clustering_warning(self) -> None:
        rows = [{"Ticker": f"T{i}", "PreBreakoutScore": 7.77} for i in range(20)]
        snapshot = ae.build_snapshot(
            pd.DataFrame(rows),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={},
        )

        dist = snapshot["diagnostics"]["prebreakout_score_distribution"]
        self.assertEqual(dist["warning"], "suspicious_score_clustering")
        self.assertIn("prebreakout_score_distribution:suspicious_score_clustering", snapshot["warnings"])

    def test_atomic_write_failure_preserves_existing_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / ae.LATEST_FILENAME
            latest.write_text('{"previous": true}\n', encoding="utf-8")
            snapshot = ae.build_snapshot(
                self._frame(),
                universe="SP500",
                scan_type="scheduled",
                market_session="regular",
                started_at_utc=self._started(),
                model_metadata={},
                env={},
            )

            with patch.object(ae.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    ae.publish_snapshot(snapshot, root=root)

            self.assertEqual(json.loads(latest.read_text(encoding="utf-8")), {"previous": True})

    def test_publish_writes_latest_status_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = ae.build_snapshot(
                self._frame(),
                universe="COMBO",
                scan_type="scheduled",
                market_session="regular",
                started_at_utc=self._started(),
                completed_at_utc=self._started(),
                model_metadata={},
                env={"GITHUB_RUN_ID": "77"},
            )
            result = ae.publish_snapshot(snapshot, root=root)

            self.assertTrue(Path(result["latest_path"]).exists())
            self.assertTrue(Path(result["status_path"]).exists())
            self.assertTrue(Path(result["history_path"]).exists())
            status = json.loads((root / ae.STATUS_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "success")
            self.assertEqual(status["candidate_count"], 1)

    def test_model_metadata_is_accepted_when_available(self) -> None:
        snapshot = ae.build_snapshot(
            self._frame(),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={"prebreakout": {"version": "pb", "trained_at": "now"}},
            env={},
        )

        self.assertEqual(snapshot["models"]["prebreakout"]["version"], "pb")

    def test_exporter_works_without_github_environment(self) -> None:
        snapshot = ae.build_snapshot(
            self._frame(),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={},
        )

        self.assertTrue(snapshot["scan"]["run_id"].startswith("local-"))
        self.assertIsNone(snapshot["scan"]["github_run_id"])

    def test_explicit_empty_environment_ignores_ambient_github_variables(self) -> None:
        with patch.dict(os.environ, {"GITHUB_RUN_ID": "999", "GITHUB_SHA": "ambient-sha"}):
            snapshot = ae.build_snapshot(
                self._frame(),
                universe="SP500",
                scan_type="scheduled",
                market_session="regular",
                started_at_utc=self._started(),
                model_metadata={},
                env={},
            )

        self.assertTrue(snapshot["scan"]["run_id"].startswith("local-"))
        self.assertIsNone(snapshot["scan"]["github_run_id"])
        self.assertIsNone(snapshot["scan"]["git_sha"])

    def test_history_retention_removes_old_snapshot_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "history" / "2026-01-01"
            old_dir.mkdir(parents=True)
            (old_dir / "old.json").write_text("{}", encoding="utf-8")

            removed = ae.prune_history(
                root,
                retention_days=30,
                now=dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc),
            )

            self.assertEqual(len(removed), 1)
            self.assertFalse((old_dir / "old.json").exists())

    def test_sensitive_environment_values_are_not_exported(self) -> None:
        snapshot = ae.build_snapshot(
            self._frame(),
            universe="SP500",
            scan_type="scheduled",
            market_session="regular",
            started_at_utc=self._started(),
            model_metadata={},
            env={"DATABASE_URL": "postgres://secret", "GITHUB_SHA": "safe-sha"},
        )
        encoded = json.dumps(snapshot)

        self.assertNotIn("postgres://secret", encoded)
        self.assertIn("safe-sha", encoded)


if __name__ == "__main__":
    unittest.main()
