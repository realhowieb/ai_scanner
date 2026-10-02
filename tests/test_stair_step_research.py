"""Outcome validation for the descriptive Day Trader Stair-stepper."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analytics.stair_step import WINDOW_OPTIONS
from analytics.stair_step_research import (
    CONTEXT,
    build_qualifying_observations,
    build_research_report,
    compute_stair_step_outcomes,
)
from db import hsf_observations as store
from scripts import mature_observations as worker
from scripts.analyze_stair_stepper import write_report

UTC = dt.timezone.utc
ANCHOR = dt.datetime(2026, 9, 25, 14, 0, tzinfo=UTC)  # 10:00 ET


def _metric_row(window=45, *, direction="up", at=ANCHOR, symbol="AAA"):
    sign = 1 if direction == "up" else -1
    return {
        "ticker": symbol,
        "status": "ok",
        "bars": window,
        "r2": 0.94,
        "trend_pct_per_hour": 1.2 * sign,
        "direction": direction,
        "max_pullback_pct": 0.2,
        "coverage": 1.0,
        "slope_per_minute": 0.02 * sign,
        "current_price": 100.0,
        "fitted_price": 99.99,
        "as_of": at,
    }


def _observation(*, direction="up", at=ANCHOR, window=45):
    rows = {candidate: [] for candidate in WINDOW_OPTIONS}
    rows[window] = [_metric_row(window, direction=direction, at=at)]
    return build_qualifying_observations(
        rows,
        r2_min=0.8,
        max_pullback_pct=1.0,
        min_trend_pct_per_hour=0.5,
    )[0]


def _bars(*, anchor=ANCHOR, direction="up", minutes=30):
    sign = 1 if direction == "up" else -1
    rows = []
    for minute in range(1, minutes + 1):
        close = 100.0 + sign * minute * 0.1
        rows.append({
            "t": (anchor + dt.timedelta(minutes=minute)).isoformat(),
            "o": close,
            "h": close + 0.05,
            "l": close - 0.05,
            "c": close,
            "v": 1_000,
        })
    return rows


class CaptureTests(unittest.TestCase):
    def test_all_supported_windows_are_captured_without_outcomes(self):
        rows = {window: [_metric_row(window)] for window in WINDOW_OPTIONS}
        observations = build_qualifying_observations(
            rows,
            r2_min=0.8,
            max_pullback_pct=1.0,
            min_trend_pct_per_hour=0.5,
            data_feed="iex",
        )
        self.assertEqual(
            [row["stair_step"]["window"] for row in observations],
            list(WINDOW_OPTIONS),
        )
        self.assertTrue(all(row["context"] == CONTEXT for row in observations))
        self.assertTrue(all("outcomes" not in row for row in observations))

    def test_dedupe_is_deterministic_per_window_direction_and_30_minute_bucket(self):
        first = _observation(at=ANCHOR + dt.timedelta(minutes=2))
        retry = _observation(at=ANCHOR + dt.timedelta(minutes=28))
        next_bucket = _observation(at=ANCHOR + dt.timedelta(minutes=31))
        other_window = _observation(at=ANCHOR + dt.timedelta(minutes=2), window=30)
        other_direction = _observation(at=ANCHOR + dt.timedelta(minutes=2), direction="down")
        self.assertEqual(first["observation_id"], retry["observation_id"])
        self.assertNotEqual(first["observation_id"], next_bucket["observation_id"])
        self.assertNotEqual(first["observation_id"], other_window["observation_id"])
        self.assertNotEqual(first["observation_id"], other_direction["observation_id"])


class OutcomeTests(unittest.TestCase):
    def test_horizons_use_only_strictly_future_bars_in_chronological_order(self):
        obs = _observation()
        bars = _bars()
        bars.append({"t": ANCHOR.isoformat(), "h": 999, "l": 1, "c": 999})
        outcomes = compute_stair_step_outcomes(obs, list(reversed(bars)))
        by_horizon = {row["horizon"]: row for row in outcomes}
        self.assertEqual(set(by_horizon), {"+5m", "+10m", "+15m", "+30m"})
        self.assertAlmostEqual(by_horizon["+5m"]["raw_return"], 0.005)
        self.assertEqual(by_horizon["+5m"]["source_bar_time"],
                         (ANCHOR + dt.timedelta(minutes=5)).isoformat())
        self.assertLess(by_horizon["+5m"]["future_high"], 200)

    def test_short_direction_adjusts_return_and_excursions(self):
        obs = _observation(direction="down")
        bars = _bars(direction="down")
        bars[1]["h"] = 101.0
        outcomes = compute_stair_step_outcomes(obs, bars, horizons=["+5m"])
        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0]["raw_return"], -0.005)
        self.assertAlmostEqual(outcomes[0]["directional_return"], 0.005)
        self.assertAlmostEqual(outcomes[0]["mfe"], 0.0055)
        self.assertAlmostEqual(outcomes[0]["mae"], -0.01)

    def test_mfe_and_mae_are_computed_per_horizon(self):
        obs = _observation()
        bars = _bars()
        bars[2]["l"] = 99.0
        bars[9]["h"] = 103.0
        outcomes = compute_stair_step_outcomes(obs, bars, horizons=["+5m", "+10m"])
        by_horizon = {row["horizon"]: row for row in outcomes}
        self.assertAlmostEqual(by_horizon["+5m"]["mae"], -0.01)
        self.assertAlmostEqual(by_horizon["+5m"]["mfe"], 0.0055)
        self.assertAlmostEqual(by_horizon["+10m"]["mfe"], 0.03)

    def test_missing_target_bar_is_missing_not_zero(self):
        obs = _observation()
        late = _bars(minutes=8)[7]
        self.assertEqual(
            compute_stair_step_outcomes(obs, [late], horizons=["+5m"]),
            [],
        )

    def test_market_close_never_crosses_into_after_hours(self):
        anchor = dt.datetime(2026, 9, 25, 19, 58, tzinfo=UTC)  # 15:58 ET
        obs = _observation(at=anchor)
        bars = _bars(anchor=anchor, minutes=10)
        self.assertEqual(
            compute_stair_step_outcomes(obs, bars, horizons=["+5m"]),
            [],
        )


class PersistenceAndWorkerTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_full_outcome_attachment_is_batched_and_preserves_values(self):
        obs = _observation()
        outcome = compute_stair_step_outcomes(obs, _bars(), horizons=["+5m"])[0]
        self.assertTrue(store.save_observation(obs, conn=self.conn))
        self.assertTrue(store.save_outcome(outcome, conn=self.conn))
        loaded = store.load_recent_observations(
            context=CONTEXT,
            attach_outcomes="full",
            conn=self.conn,
        )
        self.assertEqual(loaded[0]["outcomes"]["+5m"]["raw_return"], 0.005)

    def test_duplicate_refreshes_are_first_write_wins(self):
        obs = _observation()
        result = store.save_observations_batch([obs, obs], conn=self.conn)
        self.assertEqual(result["written"], 1)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(len(store.load_recent_observations(conn=self.conn)), 1)

    def test_scheduled_worker_uses_stair_horizons_and_same_session_math(self):
        obs = _observation()
        saved = []
        report = worker.mature_observations(
            [obs],
            now=ANCHOR + dt.timedelta(minutes=31),
            slack_min=0,
            fetch_bars=lambda _symbol, _date: _bars(),
            save_fn=lambda outcome: saved.append(outcome) or True,
            max_symbols=0,
            exclusion_reason=lambda _symbol: None,
            retire_after=None,
        )
        self.assertEqual({row["horizon"] for row in saved},
                         {"+5m", "+10m", "+15m", "+30m"})
        self.assertEqual(report["horizons"]["+10m"]["new"], 1)
        self.assertEqual(report["horizons"]["+60m"]["new"], 0)


class ReportTests(unittest.TestCase):
    def test_empty_report_is_honest_and_contains_every_window(self):
        report = build_research_report([])
        self.assertEqual(report["status"], "NO_OBSERVATIONS")
        self.assertEqual(report["best_window_verdict"]["status"],
                         "INSUFFICIENT_EVIDENCE")
        self.assertEqual([row["window"] for row in report["window_comparison"]],
                         list(WINDOW_OPTIONS))

    def test_report_artifacts_round_trip_as_strict_json(self):
        obs = _observation()
        obs["outcomes"] = {
            row["horizon"]: row for row in compute_stair_step_outcomes(obs, _bars())
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.analyze_stair_stepper.load_recent_observations",
            return_value=[obs],
        ):
            report = write_report(limit=100, out_dir=Path(directory))
            parsed = json.loads(
                (Path(directory) / "stair_stepper_validation.json").read_text()
            )
            self.assertEqual(parsed["data_coverage"]["matured_observations"], 1)
            self.assertEqual(report["window_comparison"][4]["window"], 45)
            self.assertTrue((Path(directory) / "stair_stepper_validation.md").exists())


if __name__ == "__main__":
    unittest.main()
