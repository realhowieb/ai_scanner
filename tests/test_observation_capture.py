"""Run 38A — production observation capture + maturation tests."""
import sqlite3
import unittest

from analytics import observation_capture as oc
from db import hsf_observations as store


def _row(ticker="NVDA", **kw):
    base = {"Ticker": ticker, "BreakoutScore": 8.5, "Last": 131.0, "Volume": 5_000_000,
            "PctChange": 1.5, "GapPct": 0.8, "Trend20D%": 3.0, "Trend10D%": 2.0,
            "VolRel20": 2.4, "DollarVol20": 6.5e8, "Volatility20D%": 3.1,
            "IsBreakout": True, "PatternTag": "cup"}
    base.update(kw)
    return base


class TriggerTests(unittest.TestCase):
    def test_multi_scanner_capture(self):
        scanners = oc.derive_scanner_triggers(_row())
        names = {s["name"] for s in scanners}
        self.assertIn("breakout", names)
        self.assertIn("gap_up", names)          # GapPct 0.8 > 0
        self.assertIn("unusual_vol", names)     # VolRel20 2.4 >= 2
        self.assertIn("momentum", names)        # trends positive
        self.assertIn("breakout_only", names)   # IsBreakout

    def test_gap_down_short_direction(self):
        scanners = oc.derive_scanner_triggers(_row(GapPct=-1.2))
        gd = next(s for s in scanners if s["name"] == "gap_down")
        self.assertEqual(gd["direction"], "short")

    def test_no_unusual_vol_below_threshold(self):
        names = {s["name"] for s in oc.derive_scanner_triggers(_row(VolRel20=1.1))}
        self.assertNotIn("unusual_vol", names)


class BuildTests(unittest.TestCase):
    def test_build_observation_fields_and_source(self):
        obs = oc.build_scan_observations(
            [_row()], universe="SP500", scan_timestamp="2026-08-25T14:32:00Z",
            session="morning")
        self.assertEqual(len(obs), 1)
        o = obs[0]
        self.assertEqual(o["symbol"], "NVDA")
        self.assertEqual(o["context"], "scheduled:sp500")
        self.assertEqual(o["market_context"]["source"], "scheduled")
        self.assertEqual(o["indicators"]["gap_pct"], 0.8)   # mapped
        self.assertEqual(o["market"]["price"], 131.0)
        self.assertNotIn("adx", o["indicators"])            # absent, not invented
        self.assertGreaterEqual(len(o["scanners"]), 4)

    def test_timestamp_bucketing_dedupes_retries(self):
        a = oc.build_scan_observations([_row()], universe="SP500",
                                       scan_timestamp="2026-08-25T14:32:00Z")[0]
        b = oc.build_scan_observations([_row()], universe="SP500",
                                       scan_timestamp="2026-08-25T14:58:00Z")[0]
        self.assertEqual(a["observation_id"], b["observation_id"])  # same hour bucket
        self.assertEqual(a["timestamp"], "2026-08-25T14:00:00+00:00")

    def test_universe_scopes_id(self):
        a = oc.build_scan_observations([_row()], universe="SP500",
                                       scan_timestamp="2026-08-25T14:00:00Z")[0]
        b = oc.build_scan_observations([_row()], universe="NASDAQ",
                                       scan_timestamp="2026-08-25T14:00:00Z")[0]
        self.assertNotEqual(a["observation_id"], b["observation_id"])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_batch_persist_and_dedupe(self):
        rows = [_row(f"T{i}") for i in range(5)]
        s1 = oc.capture_scan_observations(rows, universe="SP500",
                                          scan_timestamp="2026-08-25T14:00:00Z",
                                          conn=self.conn)
        self.assertEqual(s1["written"], 5)
        self.assertEqual(s1["write_failures"], 0)
        self.assertEqual(s1["capture_health"], "HEALTHY")
        # rerun (workflow retry) → all duplicates, nothing written twice
        s2 = oc.capture_scan_observations(rows, universe="SP500",
                                          scan_timestamp="2026-08-25T14:00:00Z",
                                          conn=self.conn)
        self.assertEqual(s2["written"], 0)
        self.assertEqual(s2["duplicates"], 5)

    def test_dry_run_writes_nothing(self):
        rows = [_row(f"T{i}") for i in range(3)]
        s = oc.capture_scan_observations(rows, universe="SP500",
                                         scan_timestamp="2026-08-25T14:00:00Z",
                                         dry_run=True, conn=self.conn)
        self.assertEqual(s["would_write"], 3)
        self.assertEqual(s["written"], 0)
        self.assertEqual(store.load_recent_observations(conn=self.conn), [])

    def test_persistence_failure_is_isolated(self):
        # A broken connection must not raise out of capture (scan continues).
        class _BadConn:
            def cursor(self):
                raise RuntimeError("db down")
            def close(self):
                pass
        s = oc.capture_scan_observations([_row()], universe="SP500",
                                         scan_timestamp="2026-08-25T14:00:00Z",
                                         conn=_BadConn())
        self.assertEqual(s["written"], 0)
        self.assertGreaterEqual(s["write_failures"], 1)
        self.assertIn(s["capture_health"], ("FAILED", "DEGRADED"))

    def test_quality_metadata_and_control(self):
        cov = {"funnel": {"counts": {"eligible": 500, "price_success": 480}}}
        s = oc.capture_scan_observations([_row(f"T{i}") for i in range(4)],
                                         universe="SP500",
                                         scan_timestamp="2026-08-25T14:00:00Z",
                                         coverage=cov, conn=self.conn)
        self.assertEqual(s["eligible_symbols"], 500)
        self.assertEqual(s["control_denominator"]["non_trigger_estimate"], 480 - 4)
        # breakout rows lack adx/supertrend/ewo → flagged partial, never "complete"
        self.assertGreaterEqual(s["quality_distribution"]["partial"], 1)


class MaturationTests(unittest.TestCase):
    def _obs(self):
        return oc.build_scan_observations(
            [_row()], universe="SP500", scan_timestamp="2026-08-25T14:00:00Z")[0]

    def test_outcomes_no_lookahead_and_compute(self):
        obs = self._obs()
        prices = [100.0] + [100.0 + i * 0.1 for i in range(1, 61)]  # rising
        outs = oc.compute_matured_outcomes(
            obs, prices_after=prices, horizon_bars={"+5m": 5, "+15m": 15},
            evaluation_times={"+5m": "2026-08-25T14:05:00Z",
                              "+15m": "2026-08-25T14:15:00Z"})
        labels = {o["horizon"] for o in outs}
        self.assertEqual(labels, {"+5m", "+15m"})
        self.assertGreater(outs[0]["raw_return"], 0)
        self.assertEqual(outs[0]["data_status"], "MATURED")

    def test_lookahead_eval_before_obs_is_skipped(self):
        obs = self._obs()  # scan_timestamp 2026-08-25T14:00:00Z
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        outs = oc.compute_matured_outcomes(
            obs, prices_after=prices, horizon_bars={"+5m": 5},
            evaluation_times={"+5m": "2026-08-25T13:00:00Z"})  # BEFORE obs
        self.assertEqual(outs, [])  # guarded out

    def test_insufficient_future_prices(self):
        obs = self._obs()
        self.assertEqual(oc.compute_matured_outcomes(
            obs, prices_after=[100.0], horizon_bars={"+5m": 5}), [])


if __name__ == "__main__":
    unittest.main()
