"""Run 43 — Historical Replay & Signal Timeline tests (point-in-time integrity)."""
import unittest

from analytics import replay


def _obs(ts, *, scanners=None, indicators=None, prob=None, source="scheduled",
         symbol="NVDA", stale=False, fc=1.0):
    models = {"prebreakout": {"probability": prob}} if prob is not None else {}
    return {
        "symbol": symbol, "timestamp": ts, "scan_timestamp": ts,
        "session": "morning",
        "indicators": indicators if indicators is not None else {},
        "models": models,
        "scanners": scanners if scanners is not None else [],
        "market_context": {"source": source, "market_regime": "bullish"},
        "data_quality": {"feature_completeness": fc, "fallback_used": False, "stale": stale},
    }


def _t(h, m):
    return f"2026-09-22T{h:02d}:{m:02d}:00Z"


class ConstructionTests(unittest.TestCase):
    def test_ordering_and_symbol_date_filter(self):
        obs = [_obs(_t(10, 30)), _obs(_t(10, 0)), _obs(_t(10, 15), symbol="AMD"),
               _obs("2026-09-21T10:00:00Z")]
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        tss = [v["timestamp"] for v in s["observations"]]
        self.assertEqual(tss, sorted(tss))            # sorted ascending
        self.assertEqual(len(s["observations"]), 2)   # AMD + other date excluded

    def test_source_filtering_default_scheduled(self):
        obs = [_obs(_t(10, 0), source="scheduled"), _obs(_t(10, 5), source="manual")]
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        self.assertEqual(len(s["observations"]), 1)  # manual excluded by default

    def test_stale_excluded_unless_allowed(self):
        obs = [_obs(_t(10, 0)), _obs(_t(10, 5), stale=True)]
        self.assertEqual(len(replay.build_replay_session("NVDA", "2026-09-22", obs)["observations"]), 1)
        self.assertEqual(len(replay.build_replay_session("NVDA", "2026-09-22", obs,
                                                         allow_stale=True)["observations"]), 2)

    def test_malformed_observation_isolated(self):
        obs = [_obs(_t(10, 0)), {"symbol": "NVDA", "timestamp": "garbage"},
               None, _obs(_t(10, 5))]
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        self.assertEqual(len(s["observations"]), 2)  # bad ones skipped, rest fine


class EventTests(unittest.TestCase):
    def _session(self):
        obs = [
            _obs(_t(9, 45)),  # no setup
            _obs(_t(10, 5), scanners=[{"name": "prebreakout", "direction": "long"}],
                 prob=64, indicators={"rvol": 1.5, "vs_vwap_pct": -0.1}),
            _obs(_t(10, 20), scanners=[{"name": "prebreakout", "direction": "long"}],
                 prob=64, indicators={"rvol": 2.6, "vs_vwap_pct": 0.5}),  # VWAP up + RVOL
            _obs(_t(10, 35), scanners=[{"name": "prebreakout", "direction": "long"},
                                       {"name": "unusual_vol", "direction": "long"}],
                 prob=78, indicators={"rvol": 3.0, "vs_vwap_pct": 0.6}),  # scanner added
            _obs(_t(11, 20), scanners=[{"name": "prebreakout", "direction": "long"}],
                 prob=70, indicators={"rvol": 2.0, "vs_vwap_pct": -0.4}),  # lost VWAP
        ]
        return replay.build_replay_session("NVDA", "2026-09-22", obs)

    def test_setup_appeared_and_scanner_added(self):
        types = {e["event_type"] for e in self._session()["events"]}
        self.assertIn("SETUP_APPEARED", types)
        self.assertIn("SCANNER_ADDED", types)
        self.assertIn("VWAP_CROSS_UP", types)
        self.assertIn("VWAP_CROSS_DOWN", types)

    def test_importance_present(self):
        events = self._session()["events"]
        self.assertTrue(any(e["importance"] == "MAJOR" for e in events))
        self.assertTrue(all(e["importance"] in ("MAJOR", "NOTABLE", "INFO") for e in events))

    def test_stable_period_collapsed(self):
        obs = [_obs(_t(10, 0), scanners=[{"name": "momentum", "direction": "long"}],
                    indicators={"rvol": 2.0, "chg_pct": 1.2, "vs_vwap_pct": 0.3})]
        # three identical stable obs after the first
        for m in (5, 10, 15):
            obs.append(_obs(_t(10, m), scanners=[{"name": "momentum", "direction": "long"}],
                            indicators={"rvol": 2.0, "chg_pct": 1.2, "vs_vwap_pct": 0.3}))
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        stable_spans = [t for t in s["timeline"] if t["kind"] == "stable"]
        self.assertTrue(stable_spans)  # the quiet run collapsed into a span


class PointInTimeIntegrityTests(unittest.TestCase):
    def test_no_future_leakage_regression(self):
        # observations at 10:00 / 10:15 / 10:30 / 10:45; the future ones carry
        # DISTINCTIVE signals that must NOT appear when replaying 10:15.
        obs = [
            _obs(_t(10, 0), scanners=[{"name": "prebreakout", "direction": "long"}],
                 prob=60, indicators={"rvol": 1.4, "vs_vwap_pct": 0.1, "adx": 18}),
            _obs(_t(10, 15), scanners=[{"name": "prebreakout", "direction": "long"}],
                 prob=63, indicators={"rvol": 1.6, "vs_vwap_pct": 0.2, "adx": 19}),
            # FUTURE: adds unusual_vol, big prob jump, flips direction, ADX 40
            _obs(_t(10, 30), scanners=[{"name": "prebreakout", "direction": "long"},
                                       {"name": "unusual_vol", "direction": "long"}],
                 prob=91, indicators={"rvol": 5.0, "vs_vwap_pct": 3.0, "adx": 40}),
            _obs(_t(10, 45), scanners=[{"name": "gap_down", "direction": "short"}],
                 prob=20, indicators={"rvol": 0.5, "vs_vwap_pct": -3.0, "adx": 41}),
        ]
        session = replay.build_replay_session("NVDA", "2026-09-22", obs)
        v = replay.state_at(session, _t(10, 15))
        self.assertIsNotNone(v)
        blob = str(v)
        # nothing from 10:30 / 10:45 may appear anywhere in the 10:15 state
        self.assertNotIn("unusual_vol", blob.lower())
        self.assertNotIn("Unusual Volume", blob)
        self.assertNotIn("gap_down", blob.lower())
        self.assertEqual(v["direction"], "bullish")          # not the future short
        self.assertEqual(v["_feature_snapshot"]["adx"], 19)  # 10:15 value, not 40/41
        self.assertEqual(v["scanner_count"], 1)              # not 2
        self.assertNotIn("91", str(v["scores"]))             # not the future prob
        # priority/lifecycle also reflect only <=10:15
        self.assertIn(v["alert_priority"], ("LOW", "MEDIUM"))

    def test_state_at_returns_last_leq(self):
        obs = [_obs(_t(10, 0)), _obs(_t(10, 15)), _obs(_t(10, 30))]
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        self.assertEqual(replay.state_at(s, _t(10, 20))["timestamp"],
                         "2026-09-22T10:15:00+00:00")
        self.assertIsNone(replay.state_at(s, _t(9, 0)))  # before first obs


class OutcomeSeparationTests(unittest.TestCase):
    def test_outcomes_separate_and_never_in_state(self):
        obs = [_obs(_t(10, 0), scanners=[{"name": "prebreakout", "direction": "long"}],
                    prob=64, indicators={"rvol": 2.0})]
        outcomes = {"2026-09-22T10:00:00+00:00": {
            "+5m": {"raw_return": 0.004, "data_status": "MATURED"},
            "+15m": {"raw_return": 0.009, "data_status": "MATURED"},
            "mfe": 0.018, "mae": -0.003}}
        s = replay.build_replay_session("NVDA", "2026-09-22", obs, outcomes=outcomes)
        self.assertTrue(s["outcomes_available"])
        # the reconstructed view must NOT contain outcome numbers
        v = s["observations"][0]
        self.assertNotIn("raw_return", str(v))
        self.assertNotIn("0.018", str(v))
        # outcomes retrievable separately
        oc = replay.outcomes_at(s, _t(10, 0))
        self.assertEqual(oc["horizons"]["+5m"]["raw_return"], 0.004)
        self.assertEqual(oc["horizons"]["+30m"]["status"], "PENDING")  # not estimated
        self.assertEqual(oc["mfe"], 0.018)

    def test_no_outcomes_is_honest(self):
        obs = [_obs(_t(10, 0), scanners=[{"name": "momentum", "direction": "long"}],
                    indicators={"chg_pct": 1.5})]
        s = replay.build_replay_session("NVDA", "2026-09-22", obs)
        self.assertFalse(s["outcomes_available"])
        self.assertEqual(replay.outcomes_at(s, _t(10, 0))["horizons"]["+5m"]["status"],
                         "PENDING")


class SummaryDatesTests(unittest.TestCase):
    def test_summary_descriptive_only(self):
        s = replay.build_replay_session("NVDA", "2026-09-22", [
            _obs(_t(10, 0)),
            _obs(_t(10, 5), scanners=[{"name": "prebreakout", "direction": "long"}], prob=70)])
        summ = s["summary"]
        self.assertEqual(summ["observations"], 2)
        self.assertIsNotNone(summ["first_setup"])
        for banned in ("best", "profit", "entry", "buy"):
            self.assertNotIn(banned, str(summ).lower())

    def test_available_dates(self):
        obs = [_obs(_t(10, 0)), _obs("2026-09-21T10:00:00Z"),
               _obs(_t(10, 5), source="manual")]
        self.assertEqual(replay.available_dates(obs, "NVDA"),
                         ["2026-09-21", "2026-09-22"])  # manual-only date still counted? no

    def test_empty_session(self):
        s = replay.build_replay_session("ZZZ", "2026-09-22", [])
        self.assertEqual(s["observations"], [])
        self.assertEqual(s["events"], [])
        self.assertIsNone(replay.state_at(s, _t(10, 0)))


class DeterminismTests(unittest.TestCase):
    def test_deterministic(self):
        obs = [_obs(_t(10, 0), scanners=[{"name": "prebreakout", "direction": "long"}], prob=70)]
        self.assertEqual(replay.build_replay_session("NVDA", "2026-09-22", obs),
                         replay.build_replay_session("NVDA", "2026-09-22", obs))


if __name__ == "__main__":
    unittest.main()
