"""Run 36 — canonical HSF observation/outcome schema + persistence tests."""
import sqlite3
import unittest

from analytics import hsf_observation as obs
from db import hsf_observations as store


def _obs(**kw):
    base = dict(
        symbol="nvda", timestamp="2026-08-25T13:35:00Z", context="scanner",
        session="regular", universe_version="sp500-2026-08",
        market={"price": 131.0, "previous_close": 129.0, "open": 129.5,
                "high": 132.0, "low": 129.0, "volume": 5_000_000},
        indicators={"rvol": 2.4, "adx": 31, "vwap": 130.0, "vs_vwap_pct": 0.8,
                    "supertrend_direction": "green", "ewo": 12.4, "gap_pct": 0.4,
                    "chg_pct": 1.5},
        scanners=[{"name": "breakout", "version": "b1", "triggered": True, "score": 9.1},
                  {"name": "day_trader", "version": "run32-v1", "score": 58.0,
                   "direction": "bullish", "quality": "developing"}],
        models={"prebreakout": {"version": "prebreakout-xgb-v16", "probability": 0.27},
                "ai_confidence": {"version": "ai-confidence-xgb-v1", "confidence": 0.61}},
        data_source="alpaca_iex",
    )
    base.update(kw)
    return obs.build_observation(**base)


class SchemaTests(unittest.TestCase):
    def test_creation_required_identifiers_and_schema_version(self):
        o = _obs()
        self.assertEqual(o["schema_version"], obs.OBSERVATION_SCHEMA_VERSION)
        self.assertTrue(o["observation_id"])
        self.assertEqual(o["symbol"], "NVDA")  # normalized
        self.assertEqual(o["timestamp"], "2026-08-25T13:35:00Z")

    def test_observation_id_deterministic_and_context_scoped(self):
        a = obs.make_observation_id("NVDA", "T1", "scanner")
        b = obs.make_observation_id("nvda", "T1", "scanner")
        c = obs.make_observation_id("NVDA", "T1", "market_brief")
        self.assertEqual(a, b)           # symbol case-insensitive
        self.assertNotEqual(a, c)        # context changes the id

    def test_multiple_scanner_triggers_preserved(self):
        o = _obs()
        names = [s["name"] for s in o["scanners"]]
        self.assertEqual(names, ["breakout", "day_trader"])
        self.assertEqual(len(o["scanners"]), 2)

    def test_model_versions_preserved(self):
        o = _obs()
        self.assertEqual(o["models"]["prebreakout"]["version"], "prebreakout-xgb-v16")
        self.assertEqual(o["models"]["ai_confidence"]["version"], "ai-confidence-xgb-v1")
        self.assertEqual(o["versions"]["dt_score"], "run32-v1")
        self.assertIn("schema", o["versions"])

    def test_missing_features_flagged_not_zero_filled(self):
        # Only pure-intraday fields (no daily-derived: adx/supertrend/ewo/gap/rvol)
        o = _obs(indicators={"vs_vwap_pct": 0.5, "chg_pct": 1.0})
        dq = o["data_quality"]
        self.assertNotIn("adx", o["indicators"])          # absent, not 0
        self.assertIn("adx", dq["missing_fields"])
        self.assertLess(dq["feature_completeness"], 1.0)
        self.assertTrue(dq["fallback_used"])              # no daily-derived present
        self.assertIn("daily-derived", dq["fallback_reason"])

    def test_full_feature_not_flagged_fallback(self):
        o = _obs()  # has adx/supertrend/ewo/gap/rvol present
        self.assertFalse(o["data_quality"]["fallback_used"])
        self.assertGreaterEqual(o["data_quality"]["feature_completeness"], 0.75)

    def test_nan_and_blank_treated_as_missing(self):
        o = _obs(market={"price": float("nan"), "previous_close": 129.0},
                 indicators={"supertrend_direction": "  "})
        self.assertNotIn("price", o["market"])
        self.assertNotIn("supertrend_direction", o["indicators"])


class OutcomeTests(unittest.TestCase):
    def test_attach_outcome_does_not_mutate_features(self):
        o = _obs()
        before_market = dict(o["market"])
        before_scanners = list(o["scanners"])
        oc = obs.build_outcome(
            observation_id=o["observation_id"], symbol="NVDA",
            observation_timestamp=o["timestamp"], horizon="+15m",
            evaluation_time="2026-08-25T13:50:00Z", raw_return=0.012,
            directional_return=0.012, mfe=0.02, mae=-0.005, hit=True)
        merged = obs.attach_outcome(o, oc)
        self.assertEqual(merged["outcomes"]["+15m"]["raw_return"], 0.012)
        # original untouched
        self.assertNotIn("outcomes", o)
        self.assertEqual(o["market"], before_market)
        self.assertEqual(o["scanners"], before_scanners)

    def test_no_lookahead_leakage_rejected(self):
        o = _obs()
        with self.assertRaises(ValueError):
            obs.build_outcome(
                observation_id=o["observation_id"], symbol="NVDA",
                observation_timestamp=o["timestamp"], horizon="+15m",
                evaluation_time=o["timestamp"])  # same instant → lookahead
        with self.assertRaises(ValueError):
            obs.build_outcome(
                observation_id=o["observation_id"], symbol="NVDA",
                observation_timestamp="2026-08-25T13:35:00Z", horizon="+15m",
                evaluation_time="2026-08-25T13:30:00Z")  # earlier → lookahead

    def test_pending_outcome_allows_no_eval_time(self):
        o = _obs()
        oc = obs.build_outcome(observation_id=o["observation_id"], symbol="NVDA",
                               observation_timestamp=o["timestamp"], horizon="+60m",
                               evaluation_time=None, data_status="PENDING")
        self.assertEqual(oc["data_status"], "PENDING")

    def test_attach_rejects_mismatched_observation(self):
        o = _obs()
        oc = obs.build_outcome(observation_id="deadbeef", symbol="NVDA",
                               observation_timestamp=o["timestamp"], horizon="+5m",
                               evaluation_time="2026-08-25T13:40:00Z")
        with self.assertRaises(ValueError):
            obs.attach_outcome(o, oc)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_roundtrip_and_duplicate_is_noop(self):
        o = _obs()
        self.assertTrue(store.save_observation(o, conn=self.conn))     # new
        self.assertFalse(store.save_observation(o, conn=self.conn))    # duplicate → no-op
        loaded = store.load_observation(o["observation_id"], conn=self.conn)
        self.assertEqual(loaded["symbol"], "NVDA")
        self.assertEqual(len(loaded["scanners"]), 2)
        self.assertEqual(loaded["models"]["prebreakout"]["version"], "prebreakout-xgb-v16")

    def test_outcome_persists_separately_and_first_wins(self):
        o = _obs()
        store.save_observation(o, conn=self.conn)
        oc = obs.build_outcome(observation_id=o["observation_id"], symbol="NVDA",
                               observation_timestamp=o["timestamp"], horizon="+15m",
                               evaluation_time="2026-08-25T13:50:00Z", raw_return=0.01)
        self.assertTrue(store.save_outcome(oc, conn=self.conn))
        oc2 = dict(oc, raw_return=0.99)
        self.assertFalse(store.save_outcome(oc2, conn=self.conn))  # first-write-wins
        loaded = store.load_observation(o["observation_id"], conn=self.conn)
        self.assertEqual(loaded["outcomes"]["+15m"]["raw_return"], 0.01)  # original kept
        # observation features are unchanged by outcome attachment
        self.assertEqual(loaded["market"]["price"], 131.0)

    def test_backward_compat_load_recent(self):
        for i in range(3):
            store.save_observation(_obs(timestamp=f"2026-08-25T13:3{i}:00Z"), conn=self.conn)
        recent = store.load_recent_observations(limit=10, conn=self.conn)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]["timestamp"], "2026-08-25T13:32:00Z")  # newest first

    def test_missing_id_is_safe(self):
        self.assertFalse(store.save_observation({}, conn=self.conn))
        self.assertFalse(store.save_outcome({"horizon": "+5m"}, conn=self.conn))


if __name__ == "__main__":
    unittest.main()
