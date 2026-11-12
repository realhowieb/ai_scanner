"""Shaping for the web Admin AI Confidence card (db.ai_confidence_admin)."""
import datetime as dt
import unittest

from db import ai_confidence_admin as A

NOW = dt.datetime(2026, 11, 12, 20, 0, tzinfo=dt.timezone.utc)


class AiConfidenceAdminTests(unittest.TestCase):
    def test_tier_cut_points_match_the_old_admin(self):
        self.assertEqual([A.tier(v) for v in (None, 10, 25, 39.9, 40)], ["Unknown", "Low", "Medium", "Medium", "High"])

    def test_live_buckets_report_unclear_signals_as_a_hit_rate_range(self):
        rows = [[4, 50, 44.1, 15, 30], [9, 4, 93.0, 4, 0], [2, 0, None, 0, 0]]
        out = A.live_buckets(rows)
        self.assertEqual([b["bucket"] for b in out], ["40-50%", "90-100%"])
        self.assertEqual((out[0]["unclear"], out[0]["hit_rate_low"], out[0]["hit_rate_high"]), (5, 30.0, 40.0))
        self.assertEqual((out[1]["hit_rate_low"], out[1]["hit_rate_high"]), (100.0, 100.0))

    def test_model_card_without_a_database_model_warns(self):
        card = A.model_card(None, [], NOW)
        self.assertIsNone(card["model"])
        self.assertIn("No active AI Confidence model", card["warnings"][0])

    def test_model_card_flags_missing_calibration_stale_model_and_version_mismatch(self):
        row = ["xgb-v7", dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc), None, None,
               "path_first_hit_v1: +4% before -2%", [{"bucket": "20-30%", "n": 10, "mean_confidence": 0.25, "hit_rate": 0.2}],
               None, 0, 6]
        card = A.model_card(row, ["ai-confidence-xgb-v1"], NOW)
        m = card["model"]
        self.assertEqual((m["model_version"], m["features"], m["calibration_method"]), ("xgb-v7", 6, None))
        self.assertEqual(m["training_buckets"], [{"bucket": "20-30%", "signals": 10, "mean_confidence": 0.25, "hit_rate": 0.2}])
        text = " ".join(card["warnings"])
        self.assertIn("no calibration map", text)
        self.assertIn("days ago", text)
        self.assertIn("ai-confidence-xgb-v1", text)

    def test_calibrated_current_model_has_no_warnings(self):
        row = ["xgb-v7", NOW - dt.timedelta(days=10), None, "ds1", "t", [], "isotonic", 12, 6]
        card = A.model_card(row, ["xgb-v7"], NOW)
        self.assertEqual(card["warnings"], [])
        self.assertEqual(card["model"]["calibration_method"], "isotonic")

    def test_queries_never_select_the_model_file(self):
        for sql in (A._MODEL_SQL, A._LATEST_SQL, A._LIVE_SQL, A._TICKER_SQL):
            self.assertNotIn("model_bytes", sql)


if __name__ == "__main__":
    unittest.main()
