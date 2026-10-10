"""Day Trader web rows: bounded score, session columns, quote flags."""
import datetime as dt
import unittest

from analytics import day_trade_display as dd

NOW = dt.datetime(2026, 10, 10, 23, 0, tzinfo=dt.timezone.utc)


def row(**kw):
    base = {"ticker": "X", "last": 10.0, "previous_close": 9.0, "close_today": 10.0, "vwap": 9.8,
            "chg_pct": 11.11, "gap_pct": 2.0, "vs_vwap_pct": 2.04, "rvol": 2.0}
    base.update(kw)
    return base


class ScoreTests(unittest.TestCase):
    def test_score_is_bounded_even_for_huge_moves(self):
        r = dd.enrich_row(row(chg_pct=516.0, vs_vwap_pct=121.0, rvol=18.4, gap_pct=8.0), "open", NOW)
        self.assertLessEqual(r["day_trade_score"], 100)
        self.assertIn("Extreme move", r["quote_flags"])
        self.assertIn("Far from VWAP", r["quote_flags"])

    def test_insufficient_evidence_scores_none_and_ranks_after_scored(self):
        thin = dd.enrich_row({"ticker": "T", "last": 5.0, "rvol": 3.0}, "open", NOW)
        self.assertIsNone(thin["day_trade_score"])
        self.assertEqual(thin["dt_quality"], "insufficient")
        good = dd.enrich_row(row(), "open", NOW)
        flagged = dd.enrich_row(row(chg_pct=300.0), "open", NOW)
        ranked = sorted([flagged, thin, good], key=dd.rank_key)
        self.assertEqual([r is good for r in ranked], [True, False, False])
        self.assertIs(ranked[-1], flagged)


class SessionTests(unittest.TestCase):
    def test_after_hours_splits_session_and_extended_move(self):
        r = dd.enrich_row(row(last=13.0, chg_pct=44.44, vs_vwap_pct=32.65), "afterhours", NOW)
        self.assertEqual(r["session_chg_pct"], 11.11)
        self.assertEqual(r["ext_chg_pct"], 30.0)

    def test_off_hours_scores_the_completed_session(self):
        view = dd.scoring_view(row(last=13.0, chg_pct=44.44, vs_vwap_pct=32.65), "closed")
        self.assertEqual(view["chg_pct"], 11.11)
        self.assertEqual(view["vs_vwap_pct"], 2.04)

    def test_regular_session_has_no_split_columns(self):
        r = dd.enrich_row(row(), "open", NOW)
        self.assertIsNone(r["session_chg_pct"])
        self.assertIsNone(r["ext_chg_pct"])

    def test_no_ext_move_when_last_equals_close(self):
        self.assertIsNone(dd.session_changes(row(), "afterhours")["ext_chg_pct"])


class FlagTests(unittest.TestCase):
    def test_stale_quote_with_alpaca_nanoseconds(self):
        self.assertEqual(dd.quote_flags(row(trade_ts="2026-10-01T19:59:59.123456789Z"), NOW), ["Stale quote"])
        self.assertEqual(dd.quote_flags(row(trade_ts="2026-10-09T23:59:59.123456789Z"), NOW), [])

    def test_clean_quote_has_no_flags(self):
        self.assertEqual(dd.quote_flags(row(), NOW), [])


class SparklineTests(unittest.TestCase):
    def test_thins_to_points_and_keeps_ends(self):
        t0 = dt.datetime(2026, 10, 9, 14, 0, tzinfo=dt.timezone.utc)
        bars = [{"t": t0 + dt.timedelta(minutes=i), "c": float(i + 1)} for i in range(200)]
        s = dd.sparkline(bars, points=10)
        self.assertEqual(len(s), 10)
        self.assertEqual((s[0], s[-1]), (1.0, 200.0))


if __name__ == "__main__":
    unittest.main()
