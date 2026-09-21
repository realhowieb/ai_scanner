"""Run 32 — deterministic Day Trader signal intelligence."""
import unittest

from analytics import day_trade_intel as di


def row(**kw):
    base = {"chg_pct": None, "gap_pct": None, "rvol": None, "vs_vwap_pct": None,
            "adx": None, "supertrend_direction": None, "ewo": None}
    base.update(kw)
    return base


class DirectionTests(unittest.TestCase):
    def test_strong_bullish(self):
        r = row(vs_vwap_pct=0.8, supertrend_direction="green", ewo=12.4,
                chg_pct=1.5, adx=31, rvol=2.4)
        intel = di.day_trade_intelligence(r)
        self.assertEqual(intel["direction"], "bullish")
        self.assertGreaterEqual(intel["score"], 65)
        self.assertEqual(intel["quality"], "strong")
        self.assertEqual(intel["conflicts"], [])

    def test_strong_bearish_high_score(self):
        r = row(vs_vwap_pct=-0.7, supertrend_direction="red", ewo=-9.0,
                chg_pct=-1.4, adx=29, rvol=2.1)
        intel = di.day_trade_intelligence(r)
        self.assertEqual(intel["direction"], "bearish")
        self.assertGreaterEqual(intel["score"], 65)  # bearish can score high
        self.assertEqual(intel["quality"], "strong")

    def test_adx_rvol_do_not_set_direction(self):
        # only ADX + RVOL present -> no directional signals -> neutral/insufficient
        r = row(adx=35, rvol=3.1)
        intel = di.day_trade_intelligence(r)
        self.assertEqual(intel["direction"], "neutral")
        self.assertEqual(intel["quality"], "insufficient")


class ConflictTests(unittest.TestCase):
    def test_mixed_above_vwap_supertrend_red(self):
        r = row(vs_vwap_pct=0.5, supertrend_direction="red", ewo=3.0, chg_pct=0.4, adx=14)
        intel = di.day_trade_intelligence(r)
        self.assertIn("Mixed trend signals", intel["conflicts"])
        # not a clean directional call -> reduced confidence
        self.assertIn(intel["quality"], ("developing", "weak"))

    def test_gap_fade(self):
        r = row(gap_pct=6.0, vs_vwap_pct=-0.4, chg_pct=-1.2, supertrend_direction="red", ewo=-2)
        intel = di.day_trade_intelligence(r)
        self.assertIn("Gap fading", intel["conflicts"])
        self.assertIn(intel["direction"], ("bearish", "neutral"))

    def test_momentum_disagreement(self):
        r = row(vs_vwap_pct=0.3, supertrend_direction="green", ewo=5.0, chg_pct=-0.6)
        self.assertIn("Momentum disagreement", di.day_trade_conflicts(r))

    def test_losing_vwap(self):
        r = row(vs_vwap_pct=-0.3, chg_pct=0.8, supertrend_direction="green", ewo=2)
        self.assertIn("Losing VWAP", di.day_trade_conflicts(r))


class ScorePenaltyTests(unittest.TestCase):
    def test_low_participation_penalized(self):
        aligned = dict(vs_vwap_pct=0.8, supertrend_direction="green", ewo=6, chg_pct=1.2, adx=28)
        strong = di.score_day_trade_setup(row(**aligned, rvol=2.5))
        weak = di.score_day_trade_setup(row(**aligned, rvol=0.6))
        self.assertLess(weak, strong)
        self.assertIn("Low participation", di.day_trade_conflicts(row(**aligned, rvol=0.6)))

    def test_weak_trend_penalized(self):
        aligned = dict(vs_vwap_pct=0.8, supertrend_direction="green", ewo=6, chg_pct=1.2, rvol=2.0)
        strong = di.score_day_trade_setup(row(**aligned, adx=30))
        weak = di.score_day_trade_setup(row(**aligned, adx=12))
        self.assertLess(weak, strong)
        self.assertIn("Weak trend strength", di.day_trade_conflicts(row(**aligned, adx=12)))


class MissingDataTests(unittest.TestCase):
    def test_missing_ewo_no_crash_no_fake_zero(self):
        r = row(vs_vwap_pct=0.6, supertrend_direction="green", chg_pct=1.0, adx=25, rvol=2.0)
        intel = di.day_trade_intelligence(r)  # EWO absent
        self.assertEqual(intel["direction"], "bullish")
        self.assertNotIn("EWO", " ".join(intel["reasons"]))  # not reported as 0

    def test_missing_adx_renormalizes(self):
        r = row(vs_vwap_pct=0.6, supertrend_direction="green", ewo=5, chg_pct=1.0, rvol=2.0)
        self.assertIsInstance(di.score_day_trade_setup(r), float)  # no crash, still scored

    def test_all_missing_is_insufficient(self):
        intel = di.day_trade_intelligence(row())
        self.assertEqual(intel["direction"], "neutral")
        self.assertEqual(intel["quality"], "insufficient")
        self.assertIsNone(intel["score"])


class CapTests(unittest.TestCase):
    def test_extreme_rvol_capped(self):
        base = dict(vs_vwap_pct=0.8, supertrend_direction="green", ewo=6, chg_pct=1.2, adx=28)
        s3 = di.score_day_trade_setup(row(**base, rvol=3.0))
        s20 = di.score_day_trade_setup(row(**base, rvol=20.0))
        self.assertEqual(s3, s20)  # rvol contribution capped at 3x

    def test_extreme_adx_capped(self):
        base = dict(vs_vwap_pct=0.8, supertrend_direction="green", ewo=6, chg_pct=1.2, rvol=2.0)
        s40 = di.score_day_trade_setup(row(**base, adx=40))
        s90 = di.score_day_trade_setup(row(**base, adx=90))
        self.assertEqual(s40, s90)  # adx contribution capped at 40


class ReasonsTests(unittest.TestCase):
    def test_reasons_present_only(self):
        r = row(vs_vwap_pct=0.8, supertrend_direction="green", ewo=12.4, adx=31, rvol=2.4)
        reasons = di.day_trade_signal_reasons(r)
        self.assertTrue(any("Above VWAP" in x for x in reasons))
        self.assertTrue(any("SuperTrend green" in x for x in reasons))
        self.assertTrue(any("confirms" in x for x in reasons))  # adx/rvol confirmation


if __name__ == "__main__":
    unittest.main()
