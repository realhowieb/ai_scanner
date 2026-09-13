"""Run 26 — HSF intelligence performance analysis (deterministic, read-only)."""
import unittest
from unittest import mock

from analytics import intelligence_performance as ip


def _band(key, rate, n=50):
    return {"key": key, "comparable": n, "follow_through_rate": rate, "assessment": "OK",
            "weakened": 0, "faded": 0, "dropped": 0, "recovered": 0}


def _status(key, rate, n=50):
    return {"key": key, "comparable": n, "follow_through_rate": rate, "assessment": "OK"}


def _horizon(key, rate, n=50):
    return {"key": key, "comparable": n, "follow_through_rate": rate, "assessment": "OK",
            "weakened": 0, "faded": 0, "dropped": 0, "recovered": 0}


def _opp(**kw):
    base = {"available": True, "matured": 0, "comparable": 0, "by_status": [], "by_score_band": [],
            "by_signal_count": [], "by_regime": [], "by_horizon": []}
    base.update(kw)
    return base


def _aq(**kw):
    base = {"available": True, "evaluable": 0, "by_event_type": [], "frequency": {}}
    base.update(kw)
    return base


class ReadinessTests(unittest.TestCase):
    def test_early_when_below_minimum(self):
        r = ip.evidence_readiness(_opp(comparable=4), _aq(evaluable=2))
        self.assertEqual(r["tier"], "EARLY")

    def test_developing_when_one_cohort(self):
        r = ip.evidence_readiness(_opp(comparable=12, by_status=[_status("STRONG", 0.7, 12)]),
                                  _aq(evaluable=0))
        self.assertEqual(r["tier"], "DEVELOPING")

    def test_sufficient_when_multiple_cohorts_and_horizons(self):
        opp = _opp(comparable=300,
                   by_horizon=[_horizon("NEXT", 0.7), _horizon("H24", 0.65), _horizon("H72", 0.6)],
                   by_status=[_status("STRONG", 0.7), _status("WATCH", 0.6)],
                   by_score_band=[_band("75+", 0.72)])
        r = ip.evidence_readiness(opp, _aq(evaluable=0))
        self.assertEqual(r["tier"], "SUFFICIENT")


class DecayTests(unittest.TestCase):
    def test_gradual_decay(self):
        # total drop 0.16 (< sharp 0.20, >= gradual 0.05), non-increasing
        opp = _opp(by_horizon=[_horizon("NEXT", 0.78), _horizon("H24", 0.72),
                               _horizon("H72", 0.66), _horizon("H120", 0.62)])
        self.assertEqual(ip.horizon_decay(opp)["state"], "GRADUAL_DECAY")

    def test_sharp_decay(self):
        opp = _opp(by_horizon=[_horizon("NEXT", 0.82), _horizon("H120", 0.40)])
        self.assertEqual(ip.horizon_decay(opp)["state"], "SHARP_DECAY")

    def test_stable(self):
        opp = _opp(by_horizon=[_horizon("NEXT", 0.70), _horizon("H120", 0.69)])
        self.assertEqual(ip.horizon_decay(opp)["state"], "STABLE")

    def test_insufficient(self):
        opp = _opp(by_horizon=[_horizon("NEXT", 0.70, n=3)])  # not ready
        self.assertEqual(ip.horizon_decay(opp)["state"], "INSUFFICIENT_SAMPLE")


class MonotonicityTests(unittest.TestCase):
    def test_monotonic(self):
        rows = {"75+": _band("75+", 0.74), "60-74": _band("60-74", 0.66),
                "50-59": _band("50-59", 0.54), "<50": _band("<50", 0.41)}
        self.assertEqual(ip._monotonicity(rows, ip._BAND_ORDER)["state"], "MONOTONIC")

    def test_mostly_monotonic_single_inversion(self):
        rows = {"75+": _band("75+", 0.61), "60-74": _band("60-74", 0.72),
                "50-59": _band("50-59", 0.48)}  # one meaningful inversion
        self.assertEqual(ip._monotonicity(rows, ip._BAND_ORDER)["state"], "MOSTLY_MONOTONIC")

    def test_non_monotonic_two_inversions(self):
        rows = {"75+": _band("75+", 0.50), "60-74": _band("60-74", 0.72),
                "50-59": _band("50-59", 0.60), "<50": _band("<50", 0.80)}
        self.assertEqual(ip._monotonicity(rows, ip._BAND_ORDER)["state"], "NON_MONOTONIC")

    def test_tiny_wiggle_is_mostly_or_monotonic(self):
        # 0.01 inversion is below MIN_MEANINGFUL_RATE_DELTA -> not an inversion
        rows = {"75+": _band("75+", 0.70), "60-74": _band("60-74", 0.71)}
        self.assertEqual(ip._monotonicity(rows, ip._BAND_ORDER)["state"], "MONOTONIC")

    def test_insufficient_single_band(self):
        rows = {"75+": _band("75+", 0.70)}
        self.assertEqual(ip._monotonicity(rows, ip._BAND_ORDER)["state"], "INSUFFICIENT_SAMPLE")


class FindingsTests(unittest.TestCase):
    def test_no_findings_is_valid(self):
        self.assertEqual(ip.derive_supported_findings({"opportunity": _opp()}), [])

    def test_score_band_finding_requires_meaningful_delta(self):
        # 72% vs 71% must NOT produce a finding
        summary = {"opportunity": _opp(by_score_band=[_band("75+", 0.72), _band("<50", 0.71)]),
                   "score_monotonicity": {"state": "MONOTONIC"}}
        self.assertEqual(ip.derive_supported_findings(summary), [])

    def test_score_band_finding_emitted(self):
        summary = {"opportunity": _opp(by_score_band=[_band("75+", 0.74, 142), _band("<50", 0.41, 201)]),
                   "score_monotonicity": {"state": "MONOTONIC"}}
        f = ip.derive_supported_findings(summary)
        self.assertTrue(any(x["dimension"] == "score_band" and "Higher HSF" in x["statement"] for x in f))

    def test_status_finding_strong_vs_watch(self):
        summary = {"opportunity": _opp(by_status=[_status("STRONG", 0.75, 120), _status("WATCH", 0.60, 140)])}
        f = ip.derive_supported_findings(summary)
        self.assertTrue(any(x["dimension"] == "status" for x in f))

    def test_non_monotonic_reports_mixed_not_cherry_pick(self):
        summary = {"opportunity": _opp(by_score_band=[_band("75+", 0.61), _band("60-74", 0.72)]),
                   "score_monotonicity": {"state": "NON_MONOTONIC"}}
        f = ip.derive_supported_findings(summary)
        self.assertTrue(any("not consistently ordered" in x["statement"] for x in f))


class StrengthTests(unittest.TestCase):
    def test_evidence_strength_tiers(self):
        self.assertEqual(ip.evidence_strength(60, 80), "STRONG")
        self.assertEqual(ip.evidence_strength(25), "MODERATE")
        self.assertEqual(ip.evidence_strength(10), "EARLY")


class NoiseTests(unittest.TestCase):
    def test_noise_maps_24a_assessment(self):
        aq = _aq(by_event_type=[
            {"event_type": "FADING", "matured": 30, "evaluable": 30, "confirmed": 24,
             "reversed": 3, "confirmation_rate": 0.8, "assessment": "Promising"},
            {"event_type": "RISING", "matured": 20, "evaluable": 20, "confirmed": 5,
             "reversed": 12, "confirmation_rate": 0.25, "assessment": "High reversal"}],
            frequency={"event_type_distribution": {"FADING": 31, "RISING": 400}})
        out = {n["event_type"]: n["assessment"] for n in ip.alert_noise_assessment(aq)}
        self.assertEqual(out["FADING"], "HIGH_VALUE")
        self.assertEqual(out["RISING"], "POTENTIALLY_NOISY")


class DefaultCohortTests(unittest.TestCase):
    def test_defaults_discovered_from_code(self):
        # enabled/disabled sets must come from the real DEFAULT_PREFERENCES
        from analytics.opportunity_events import DEFAULT_PREFERENCES
        aq = _aq(by_event_type=[
            {"event_type": "STATUS_UPGRADE", "evaluable": 50, "confirmed": 40, "reversed": 5},
            {"event_type": "RISING", "evaluable": 50, "confirmed": 20, "reversed": 25}])
        d = ip.default_preference_cohorts(aq)
        self.assertIn("STATUS_UPGRADE", d["enabled_events"])
        self.assertIn("RISING", d["disabled_events"])
        self.assertTrue(DEFAULT_PREFERENCES["upgrade"] and not DEFAULT_PREFERENCES["rising"])
        self.assertEqual(d["assessment"], "SUPPORTED")  # 0.8 vs 0.4

    def test_insufficient_sample(self):
        aq = _aq(by_event_type=[{"event_type": "STATUS_UPGRADE", "evaluable": 3, "confirmed": 2, "reversed": 0}])
        self.assertEqual(ip.default_preference_cohorts(aq)["assessment"], "INSUFFICIENT_SAMPLE")


class SummaryReadOnlyTests(unittest.TestCase):
    def test_summary_composes_and_is_read_only(self):
        opp = _opp(comparable=200, by_regime=[{"key": "UNKNOWN", "comparable": 200}],
                   by_horizon=[_horizon("NEXT", 0.78), _horizon("H24", 0.69), _horizon("H72", 0.60)],
                   by_status=[_status("STRONG", 0.75, 120), _status("WATCH", 0.60, 140)],
                   by_score_band=[_band("75+", 0.74, 142), _band("<50", 0.41, 201)])
        with (
            mock.patch("db.opportunity_outcomes.get_opportunity_outcome_summary", return_value=opp),
            mock.patch("db.intelligence_alerts.get_alert_quality_summary", return_value=_aq()),
            mock.patch("db.opportunity_outcomes.get_degraded_cohort_outcomes",
                       return_value={"available": False, "comparable": 0}),
            mock.patch("db.opportunity_outcomes.get_intelligence_evidence_freshness",
                       return_value={"latest_frozen_observation": None}),
        ):
            s = ip.get_intelligence_performance_summary()
        self.assertTrue(s["available"])
        self.assertEqual(s["regime"]["state"], "INSUFFICIENT_COVERAGE")  # all UNKNOWN
        self.assertIn(s["readiness"]["tier"], ("SUFFICIENT", "DEVELOPING"))
        self.assertTrue(any(f["dimension"] == "status" for f in s["findings"]))


if __name__ == "__main__":
    unittest.main()
