"""Run 40 — Intelligent Alerts / Opportunity View tests (deterministic)."""
import unittest

from analytics import opportunity_view as ov


def _obs(symbol="NVDA", scanners=None, indicators=None, models=None, dq=None,
         session="morning", price=131.0):
    return {
        "observation_id": "obs1", "symbol": symbol, "timestamp": "2026-08-25T14:00:00Z",
        "session": session, "market": {"price": price},
        "indicators": indicators if indicators is not None else
            {"rvol": 3.1, "vs_vwap_pct": 0.8, "adx": 28, "chg_pct": 1.5, "gap_pct": 0.4},
        "models": models if models is not None else
            {"prebreakout": {"probability": 0.78}},
        "scanners": scanners if scanners is not None else
            [{"name": "prebreakout", "triggered": True, "direction": "long"},
             {"name": "momentum", "triggered": True, "direction": "long"},
             {"name": "unusual_vol", "triggered": True, "direction": "long"}],
        "data_quality": dq if dq is not None else
            {"feature_completeness": 1.0, "fallback_used": False, "stale": False},
    }


class PrimarySetupTests(unittest.TestCase):
    def test_prebreakout_wins(self):
        n, lbl = ov.select_primary_setup(_obs()["scanners"])
        self.assertEqual((n, lbl), ("prebreakout", "PreBreakout"))

    def test_order_without_prebreakout(self):
        s = [{"name": "unusual_vol"}, {"name": "breakout"}, {"name": "momentum"}]
        _, lbl = ov.select_primary_setup(s)
        self.assertEqual(lbl, "Breakout")  # breakout ahead of momentum/unusual_vol

    def test_empty(self):
        self.assertEqual(ov.select_primary_setup([]), (None, None))


class AgreementTests(unittest.TestCase):
    def test_count_and_names(self):
        a = ov.scanner_agreement(_obs()["scanners"])
        self.assertEqual(a["count"], 3)
        self.assertIn("Unusual Volume", a["labels"])

    def test_direction(self):
        self.assertEqual(ov.overall_direction(_obs()["scanners"]), "bullish")
        mixed = [{"name": "gap_up", "direction": "long"},
                 {"name": "gap_down", "direction": "short"}]
        self.assertEqual(ov.overall_direction(mixed), "mixed")


class ExplanationTests(unittest.TestCase):
    def test_positive_reasons_evidence_based(self):
        r = ov.positive_reasons(_obs())
        self.assertTrue(any("PreBreakout probability 78%" in x for x in r))
        self.assertTrue(any("RVOL 3.1x" in x for x in r))
        self.assertTrue(any("Above VWAP" in x for x in r))
        self.assertTrue(any("3 scanners agree" in x for x in r))

    def test_no_invented_reasons_when_absent(self):
        r = ov.positive_reasons(_obs(indicators={}, models={}))
        self.assertFalse(any("PreBreakout" in x for x in r))
        self.assertFalse(any("RVOL" in x for x in r))

    def test_risks_from_data(self):
        risks = ov.risk_reasons(_obs(indicators={"vs_vwap_pct": -0.5, "rvol": 0.6,
                                                  "adx": 12, "gap_pct": 6.0}))
        self.assertIn("Below VWAP", risks)
        self.assertTrue(any("Low participation" in x for x in risks))
        self.assertTrue(any("Weak trend" in x for x in risks))
        self.assertTrue(any("Large opening gap" in x for x in risks))

    def test_data_quality_risks(self):
        self.assertIn("Stale data", ov.risk_reasons(_obs(dq={"stale": True})))
        self.assertIn("Incomplete data (fallback)",
                      ov.risk_reasons(_obs(dq={"fallback_used": True})))

    def test_conflicting_directions_risk(self):
        r = ov.risk_reasons(_obs(scanners=[{"name": "gap_up", "direction": "long"},
                                           {"name": "gap_down", "direction": "short"}]))
        self.assertIn("Conflicting scanner directions", r)


class ChangeDetectionTests(unittest.TestCase):
    def test_no_prior_no_changes(self):
        self.assertEqual(ov.detect_changes(_obs(), None), [])

    def test_prob_rvol_vwap_scanner_direction(self):
        prior = _obs(indicators={"rvol": 1.8, "vs_vwap_pct": -0.2, "chg_pct": 0.1},
                     models={"prebreakout": {"probability": 0.61}},
                     scanners=[{"name": "prebreakout", "direction": "long"}])
        cur = _obs(indicators={"rvol": 3.2, "vs_vwap_pct": 0.5, "chg_pct": 1.0},
                   models={"prebreakout": {"probability": 0.76}},
                   scanners=[{"name": "prebreakout", "direction": "long"},
                             {"name": "unusual_vol", "direction": "long"}])
        changes = ov.detect_changes(cur, prior)
        self.assertTrue(any("61% → 76%" in c for c in changes))
        self.assertTrue(any("1.8x → 3.2x" in c for c in changes))
        self.assertTrue(any("Moved above VWAP" in c for c in changes))
        self.assertTrue(any("New scanner: Unusual Volume" in c for c in changes))
        self.assertTrue(any("agreement: 1 → 2" in c for c in changes))

    def test_small_changes_ignored(self):
        prior = _obs(indicators={"rvol": 3.0}, models={"prebreakout": {"probability": 0.75}})
        cur = _obs(indicators={"rvol": 3.1}, models={"prebreakout": {"probability": 0.76}})
        self.assertEqual(ov.detect_changes(cur, prior), [])  # below thresholds


class PriorityTests(unittest.TestCase):
    def test_high_on_three_scanners(self):
        p, reason = ov.alert_priority(_obs())
        self.assertEqual(p, "HIGH")
        self.assertIn("3 scanners agree", reason)

    def test_medium_two_scanners(self):
        obs = _obs(scanners=[{"name": "prebreakout", "direction": "long"},
                             {"name": "momentum", "direction": "long"}],
                   models={"prebreakout": {"probability": 0.3}})
        self.assertEqual(ov.alert_priority(obs)[0], "MEDIUM")

    def test_low_single_setup(self):
        obs = _obs(scanners=[{"name": "most_active", "direction": "long"}],
                   indicators={"rvol": 1.1}, models={})
        self.assertEqual(ov.alert_priority(obs)[0], "LOW")

    def test_stale_caps_high_to_medium(self):
        obs = _obs(dq={"stale": True})  # would be HIGH on 3 scanners
        p, reason = ov.alert_priority(obs)
        self.assertEqual(p, "MEDIUM")
        self.assertIn("data quality", reason)

    def test_priority_not_predictive_language(self):
        _, reason = ov.alert_priority(_obs())
        for banned in ("return", "win", "probability of profit", "edge"):
            self.assertNotIn(banned, reason.lower())


class LifecycleTests(unittest.TestCase):
    def test_new(self):
        self.assertEqual(ov.lifecycle_state(prior_state=None, changes=[]), "NEW")

    def test_strengthening(self):
        self.assertEqual(
            ov.lifecycle_state(prior_state="ACTIVE",
                               changes=["New scanner: Unusual Volume"]), "STRENGTHENING")

    def test_weakening(self):
        self.assertEqual(
            ov.lifecycle_state(prior_state="ACTIVE", changes=["Lost VWAP"]), "WEAKENING")

    def test_active(self):
        self.assertEqual(ov.lifecycle_state(prior_state="ACTIVE", changes=[]), "ACTIVE")

    def test_resolved(self):
        self.assertEqual(
            ov.lifecycle_state(prior_state="ACTIVE", changes=[], present=False), "RESOLVED")


class DedupTests(unittest.TestCase):
    def test_first_trigger_alerts(self):
        v = {"alert_priority": "HIGH", "changes_since_prior": []}
        self.assertEqual(ov.should_alert(v, None), (True, "first trigger"))

    def test_no_change_suppressed(self):
        v = {"alert_priority": "HIGH", "changes_since_prior": [], "lifecycle_state": "ACTIVE"}
        prior = {"alert_priority": "HIGH", "lifecycle_state": "ACTIVE"}
        fired, _ = ov.should_alert(v, prior)
        self.assertFalse(fired)

    def test_priority_increase_alerts(self):
        v = {"alert_priority": "HIGH", "changes_since_prior": []}
        prior = {"alert_priority": "MEDIUM", "lifecycle_state": "ACTIVE"}
        self.assertTrue(ov.should_alert(v, prior)[0])

    def test_material_change_alerts(self):
        v = {"alert_priority": "MEDIUM",
             "changes_since_prior": ["New scanner: Unusual Volume"]}
        prior = {"alert_priority": "MEDIUM", "lifecycle_state": "ACTIVE"}
        self.assertTrue(ov.should_alert(v, prior)[0])

    def test_reappearance_alerts(self):
        v = {"alert_priority": "MEDIUM", "changes_since_prior": []}
        prior = {"alert_priority": "MEDIUM", "lifecycle_state": "RESOLVED"}
        self.assertTrue(ov.should_alert(v, prior)[0])


class ViewTests(unittest.TestCase):
    def test_build_view_and_watchlist(self):
        v = ov.build_opportunity_view(_obs(), watchlist=["NVDA", "AMD"])
        self.assertEqual(v["symbol"], "NVDA")
        self.assertEqual(v["primary_setup"], "PreBreakout")
        self.assertTrue(v["is_watchlist"])
        self.assertEqual(v["alert_priority"], "HIGH")
        self.assertEqual(v["freshness"], "Fresh")
        self.assertEqual(v["lifecycle_state"], "NEW")
        self.assertNotIn("PreBreakout", v["secondary_setups"])  # primary excluded

    def test_deterministic(self):
        self.assertEqual(ov.build_opportunity_view(_obs()),
                         ov.build_opportunity_view(_obs()))

    def test_no_pii_in_view(self):
        v = ov.build_opportunity_view(_obs())
        blob = str(v).lower()
        for pii in ("email", "user_id", "@", "password", "billing"):
            self.assertNotIn(pii, blob)

    def test_rank_feed_priority_order(self):
        hi = ov.build_opportunity_view(_obs("AAA"))
        lo = ov.build_opportunity_view(_obs("BBB",
                scanners=[{"name": "most_active", "direction": "long"}],
                indicators={"rvol": 1.0}, models={}))
        ranked = ov.rank_feed([lo, hi])
        self.assertEqual(ranked[0]["symbol"], "AAA")  # HIGH before LOW


class FilterTests(unittest.TestCase):
    def _views(self):
        hi = ov.build_opportunity_view(_obs("AAA"), watchlist=["AAA"])
        lo = ov.build_opportunity_view(_obs("BBB",
                scanners=[{"name": "gap_down", "direction": "short"}],
                indicators={"gap_pct": -3.0}, models={}))
        return [hi, lo]

    def test_high_priority_filter(self):
        views = self._views()
        self.assertEqual([v["symbol"] for v in ov.filter_feed(views, "High Priority")],
                         ["AAA"])

    def test_bearish_and_watchlist_filters(self):
        views = self._views()
        self.assertEqual([v["symbol"] for v in ov.filter_feed(views, "Bearish")], ["BBB"])
        self.assertEqual([v["symbol"] for v in ov.filter_feed(views, "Watchlist")], ["AAA"])

    def test_all_returns_everything(self):
        views = self._views()
        self.assertEqual(len(ov.filter_feed(views, "All")), 2)


class EmptyStateTests(unittest.TestCase):
    def test_real_counts(self):
        msg = ov.empty_state_message(scanned=7842, detected=23, high_priority=0)
        self.assertIn("7,842", msg)
        self.assertIn("23", msg)

    def test_suppressed_when_high_priority_exists(self):
        self.assertEqual(ov.empty_state_message(scanned=100, detected=5, high_priority=2), "")

    def test_omits_unknown_counts(self):
        msg = ov.empty_state_message(scanned=None, detected=None)
        self.assertIn("No high-priority", msg)
        self.assertNotIn("None", msg)


if __name__ == "__main__":
    unittest.main()
