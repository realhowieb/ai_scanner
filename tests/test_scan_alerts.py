"""Scan alerts (SCAN_MATCH, Pro): saved Scanner filters on the whole market that fire
once per scan bringing new matching stocks. Pure checks plus the evaluator end to end
on SQLite, reusing the alert-rules engine test helpers."""
import datetime as dt
import importlib.util
import json
import unittest
from unittest import mock

from tests.test_alert_rules_engine import NOW, Store, observation, opp


class ScanFilterTests(unittest.TestCase):
    def setUp(self):
        from api import alert_rules

        self.ar = alert_rules

    def test_parse_and_normalize(self):
        f = self.ar.parse_scan_filters('{"name": " Big breakouts ", "setup": "Breakout", "min_price": "5", "min_rvol": 2}')
        self.assertEqual(f, {"name": "Big breakouts", "setup": "breakout", "min_price": 5.0, "min_rvol": 2.0})
        self.assertEqual(self.ar.parse_scan_filters(None), {})

    def test_bad_filters_are_rejected(self):
        for bad in ('[1]', 'nope', '{"sector": "tech"}', '{"setup": "moonshot"}', '{"min_price": -1}',
                    '{"min_price": 10, "max_price": 5}', '{"min_rvol": "x"}'):
            with self.subTest(bad=bad), self.assertRaises(self.ar.RuleError):
                self.ar.parse_scan_filters(bad)

    def test_validate_market_scope_and_plan(self):
        pro = {"can_scan_alerts": True, "can_email_alerts": True}
        out = self.ar.validate(pro, rule_type="SCAN_MATCH", threshold=80, value='{"setup":"gapper"}', ticker=None,
                               watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        self.assertEqual((out["ticker"], out["watchlist_id"], out["value"], out["cooldown_seconds"]),
                         (None, None, '{"setup":"gapper"}', 0))
        with self.assertRaises(self.ar.RuleError):
            self.ar.validate(pro, rule_type="SCAN_MATCH", threshold=80, value=None, ticker="AAPL",
                             watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        with self.assertRaises(self.ar.RuleForbidden):
            self.ar.validate({}, rule_type="SCAN_MATCH", threshold=80, value=None, ticker=None,
                             watchlist_id=None, delivery_channels=None, cooldown_seconds=None)
        with self.assertRaises(self.ar.RuleForbidden):  # PreBreakout filter stays Premium
            self.ar.validate(pro, rule_type="SCAN_MATCH", threshold=80, value='{"setup":"prebreakout"}',
                             ticker=None, watchlist_id=None, delivery_channels=None, cooldown_seconds=None)

    def test_scan_alerts_are_pro(self):
        from domain.plans import compute_entitlements

        self.assertFalse(compute_entitlements(tier_obj="basic", is_admin=False)["can_scan_alerts"])
        self.assertTrue(compute_entitlements(tier_obj="pro", is_admin=False)["can_scan_alerts"])
        types = {t["type"]: t for t in self.ar.rule_types({"can_scan_alerts": True})}
        self.assertEqual(types["SCAN_MATCH"]["scope"], "market")
        self.assertEqual(types["RVOL_ABOVE"]["scope"], "symbol")

    def test_check_scan_baseline_then_new_names_only(self):
        fires, fresh, st = self.ar.check_scan(["AAA", "BBB"], None)
        self.assertEqual((fires, fresh), (False, []))
        st = {**st, "observation_id": "run:1"}
        fires, fresh, st2 = self.ar.check_scan(["CCC", "AAA", "DDD"], st)
        self.assertEqual((fires, fresh), (True, ["CCC", "DDD"]))
        fires, fresh, _ = self.ar.check_scan(["AAA"], {**st2, "observation_id": "run:2"})
        self.assertEqual((fires, fresh), (False, []))

    def test_message_lists_names_and_caps(self):
        rule = {"value": '{"name":"Gappers"}'}
        self.assertEqual(self.ar.scan_message(rule, ["AAA"]), "1 new stock matches Gappers: AAA")
        many = [f"T{i}" for i in range(11)]
        self.assertTrue(self.ar.scan_message({"value": None}, many).endswith("and 3 more"))


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "needs fastapi (entitlements_for lives in api.main)")
class ScanAlertEvaluatorTests(unittest.TestCase):
    def setUp(self):
        from api import alert_rules
        from db import alert_rules as store

        self.ar, self.store = alert_rules, store
        self.db = Store()
        self.addCleanup(self.db.close)
        self.db.user("a@x.com", "pro")
        self.db.user("b@x.com", "basic")
        self.emails = []
        mock.patch("api.alert_rules._send_email",
                   side_effect=lambda u, m: (self.emails.append((u, m)) or (True, None))).start()
        mock.patch("api.watchlist_intel.observation_stale", return_value=False).start()
        self.addCleanup(mock.patch.stopall)

    def rule(self, user, threshold=70.0, value=None, channels=("in_app",)):
        return self.store.create_rule(user, rule_type="SCAN_MATCH", operator="new_matches", threshold=threshold,
                                      value=value, ticker=None, watchlist_id=None,
                                      delivery_channels=list(channels), cooldown_seconds=0)

    def test_fires_once_per_scan_with_new_matches(self):
        self.rule("a@x.com", 70, json.dumps({"name": "Strong breakouts", "setup": "breakout"}),
                  channels=("in_app", "email"))
        first = observation(1, {"AAA": opp(80, 1), "BBB": opp(60, 2)})
        self.assertEqual(self.ar.evaluate_once(NOW, observation=first)["triggered"], 0)  # baseline
        second = observation(2, {"AAA": opp(81, 2), "CCC": opp(90, 1), "DDD": opp(75, 3, signals=("gapper",), setup="Gapper"),
                                 "EEE": opp(72, 4)})
        r = self.ar.evaluate_once(NOW + dt.timedelta(minutes=60), observation=second)
        self.assertEqual(r["triggered"], 1)
        e = self.db.events()[-1]
        self.assertEqual((e["ticker"], e["trigger_value"]), ("CCC", 2.0))  # best rank first; DDD isn't a breakout
        self.assertEqual(e["message"], "2 new stocks match Strong breakouts: CCC, EEE")
        self.assertEqual(len(self.emails), 1)
        again = self.ar.evaluate_once(NOW + dt.timedelta(minutes=61), observation=second)
        self.assertEqual((again["triggered"], again["evaluated"]), (0, 0))
        third = observation(3, {"AAA": opp(81, 1), "CCC": opp(90, 2), "EEE": opp(72, 3)})
        self.assertEqual(self.ar.evaluate_once(NOW + dt.timedelta(hours=2), observation=third)["triggered"], 0)

    def test_price_and_volume_filters(self):
        self.rule("a@x.com", 0, json.dumps({"min_price": 20, "min_rvol": 2}))
        self.ar.evaluate_once(NOW, observation=observation(1, {}))
        cheap = opp(80, 1, rvol=3.0)  # last 10.0
        pricey = {**opp(80, 2, rvol=3.0), "last": 50.0}
        quiet = {**opp(80, 3, rvol=1.0), "last": 50.0}
        self.ar.evaluate_once(NOW + dt.timedelta(hours=1),
                              observation=observation(2, {"CHP": cheap, "BIG": pricey, "QUI": quiet}))
        self.assertEqual([e["ticker"] for e in self.db.events()], ["BIG"])

    def test_downgraded_owner_stops(self):
        self.rule("b@x.com", 0)
        r = self.ar.evaluate_once(NOW, observation=observation(1, {"AAA": opp(80, 1)}))
        self.assertEqual(r["rules"], 0)


if __name__ == "__main__":
    unittest.main()
