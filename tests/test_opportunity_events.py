"""Run 22 — canonical opportunity event engine, severity, collapsing, policy."""
import unittest

import analytics.opportunity_events as oe


def opp(t, s, st_, ver="1.0", sig=None, fading=False):
    return {"ticker": t, "score": s, "status": st_, "score_version": ver,
            "signals": sig or [], "primary_setup": "Breakout", "fading": fading}


class EventDetectionTests(unittest.TestCase):
    def test_new_only_when_absent_from_baseline(self):
        ev = oe.derive_opportunity_events(
            [opp("AMD", 60, "WATCH"), opp("NVDA", 70, "STRONG")],
            [opp("AMD", 60, "WATCH"), opp("NVDA", 70, "STRONG"), opp("TSLA", 64, "WATCH")])
        self.assertTrue(any(e["event_type"] == "NEW_OPPORTUNITY" and e["ticker"] == "TSLA" for e in ev))
        self.assertFalse(any(e["ticker"] in ("AMD", "NVDA") and e["event_type"] == "NEW_OPPORTUNITY" for e in ev))

    def test_no_baseline_yields_no_events(self):
        self.assertEqual(oe.derive_opportunity_events(None, [opp("AMD", 60, "WATCH")]), [])
        self.assertEqual(oe.derive_opportunity_events([], [opp("AMD", 60, "WATCH")]), [])

    def test_status_upgrade_and_downgrade(self):
        up = oe.derive_opportunity_events([opp("NVDA", 71, "WATCH")], [opp("NVDA", 79, "STRONG")])
        self.assertTrue(any(e["event_type"] == "STATUS_UPGRADE" for e in up))
        dn = oe.derive_opportunity_events([opp("NVDA", 81, "STRONG")], [opp("NVDA", 72, "WATCH")])
        self.assertTrue(any(e["event_type"] == "STATUS_DOWNGRADE" for e in dn))

    def test_version_change_blocks_score_move(self):
        vc = oe.derive_opportunity_events([opp("NVDA", 70, "WATCH", "1.0")], [opp("NVDA", 82, "STRONG", "2.0")])
        self.assertEqual([e["event_type"] for e in vc], ["VERSION_CHANGED"])

    def test_signal_add_remove_order_insensitive(self):
        add = oe.derive_opportunity_events([opp("N", 62, "WATCH", sig=["prebreakout"])],
                                           [opp("N", 63, "WATCH", sig=["golden_cross", "prebreakout"])])
        self.assertTrue(any(e["event_type"] == "SIGNAL_ADDED" and e.get("signal") == "golden_cross" for e in add))
        same = oe.derive_opportunity_events([opp("N", 62, "WATCH", sig=["breakout", "golden_cross"])],
                                            [opp("N", 62, "WATCH", sig=["golden_cross", "breakout"])])
        self.assertFalse(any(e["event_type"] in ("SIGNAL_ADDED", "SIGNAL_REMOVED") for e in same))
        rem = oe.derive_opportunity_events([opp("N", 68, "WATCH", sig=["golden_cross", "prebreakout"])],
                                           [opp("N", 68, "WATCH", sig=["golden_cross"])])
        self.assertTrue(any(e["event_type"] == "SIGNAL_REMOVED" and e.get("signal") == "prebreakout" for e in rem))

    def test_dropped(self):
        dr = oe.derive_opportunity_events(
            [opp("AMD", 60, "WATCH"), opp("NVDA", 80, "STRONG"), opp("TSLA", 64, "WATCH")],
            [opp("AMD", 60, "WATCH"), opp("TSLA", 64, "WATCH")])
        d = [e for e in dr if e["event_type"] == "DROPPED"]
        self.assertEqual([e["ticker"] for e in d], ["NVDA"])

    def test_fading_needs_flag_and_weakening(self):
        fade = oe.derive_opportunity_events([opp("T", 76, "STRONG")],
                                            [opp("T", 68, "WATCH", fading=True)])
        self.assertTrue(any(e["event_type"] == "FADING" for e in fade))
        # fading flag but strengthening -> no FADING event
        nofade = oe.derive_opportunity_events([opp("T", 60, "WATCH")],
                                              [opp("T", 70, "WATCH", fading=True)])
        self.assertFalse(any(e["event_type"] == "FADING" for e in nofade))

    def test_malformed_current_rows_skipped(self):
        ev = oe.derive_opportunity_events([opp("A", 60, "WATCH")],
                                          [{"ticker": None}, {"score": 5}, opp("A", 66, "WATCH")])
        self.assertTrue(any(e["ticker"] == "A" for e in ev))  # valid row survives, no crash


class SeverityAndCollapseTests(unittest.TestCase):
    def test_severity_mapping(self):
        self.assertEqual(oe.event_severity({"event_type": "STATUS_UPGRADE", "current_status": "STRONG"}), "HIGH")
        self.assertEqual(oe.event_severity({"event_type": "STATUS_UPGRADE", "current_status": "WATCH"}), "MEDIUM")
        self.assertEqual(oe.event_severity({"event_type": "STATUS_DOWNGRADE", "previous_status": "STRONG"}), "HIGH")
        self.assertEqual(oe.event_severity({"event_type": "RISING"}), "LOW")
        self.assertEqual(oe.event_severity({"event_type": "SIGNAL_ADDED", "signal": "breakout"}), "MEDIUM")

    def test_collapse_one_notification_with_facts(self):
        up = oe.derive_opportunity_events([opp("NVDA", 71, "WATCH", sig=["golden_cross"])],
                                          [opp("NVDA", 79, "STRONG", sig=["golden_cross", "breakout"])])
        notes = oe.collapse_events(up)
        self.assertEqual(len(notes), 1)
        n = notes[0]
        self.assertEqual(n["event_type"], "STATUS_UPGRADE")   # strongest wins
        self.assertIn("breakout", n["signals_added"])
        self.assertIn("RISING", n["contributing"])
        copy = oe.notification_copy(n)
        self.assertIn("WATCH → STRONG", copy)
        self.assertIn("HSF 71 → 79", copy)
        for banned in ("BUY", "SELL", "$"):
            self.assertNotIn(banned, copy)

    def test_dropped_copy_is_ranking_not_price(self):
        n = oe.collapse_events(oe.derive_opportunity_events(
            [opp("A", 60, "WATCH"), opp("B", 80, "STRONG")], [opp("A", 60, "WATCH")]))
        b = next(x for x in n if x["ticker"] == "B")
        self.assertIn("dropped from HSF opportunities", oe.notification_copy(b))
        self.assertNotIn("price", oe.notification_copy(b).lower())


class PolicyTests(unittest.TestCase):
    def _note(self, et, **kw):
        base = {"ticker": "NVDA", "event_type": et, "previous_status": "WATCH",
                "current_status": "STRONG", "previous_score": 71, "current_score": 79,
                "signals_added": [], "signals_removed": []}
        base.update(kw)
        return base

    def test_preferences_gate(self):
        prefs = {"upgrade": True, "new": False, "fading": True}
        self.assertTrue(oe.should_notify(self._note("STATUS_UPGRADE"), prefs)["notify"])
        self.assertFalse(oe.should_notify(self._note("NEW_OPPORTUNITY"), prefs)["notify"])
        self.assertTrue(oe.should_notify(self._note("FADING", previous_status="STRONG", current_status="WATCH"), prefs)["notify"])

    def test_version_changed_never_notifies(self):
        self.assertFalse(oe.should_notify(self._note("VERSION_CHANGED"), None)["notify"])

    def test_dedupe_by_fingerprint(self):
        n = self._note("STATUS_UPGRADE")
        fp = oe.event_fingerprint("u1", n)
        # first time eligible; once fingerprint is 'recent', suppressed.
        self.assertTrue(oe.should_notify(n, {"upgrade": True}, set(), user_id="u1")["notify"])
        self.assertFalse(oe.should_notify(n, {"upgrade": True}, {fp}, user_id="u1")["notify"])

    def test_fingerprint_is_state_aware_not_timestamp(self):
        a = oe.event_fingerprint("u1", self._note("STATUS_UPGRADE"))
        b = oe.event_fingerprint("u1", self._note("STATUS_UPGRADE", current_score=99))  # different bucket
        self.assertNotEqual(a, b)
        # per-user isolation baked into the key
        self.assertNotEqual(oe.event_fingerprint("u1", self._note("STATUS_UPGRADE")),
                            oe.event_fingerprint("u2", self._note("STATUS_UPGRADE")))


if __name__ == "__main__":
    unittest.main()
