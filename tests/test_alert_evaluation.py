"""Run 22 — background intelligence-alert evaluation (Stage A + B)."""
import unittest
from unittest import mock

from analytics import alert_evaluation as ae


def opp(t, s, st_, ver="1.0", sig=None, fading=False):
    return {"ticker": t, "score": s, "status": st_, "score_version": ver,
            "signals": sig or [], "primary_setup": "Breakout", "fading": fading}


def _snaps():
    # newest first: current (NVDA upgraded), previous
    return [
        {"snapshot_time": "t2", "opportunities": [opp("NVDA", 79, "STRONG", sig=["breakout"]),
                                                  opp("AMD", 60, "WATCH")]},
        {"snapshot_time": "t1", "opportunities": [opp("NVDA", 71, "WATCH"),
                                                  opp("AMD", 60, "WATCH")]},
    ]


class BackgroundEvaluationTests(unittest.TestCase):
    def _run(self, *, watchers, prefs_by_user, recent=None, deliver=None, snaps=None):
        recent = recent or {}
        record = mock.MagicMock(return_value=1)
        update = mock.MagicMock(return_value=True)
        with (
            mock.patch("db.opportunity_snapshots.load_recent_snapshots",
                       return_value=snaps if snaps is not None else _snaps()) as loadsnap,
            mock.patch.object(ae, "_watchers", return_value=watchers),
            mock.patch("db.intelligence_alerts.get_hsf_alert_prefs",
                       side_effect=lambda u: prefs_by_user.get(u)),
            mock.patch("db.intelligence_alerts.recent_fingerprints",
                       side_effect=lambda u, **k: set(recent.get(u, set()))),
            mock.patch("db.intelligence_alerts.record_intelligence_alert", record),
            mock.patch("db.intelligence_alerts.update_delivery_status", update),
            mock.patch("ui.ai.ask_claude") as claude,
        ):
            m = ae.run_intelligence_alert_evaluation(deliver=deliver)
        # Stage A detects once (one snapshot load), no Claude ever.
        self.assertEqual(loadsnap.call_count, 1)
        claude.assert_not_called()
        return m, record

    def test_batch_detect_once_and_user_isolation(self):
        delivered = []
        m, _ = self._run(
            watchers={"A": {"NVDA"}, "B": {"AMD"}},
            prefs_by_user={"A": {"upgrade": True}, "B": {"upgrade": True}},
            deliver=lambda u, c, n: delivered.append((u, n["ticker"])) or True)
        # A watches NVDA (upgraded) -> notified; B watches AMD (unchanged) -> not.
        self.assertIn(("A", "NVDA"), delivered)
        self.assertFalse(any(u == "B" for u, _ in delivered))
        self.assertEqual(m["delivered"], 1)

    def test_preferences_suppress(self):
        delivered = []
        m, _ = self._run(
            watchers={"A": {"NVDA"}},
            prefs_by_user={"A": {"upgrade": False}},   # upgrades off
            deliver=lambda u, c, n: delivered.append(u) or True)
        self.assertEqual(delivered, [])
        self.assertEqual(m["delivered"], 0)

    def test_dedupe_within_cooldown(self):
        from analytics.opportunity_events import (
            collapse_events,
            derive_opportunity_events,
            event_fingerprint,
        )
        notes = collapse_events(derive_opportunity_events(
            _snaps()[1]["opportunities"], _snaps()[0]["opportunities"]))
        fp = event_fingerprint("A", next(n for n in notes if n["ticker"] == "NVDA"))
        delivered = []
        m, _ = self._run(
            watchers={"A": {"NVDA"}}, prefs_by_user={"A": {"upgrade": True}},
            recent={"A": {fp}},  # already delivered within cooldown
            deliver=lambda u, c, n: delivered.append(u) or True)
        self.assertEqual(delivered, [])
        self.assertEqual(m["deduped"], 1)

    def test_delivery_failure_recorded_not_faked(self):
        def boom(u, c, n):
            raise RuntimeError("smtp down")
        m, record = self._run(
            watchers={"A": {"NVDA"}}, prefs_by_user={"A": {"upgrade": True}}, deliver=boom)
        self.assertEqual(m["failed"], 1)
        self.assertEqual(m["delivered"], 0)
        self.assertTrue(record.called)  # event retained despite delivery failure

    def test_needs_two_snapshots(self):
        m, _ = self._run(watchers={"A": {"NVDA"}}, prefs_by_user={"A": {}},
                         snaps=[_snaps()[0]])  # only one
        self.assertEqual(m["events_detected"], 0)
        self.assertIn("two", m["reason"])

    def test_malformed_snapshot_does_not_abort(self):
        bad = [{"snapshot_time": "t2", "opportunities": [{"ticker": None}, opp("NVDA", 79, "STRONG")]},
               {"snapshot_time": "t1", "opportunities": [opp("NVDA", 71, "WATCH")]}]
        m, _ = self._run(watchers={"A": {"NVDA"}}, prefs_by_user={"A": {"upgrade": True}},
                         deliver=lambda u, c, n: True, snaps=bad)
        self.assertGreaterEqual(m["delivered"], 1)  # valid NVDA still processed


if __name__ == "__main__":
    unittest.main()
