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


class StatusSemanticsTests(unittest.TestCase):
    def _run(self, *, watchers, prefs, snaps=None, deliver=None, detect_raises=False, recent=None):
        from contextlib import ExitStack
        recent = recent or {}
        recmock = mock.MagicMock(return_value=1)
        with ExitStack() as es:
            es.enter_context(mock.patch("db.opportunity_snapshots.load_recent_snapshots",
                                        return_value=snaps if snaps is not None else _snaps()))
            es.enter_context(mock.patch.object(ae, "_watchers", return_value=watchers))
            es.enter_context(mock.patch("db.intelligence_alerts.get_hsf_alert_prefs",
                                        side_effect=lambda u: prefs.get(u)))
            es.enter_context(mock.patch("db.intelligence_alerts.recent_fingerprints",
                                        side_effect=lambda u, **k: set(recent.get(u, set()))))
            es.enter_context(mock.patch("db.intelligence_alerts.record_intelligence_alert",
                                        mock.MagicMock(return_value=1)))
            es.enter_context(mock.patch("db.intelligence_alerts.update_delivery_status",
                                        mock.MagicMock(return_value=True)))
            es.enter_context(mock.patch("db.intelligence_alerts.record_evaluation_run", recmock))
            if detect_raises:
                es.enter_context(mock.patch("analytics.opportunity_events.derive_opportunity_events",
                                            side_effect=RuntimeError("boom")))
            r = ae.run_intelligence_alert_evaluation(deliver=deliver)
        return r, recmock

    def test_healthy_success_and_run_persisted(self):
        r, rec = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {"upgrade": True}},
                           deliver=lambda u, c, n: True)
        self.assertEqual(r["status"], "SUCCESS")
        self.assertEqual(r["delivered"], 1)
        self.assertTrue(rec.called)  # evaluation-run record persisted

    def test_zero_events_is_success_not_failed(self):
        same = [{"snapshot_time": "t2", "opportunities": [opp("NVDA", 71, "WATCH")]},
                {"snapshot_time": "t1", "opportunities": [opp("NVDA", 71, "WATCH")]}]
        r, _ = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {}}, snaps=same)
        self.assertEqual(r["status"], "SUCCESS")
        self.assertEqual(r["events_detected"], 0)

    def test_missing_baseline_is_skipped(self):
        r, _ = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {}}, snaps=[_snaps()[0]])
        self.assertEqual(r["status"], "SKIPPED")
        self.assertIn("two", r["reason"])

    def test_detection_failure_is_failed_with_stage(self):
        r, _ = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {}}, detect_raises=True)
        self.assertEqual(r["status"], "FAILED")
        self.assertEqual(r["error_stage"], "DETECT_EVENTS")

    def test_delivery_failure_is_partial(self):
        r, _ = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {"upgrade": True}},
                         deliver=lambda u, c, n: False)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["failed"], 1)
        self.assertEqual(r["delivered"], 0)

    def test_filtered_by_preferences_distinct_from_deduped(self):
        r, _ = self._run(watchers={"A": {"NVDA"}}, prefs={"A": {"upgrade": False}},
                         deliver=lambda u, c, n: True)
        self.assertEqual(r["filtered_by_preferences"], 1)
        self.assertEqual(r["deduped"], 0)


class HealthReaderTests(unittest.TestCase):
    def _health(self, latest, last_success, recent_fail, since, since_success):
        cur = mock.MagicMock()
        cur.fetchone.side_effect = [latest, last_success, recent_fail, since, since_success]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        from db import intelligence_alerts as ia
        with mock.patch.object(ia, "get_neon_conn", return_value=conn):
            return ia.get_intelligence_health(), cur

    # latest row layout: (started, status, events, matched, delivered, failed, deduped, filtered, users, error_stage, reason, dur)
    def test_health_read_only_no_writes(self):
        latest = ("t", "SUCCESS", 6, 3, 2, 0, 1, 0, 4, None, None, 12)
        h, cur = self._health(latest, ("t",), (0,), (5.0,), (5.0,))
        self.assertEqual(h["status"], "HEALTHY")
        # only SELECTs — never an INSERT/UPDATE/DELETE from health.
        for c in cur.execute.call_args_list:
            self.assertNotRegex(c[0][0].upper(), r"\b(INSERT|UPDATE|DELETE)\b")

    def test_unknown_when_no_runs(self):
        h, _ = self._health(None, (None,), (0,), (None,), (None,))
        self.assertEqual(h["status"], "UNKNOWN")

    def test_stale_when_no_recent_success(self):
        latest = ("t", "SUCCESS", 0, 0, 0, 0, 0, 0, 0, None, None, 5)
        h, _ = self._health(latest, ("old",), (0,), (10.0,), (2000.0,))  # 2000 min > 24h
        self.assertEqual(h["status"], "STALE")

    def test_degraded_when_latest_failed(self):
        latest = ("t", "FAILED", 0, 0, 0, 0, 0, 0, 0, "DETECT_EVENTS", "boom", 3)
        h, _ = self._health(latest, ("t2",), (1,), (2.0,), (30.0,))
        self.assertEqual(h["status"], "DEGRADED")
        self.assertEqual(h["error_stage"], "DETECT_EVENTS")

    def test_degraded_when_partial(self):
        latest = ("t", "PARTIAL", 5, 3, 1, 2, 0, 0, 2, None, "delivery failures", 8)
        h, _ = self._health(latest, ("t",), (0,), (2.0,), (2.0,))
        self.assertEqual(h["status"], "DEGRADED")
        self.assertEqual(h["delivery_failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
