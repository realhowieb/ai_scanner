"""Run 24 — HSF intelligence alert QUALITY measurement (deterministic)."""
import datetime as dt
import unittest
from unittest import mock

from analytics import alert_quality as aq


def _alert(event_type, *, cur_status, cur_score, prev_status=None, version="1.0",
           ticker="NVDA", alert_time=None, fading=False):
    return {
        "id": 1, "ticker": ticker, "event_type": event_type,
        "alert_time": alert_time or dt.datetime(2026, 9, 1, 16, 0, tzinfo=dt.timezone.utc),
        "payload": {
            "ticker": ticker, "event_type": event_type,
            "current_status": cur_status, "current_score": cur_score,
            "previous_status": prev_status, "score_version": version, "fading": fading,
        },
    }


def _snap(when, opps):
    return {"snapshot_time": when, "opportunities": opps}


def _opp(ticker, status, score, version="1.0", fading=False):
    return {"ticker": ticker, "status": status, "score": score,
            "score_version": version, "fading": fading}


NOW = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)


class ClassifyTests(unittest.TestCase):
    def test_status_upgrade_confirmed(self):
        a = {"status": "STRONG", "score": 79, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": True, "status": "STRONG", "score": 80, "score_version": "1.0"}
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", a, s), aq.CONFIRMED)

    def test_status_upgrade_reversed(self):
        a = {"status": "STRONG", "score": 79, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": True, "status": "WATCH", "score": 60, "score_version": "1.0"}
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", a, s), aq.REVERSED)

    def test_fading_confirmed(self):
        a = {"status": "CAUTION", "score": 55, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": True, "status": "CAUTION", "score": 40, "score_version": "1.0", "fading": True}
        self.assertEqual(aq.classify_outcome("FADING", a, s), aq.CONFIRMED)

    def test_fading_recovered(self):
        a = {"status": "CAUTION", "score": 55, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": True, "status": "STRONG", "score": 78, "score_version": "1.0"}
        self.assertEqual(aq.classify_outcome("FADING", a, s), aq.RECOVERED)

    def test_dropped_remains_absent_confirmed(self):
        a = {"status": None, "score": None, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": False, "status": None, "score": None, "score_version": None}
        # DROPPED is a ranking change, not a price fall — absence confirms it.
        self.assertEqual(aq.classify_outcome("DROPPED", a, s), aq.CONFIRMED)

    def test_dropped_reentry_recovered(self):
        a = {"status": None, "score": None, "previous_status": "STRONG", "score_version": "1.0"}
        s = {"present": True, "status": "STRONG", "score": 80, "score_version": "1.0"}
        self.assertEqual(aq.classify_outcome("DROPPED", a, s), aq.RECOVERED)

    def test_new_opportunity_persisted(self):
        a = {"status": "WATCH", "score": 60, "previous_status": None, "score_version": "1.0"}
        s = {"present": True, "status": "WATCH", "score": 61, "score_version": "1.0"}
        self.assertEqual(aq.classify_outcome("NEW_OPPORTUNITY", a, s), aq.PERSISTED)

    def test_new_opportunity_not_persisted(self):
        a = {"status": "WATCH", "score": 60, "previous_status": None, "score_version": "1.0"}
        s = {"present": False, "status": None, "score": None, "score_version": None}
        self.assertEqual(aq.classify_outcome("NEW_OPPORTUNITY", a, s), aq.REVERSED)

    def test_version_change_blocks_numeric(self):
        a = {"status": "STRONG", "score": 79, "previous_status": "WATCH", "score_version": "1.0"}
        s = {"present": True, "status": "STRONG", "score": 80, "score_version": "2.0"}
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", a, s), aq.VERSION_CHANGED)


class HorizonTests(unittest.TestCase):
    def test_pending_before_horizon(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH",
                   alert_time=dt.datetime(2026, 9, 29, 16, 0, tzinfo=dt.timezone.utc))
        r = aq.evaluate_alert_at_horizon(a, "D5", [], now=NOW)  # 5d not elapsed
        self.assertEqual(r["data_status"], aq.PENDING)
        self.assertEqual(r["quality_classification"], aq.PENDING)

    def test_unavailable_when_no_later_snapshot(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        # horizon elapsed (NOW is weeks later) but no snapshot provided
        r = aq.evaluate_alert_at_horizon(a, "NEXT", [], now=NOW)
        self.assertEqual(r["data_status"], aq.UNAVAILABLE)

    def test_matured_picks_first_comparable(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        t0 = a["alert_time"]
        snaps = [
            _snap(t0 + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)]),   # earliest after
            _snap(t0 + dt.timedelta(hours=30), [_opp("NVDA", "WATCH", 60)]),
        ]
        r = aq.evaluate_alert_at_horizon(a, "NEXT", snaps, now=NOW)
        self.assertEqual(r["data_status"], aq.MATURED)
        self.assertEqual(r["evaluation_time"], t0 + dt.timedelta(hours=2))  # earliest wins
        self.assertEqual(r["quality_classification"], aq.CONFIRMED)
        self.assertTrue(r["still_present"])

    def test_no_leakage_ignores_pre_horizon_snapshots(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        t0 = a["alert_time"]
        snaps = [
            _snap(t0 - dt.timedelta(hours=5), [_opp("NVDA", "CAUTION", 30)]),  # before alert
            _snap(t0 + dt.timedelta(hours=80), [_opp("NVDA", "STRONG", 82)]),  # after D3 cutoff
        ]
        r = aq.evaluate_alert_at_horizon(a, "D3", snaps, now=NOW)
        self.assertEqual(r["data_status"], aq.MATURED)
        self.assertEqual(r["evaluation_time"], t0 + dt.timedelta(hours=80))

    def test_dropped_not_a_price_fall(self):
        a = _alert("DROPPED", cur_status=None, cur_score=None, prev_status="WATCH")
        t0 = a["alert_time"]
        snaps = [_snap(t0 + dt.timedelta(hours=2), [_opp("AMD", "STRONG", 80)])]  # NVDA absent
        r = aq.evaluate_alert_at_horizon(a, "NEXT", snaps, now=NOW)
        self.assertEqual(r["quality_classification"], aq.CONFIRMED)
        self.assertFalse(r["still_present"])


class MaturationTests(unittest.TestCase):
    def test_idempotent_persist_only_terminal(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        t0 = a["alert_time"]
        snaps = [_snap(t0 + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)])]
        persisted = {}

        def _persist(res):
            key = (res["alert_id"], res["evaluation_horizon"])
            if key in persisted:
                return False  # ON CONFLICT DO NOTHING
            persisted[key] = res
            return True

        with (
            mock.patch("db.intelligence_alerts.fetch_alerts_for_maturation", return_value=[a]),
            mock.patch("db.intelligence_alerts.persist_alert_outcome", side_effect=_persist),
            mock.patch("db.opportunity_snapshots.load_recent_snapshots", return_value=snaps),
            mock.patch("analytics.alert_quality._dt") as dtmock,
        ):
            dtmock.datetime.now.return_value = NOW
            dtmock.timedelta = dt.timedelta
            dtmock.timezone = dt.timezone
            n1 = aq.mature_alert_outcomes()
            n2 = aq.mature_alert_outcomes()  # second run: no new rows
        # NEXT matured -> 1 write; later horizons UNAVAILABLE (no snapshot) -> skipped.
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 0)

    def test_version_change_internal_alerts_skipped(self):
        a = _alert("VERSION_CHANGED", cur_status="STRONG", cur_score=79)
        with (
            mock.patch("db.intelligence_alerts.fetch_alerts_for_maturation", return_value=[a]),
            mock.patch("db.intelligence_alerts.persist_alert_outcome") as p,
            mock.patch("db.opportunity_snapshots.load_recent_snapshots", return_value=[]),
        ):
            n = aq.mature_alert_outcomes()
        self.assertEqual(n, 0)
        p.assert_not_called()


class QualityReadOnlyTests(unittest.TestCase):
    def test_summary_is_read_only(self):
        cur = mock.MagicMock()
        cur.fetchone.side_effect = [
            (40, 30, 20),          # total/elapsed pairs + alerts_total
            (5, 0),                # distinct tickers + delivered
            (100, 40, 10, 8),      # matched/deduped/filtered/runs
        ]
        # grouped queries
        cur.fetchall.side_effect = [
            [("CONFIRMED", 12), ("REVERSED", 3)],                 # overall
            [("STATUS_UPGRADE", "CONFIRMED", 12), ("STATUS_UPGRADE", "REVERSED", 3)],  # by event
            [("NEXT", "CONFIRMED", 12), ("NEXT", "REVERSED", 3)], # by horizon
            [("STATUS_UPGRADE", 15), ("RISING", 5)],              # distribution
        ]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        from db import intelligence_alerts as ia
        with mock.patch.object(ia, "get_neon_conn", return_value=conn):
            q = ia.get_alert_quality_summary(min_sample=10)
        self.assertTrue(q["available"])
        self.assertEqual(q["matured"], 15)
        self.assertEqual(q["confirmed"], 12)
        self.assertEqual(q["reversed"], 3)
        # pending = total(40) - elapsed(30); unavailable = elapsed(30) - matured(15)
        self.assertEqual(q["pending"], 10)
        self.assertEqual(q["unavailable"], 15)
        for c in cur.execute.call_args_list:
            self.assertNotRegex(c[0][0].upper(), r"\b(INSERT|UPDATE|DELETE)\b")

    def test_small_sample_is_insufficient(self):
        cur = mock.MagicMock()
        cur.fetchone.side_effect = [(8, 8, 2), (2, 0), (8, 0, 0, 2)]
        cur.fetchall.side_effect = [
            [("CONFIRMED", 6)],
            [("STATUS_UPGRADE", "CONFIRMED", 6)],
            [("NEXT", "CONFIRMED", 6)],
            [("STATUS_UPGRADE", 6)],
        ]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        from db import intelligence_alerts as ia
        with mock.patch.object(ia, "get_neon_conn", return_value=conn):
            q = ia.get_alert_quality_summary(min_sample=10)
        self.assertIsNone(q["confirmation_rate"])  # below threshold
        self.assertEqual(q["by_event_type"][0]["assessment"], "INSUFFICIENT_SAMPLE")


if __name__ == "__main__":
    unittest.main()
