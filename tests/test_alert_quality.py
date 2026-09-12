"""Run 24 + 24A — HSF intelligence alert QUALITY measurement (deterministic).

24A adds the hardening matrix: NEXT same-snapshot protection, leakage, snapshot
validity, out-of-order/duplicate robustness, version contamination, first-
observation immutability, and denominator correctness.
"""
import datetime as dt
import unittest
from unittest import mock

from analytics import alert_quality as aq

SRC = dt.datetime(2026, 9, 1, 16, 0, tzinfo=dt.timezone.utc)  # source snapshot time
NOW = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)


def _alert(event_type, *, cur_status, cur_score, prev_status=None, version="1.0",
           ticker="NVDA", source_time=SRC, fading=False):
    return {
        "id": 1, "ticker": ticker, "event_type": event_type,
        "alert_time": source_time + dt.timedelta(minutes=5),  # recorded after snapshot
        "payload": {
            "ticker": ticker, "event_type": event_type,
            "current_status": cur_status, "current_score": cur_score,
            "previous_status": prev_status, "score_version": version, "fading": fading,
            "current_snapshot_time": source_time,
        },
    }


def _snap(when, opps):
    return {"snapshot_time": when, "opportunities": opps}


def _opp(ticker, status, score, version="1.0", fading=False):
    return {"ticker": ticker, "status": status, "score": score,
            "score_version": version, "fading": fading}


def _st(status, score, *, present=True, version="1.0", fading=False):
    return {"present": present, "status": status, "score": score,
            "score_version": version, "fading": fading}


def _al(status, score, prev=None, version="1.0"):
    return {"status": status, "score": score, "previous_status": prev, "score_version": version}


class ClassifyTests(unittest.TestCase):
    def test_status_upgrade_confirmed(self):
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", _al("STRONG", 79, "WATCH"),
                                             _st("STRONG", 80)), aq.CONFIRMED)

    def test_status_upgrade_reversed(self):
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", _al("STRONG", 79, "WATCH"),
                                             _st("WATCH", 60)), aq.REVERSED)

    def test_status_upgrade_absent_reversed(self):
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", _al("STRONG", 79, "WATCH"),
                                             _st(None, None, present=False)), aq.REVERSED)

    def test_status_downgrade_confirmed(self):
        self.assertEqual(aq.classify_outcome("STATUS_DOWNGRADE", _al("WATCH", 60, "STRONG"),
                                             _st("WATCH", 58)), aq.CONFIRMED)

    def test_status_downgrade_recovered(self):
        self.assertEqual(aq.classify_outcome("STATUS_DOWNGRADE", _al("WATCH", 60, "STRONG"),
                                             _st("STRONG", 80)), aq.RECOVERED)

    def test_status_downgrade_absent_is_confirmed_deterioration(self):
        # rank(None)=0 <= downgraded rank -> consistent with further deterioration
        self.assertEqual(aq.classify_outcome("STATUS_DOWNGRADE", _al("WATCH", 60, "STRONG"),
                                             _st(None, None, present=False)), aq.CONFIRMED)

    def test_fading_confirmed(self):
        self.assertEqual(aq.classify_outcome("FADING", _al("CAUTION", 55, "WATCH"),
                                             _st("CAUTION", 40, fading=True)), aq.CONFIRMED)

    def test_fading_recovered_requires_meaningful_strength(self):
        self.assertEqual(aq.classify_outcome("FADING", _al("CAUTION", 55, "WATCH"),
                                             _st("STRONG", 78)), aq.RECOVERED)

    def test_fading_tiny_wiggle_not_recovered(self):
        # +2 is below the canonical threshold and status unchanged -> NEUTRAL
        self.assertEqual(aq.classify_outcome("FADING", _al("CAUTION", 55, "WATCH"),
                                             _st("CAUTION", 57)), aq.NEUTRAL)

    def test_rising_small_persistence_confirmed(self):
        self.assertEqual(aq.classify_outcome("RISING", _al("WATCH", 70), _st("WATCH", 71)),
                         aq.CONFIRMED)

    def test_rising_threshold_reversal(self):
        # -4 crosses the canonical -3 threshold -> REVERSED
        self.assertEqual(aq.classify_outcome("RISING", _al("WATCH", 70), _st("WATCH", 66)),
                         aq.REVERSED)

    def test_dropped_remains_absent_confirmed(self):
        self.assertEqual(aq.classify_outcome("DROPPED", _al(None, None, "WATCH"),
                                             _st(None, None, present=False)), aq.CONFIRMED)

    def test_dropped_reentry_recovered(self):
        self.assertEqual(aq.classify_outcome("DROPPED", _al(None, None, "STRONG"),
                                             _st("STRONG", 80)), aq.RECOVERED)

    def test_new_opportunity_persisted_not_confirmed(self):
        # PERSISTED is distinct from CONFIRMED (non-directional existence held)
        self.assertEqual(aq.classify_outcome("NEW_OPPORTUNITY", _al("WATCH", 60),
                                             _st("WATCH", 61)), aq.PERSISTED)

    def test_new_opportunity_gone_reversed(self):
        self.assertEqual(aq.classify_outcome("NEW_OPPORTUNITY", _al("WATCH", 60),
                                             _st(None, None, present=False)), aq.REVERSED)

    def test_signal_removed_stable_neutral(self):
        self.assertEqual(aq.classify_outcome("SIGNAL_REMOVED", _al("WATCH", 60),
                                             _st("WATCH", 61)), aq.NEUTRAL)

    def test_version_mismatch_blocks_numeric(self):
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE", _al("STRONG", 79, "WATCH"),
                                             _st("STRONG", 80, version="2.0")), aq.VERSION_CHANGED)

    def test_missing_version_is_compatible(self):
        # matches canonical compare_opportunities: a missing version is compatible
        self.assertEqual(aq.classify_outcome("STATUS_UPGRADE",
                                             _al("STRONG", 79, "WATCH", version=None),
                                             _st("STRONG", 80, version=None)), aq.CONFIRMED)


class ValiditySelectionTests(unittest.TestCase):
    def test_valid_empty_snapshot_allowed(self):
        self.assertTrue(aq.is_valid_quality_snapshot(_snap(SRC, [])))

    def test_malformed_payload_invalid(self):
        self.assertFalse(aq.is_valid_quality_snapshot({"snapshot_time": SRC, "opportunities": "nope"}))

    def test_missing_timestamp_invalid(self):
        self.assertFalse(aq.is_valid_quality_snapshot({"snapshot_time": None, "opportunities": []}))

    def test_select_rejects_same_timestamp_as_source(self):
        # candidate exactly at source time must be rejected (strict-after)
        obs = aq.select_first_valid_observation([_snap(SRC, [_opp("NVDA", "STRONG", 80)])],
                                                source_time=SRC, cutoff=SRC)
        self.assertIsNone(obs)

    def test_select_earliest_of_out_of_order_and_duplicates(self):
        a = _snap(SRC + dt.timedelta(hours=10), [_opp("NVDA", "STRONG", 81)])
        b = _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 80)])   # earliest valid
        dup = _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "WATCH", 60)])  # duplicate ts
        chosen = aq.select_first_valid_observation([a, b, dup], source_time=SRC, cutoff=SRC)
        self.assertEqual(chosen["snapshot_time"], SRC + dt.timedelta(hours=2))

    def test_select_filters_invalid_before_picking(self):
        bad = {"snapshot_time": SRC + dt.timedelta(hours=1), "opportunities": None}
        good = _snap(SRC + dt.timedelta(hours=3), [_opp("NVDA", "STRONG", 80)])
        chosen = aq.select_first_valid_observation([bad, good], source_time=SRC, cutoff=SRC)
        self.assertEqual(chosen["snapshot_time"], SRC + dt.timedelta(hours=3))


class HorizonTests(unittest.TestCase):
    def test_pending_before_horizon(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH",
                   source_time=NOW - dt.timedelta(hours=1))  # H120 not elapsed
        r = aq.evaluate_alert_at_horizon(a, "H120", [], now=NOW)
        self.assertEqual(r["data_status"], aq.PENDING)

    def test_unavailable_when_no_later_snapshot(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        r = aq.evaluate_alert_at_horizon(a, "NEXT", [], now=NOW)
        self.assertEqual(r["data_status"], aq.UNAVAILABLE)

    def test_next_rejects_source_snapshot_itself(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        # only snapshot available is the source snapshot itself
        r = aq.evaluate_alert_at_horizon(a, "NEXT", [_snap(SRC, [_opp("NVDA", "STRONG", 79)])], now=NOW)
        self.assertEqual(r["data_status"], aq.UNAVAILABLE)

    def test_next_picks_earliest_later(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        snaps = [
            _snap(SRC + dt.timedelta(hours=30), [_opp("NVDA", "WATCH", 60)]),
            _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)]),  # earliest later
        ]
        r = aq.evaluate_alert_at_horizon(a, "NEXT", snaps, now=NOW)
        self.assertEqual(r["data_status"], aq.MATURED)
        self.assertEqual(r["evaluation_time"], SRC + dt.timedelta(hours=2))
        self.assertEqual(r["quality_classification"], aq.CONFIRMED)

    def test_no_leakage_ignores_pre_cutoff(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        snaps = [
            _snap(SRC - dt.timedelta(hours=5), [_opp("NVDA", "CAUTION", 30)]),   # before source
            _snap(SRC + dt.timedelta(hours=10), [_opp("NVDA", "WATCH", 60)]),    # before H72 cutoff
            _snap(SRC + dt.timedelta(hours=80), [_opp("NVDA", "STRONG", 82)]),   # after H72 cutoff
        ]
        r = aq.evaluate_alert_at_horizon(a, "H72", snaps, now=NOW)
        self.assertEqual(r["evaluation_time"], SRC + dt.timedelta(hours=80))
        self.assertEqual(r["quality_classification"], aq.CONFIRMED)

    def test_dropped_invalid_snapshot_not_confirmed(self):
        a = _alert("DROPPED", cur_status=None, cur_score=None, prev_status="WATCH")
        bad = {"snapshot_time": SRC + dt.timedelta(hours=2), "opportunities": "broken"}
        r = aq.evaluate_alert_at_horizon(a, "NEXT", [bad], now=NOW)
        self.assertEqual(r["data_status"], aq.UNAVAILABLE)  # never a fake confirmed drop

    def test_dropped_valid_empty_snapshot_confirms(self):
        a = _alert("DROPPED", cur_status=None, cur_score=None, prev_status="WATCH")
        r = aq.evaluate_alert_at_horizon(a, "NEXT", [_snap(SRC + dt.timedelta(hours=2), [])], now=NOW)
        self.assertEqual(r["data_status"], aq.MATURED)
        self.assertEqual(r["quality_classification"], aq.CONFIRMED)
        self.assertFalse(r["still_present"])

    def test_version_change_yields_null_delta(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        snaps = [_snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 90, version="2.0")])]
        r = aq.evaluate_alert_at_horizon(a, "NEXT", snaps, now=NOW)
        self.assertEqual(r["quality_classification"], aq.VERSION_CHANGED)
        self.assertIsNone(r["score_delta"])


class MaturationTests(unittest.TestCase):
    def _run_mature(self, alert, snaps_first, snaps_second=None):
        persisted = {}

        def _persist(res):
            key = (res["alert_id"], res["evaluation_horizon"])
            if key in persisted:
                return False  # ON CONFLICT DO NOTHING — first observation immutable
            persisted[key] = dict(res)
            return True

        def _go(snaps):
            with (
                mock.patch("db.intelligence_alerts.fetch_alerts_for_maturation", return_value=[alert]),
                mock.patch("db.intelligence_alerts.persist_alert_outcome", side_effect=_persist),
                mock.patch("db.opportunity_snapshots.load_recent_snapshots", return_value=snaps),
            ):
                return aq.mature_alert_outcomes(now=NOW)

        n1 = _go(snaps_first)
        n2 = _go(snaps_second if snaps_second is not None else snaps_first)
        return n1, n2, persisted

    def test_idempotent_and_first_observation_immutable(self):
        a = _alert("STATUS_UPGRADE", cur_status="STRONG", cur_score=79, prev_status="WATCH")
        first = [_snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)])]
        # second run: a later, stronger snapshot appears — must NOT rewrite history
        second = first + [_snap(SRC + dt.timedelta(hours=6), [_opp("NVDA", "STRONG", 95)])]
        n1, n2, persisted = self._run_mature(a, first, second)
        self.assertEqual(n1, 1)   # NEXT matured; later horizons UNAVAILABLE -> skipped
        self.assertEqual(n2, 0)   # nothing new written
        rec = persisted[(1, "NEXT")]
        self.assertEqual(rec["evaluation_time"], SRC + dt.timedelta(hours=2))  # unchanged
        self.assertEqual(rec["subsequent_score"], 81)

    def test_version_change_alert_skipped(self):
        a = _alert("VERSION_CHANGED", cur_status="STRONG", cur_score=79)
        with (
            mock.patch("db.intelligence_alerts.fetch_alerts_for_maturation", return_value=[a]),
            mock.patch("db.intelligence_alerts.persist_alert_outcome") as p,
            mock.patch("db.opportunity_snapshots.load_recent_snapshots", return_value=[]),
        ):
            n = aq.mature_alert_outcomes()
        self.assertEqual(n, 0)
        p.assert_not_called()


class QualityAggregationTests(unittest.TestCase):
    def _summary(self, pair, tick, run, overall, event, horizon, dist, min_sample=10):
        cur = mock.MagicMock()
        cur.fetchone.side_effect = [pair, tick, run]
        cur.fetchall.side_effect = [overall, event, horizon, dist]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        from db import intelligence_alerts as ia
        with mock.patch.object(ia, "get_neon_conn", return_value=conn):
            return ia.get_alert_quality_summary(min_sample=min_sample), cur

    def test_read_only_and_denominator_excludes_version_changed(self):
        q, cur = self._summary(
            pair=(40, 30, 20), tick=(5, 0), run=(100, 40, 10, 8),
            overall=[("CONFIRMED", 12), ("REVERSED", 3), ("VERSION_CHANGED", 5)],
            event=[("STATUS_UPGRADE", "CONFIRMED", 12), ("STATUS_UPGRADE", "REVERSED", 3),
                   ("STATUS_UPGRADE", "VERSION_CHANGED", 5)],
            horizon=[("NEXT", "CONFIRMED", 12), ("NEXT", "REVERSED", 3)],
            dist=[("STATUS_UPGRADE", 15)])
        self.assertEqual(q["matured"], 20)       # all persisted matured rows
        self.assertEqual(q["evaluable"], 15)     # excludes 5 VERSION_CHANGED
        self.assertEqual(q["version_changed"], 5)
        # confirmation rate is over EVALUABLE (15), not matured (20)
        self.assertAlmostEqual(q["confirmation_rate"], 12 / 15)
        self.assertEqual(q["pending"], 10)
        self.assertEqual(q["unavailable"], 10)   # elapsed(30) - matured(20)
        for c in cur.execute.call_args_list:
            self.assertNotRegex(c[0][0].upper(), r"\b(INSERT|UPDATE|DELETE)\b")

    def test_small_evaluable_is_insufficient(self):
        q, _ = self._summary(
            pair=(8, 8, 2), tick=(2, 0), run=(8, 0, 0, 2),
            overall=[("CONFIRMED", 6)],
            event=[("STATUS_UPGRADE", "CONFIRMED", 6)],
            horizon=[("NEXT", "CONFIRMED", 6)],
            dist=[("STATUS_UPGRADE", 6)])
        self.assertIsNone(q["confirmation_rate"])
        self.assertEqual(q["by_event_type"][0]["assessment"], "INSUFFICIENT_SAMPLE")


class HorizonDefinitionTests(unittest.TestCase):
    def test_canonical_horizons_single_source(self):
        hs = aq.get_quality_horizons()
        self.assertEqual([h["key"] for h in hs], ["NEXT", "H24", "H72", "H120"])
        self.assertEqual([h["offset_hours"] for h in hs], [0, 24, 72, 120])
        # labels must be honest about elapsed time (no "1 day"/"3 day")
        for h in hs:
            self.assertNotRegex(h["label"].lower(), r"\bday")


if __name__ == "__main__":
    unittest.main()
