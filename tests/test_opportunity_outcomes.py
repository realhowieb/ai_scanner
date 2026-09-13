"""Run 25 — HSF opportunity outcome intelligence (deterministic)."""
import datetime as dt
import unittest
from unittest import mock

from analytics import opportunity_outcomes as oo

SRC = dt.datetime(2026, 9, 1, 16, 0, tzinfo=dt.timezone.utc)
NOW = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)


def _obs(*, ticker="NVDA", score=72, status="WATCH", version="1.0", fading=False,
         n_signals=2, oid=1, when=SRC):
    return {"observation_id": oid, "ticker": ticker, "snapshot_time": when, "score": score,
            "status": status, "score_version": version, "signals": ["breakout", "gainer"][:n_signals],
            "n_signals": n_signals, "fading": fading}


def _snap(when, opps):
    return {"snapshot_time": when, "opportunities": opps}


def _opp(ticker, status, score, version="1.0", fading=False):
    return {"ticker": ticker, "status": status, "score": score,
            "score_version": version, "fading": fading}


def _init(status, score, version="1.0", fading=False):
    return {"status": status, "score": score, "score_version": version, "fading": fading}


def _sub(status, score, *, present=True, version="1.0", fading=False):
    return {"present": present, "status": status, "score": score,
            "score_version": version, "fading": fading}


class EligibilityTests(unittest.TestCase):
    def test_valid_eligible(self):
        self.assertTrue(oo.is_eligible_opportunity_observation(_obs()))

    def test_missing_timestamp_ineligible(self):
        self.assertFalse(oo.is_eligible_opportunity_observation(_obs(when=None)))

    def test_missing_ticker_ineligible(self):
        self.assertFalse(oo.is_eligible_opportunity_observation(_obs(ticker="")))

    def test_missing_state_ineligible(self):
        o = _obs(); o["score"] = None
        self.assertFalse(oo.is_eligible_opportunity_observation(o))

    def test_unknown_version_ineligible(self):
        self.assertFalse(oo.is_eligible_opportunity_observation(_obs(version="9.9")))

    def test_missing_version_is_eligible(self):
        self.assertTrue(oo.is_eligible_opportunity_observation(_obs(version=None)))


class ClassifyTests(unittest.TestCase):
    def test_strengthened_by_status(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70), _sub("STRONG", 80)),
                         oo.STRENGTHENED)

    def test_strengthened_by_score(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70), _sub("WATCH", 74)),
                         oo.STRENGTHENED)

    def test_persisted(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("STRONG", 80), _sub("STRONG", 81)),
                         oo.PERSISTED)

    def test_weakened(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("STRONG", 80), _sub("WATCH", 66)),
                         oo.WEAKENED)

    def test_faded_precedence_over_weakened(self):
        # subsequent fading True wins even though score also dropped
        self.assertEqual(oo.classify_opportunity_outcome(_init("STRONG", 80), _sub("WATCH", 66, fading=True)),
                         oo.FADED)

    def test_dropped_when_absent(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70), _sub(None, None, present=False)),
                         oo.DROPPED)

    def test_recovered_from_weak(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("CAUTION", 48, fading=True),
                                                         _sub("STRONG", 78)), oo.RECOVERED)

    def test_healthy_improvement_is_strengthened_not_recovered(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70), _sub("STRONG", 82)),
                         oo.STRENGTHENED)

    def test_version_changed_precedence(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70),
                                                         _sub("STRONG", 90, version="2.0")),
                         oo.VERSION_CHANGED)

    def test_degraded_tickup_still_degraded_is_strengthened_not_recovered(self):
        # CAUTION 44 -> CAUTION 48: improves but stays in a degraded tier => not RECOVERED
        self.assertEqual(oo.classify_opportunity_outcome(_init("CAUTION", 44, fading=True),
                                                         _sub("CAUTION", 48)), oo.STRENGTHENED)

    def test_recovered_requires_healthy_tier(self):
        # CAUTION+fading -> WATCH not fading with real improvement => RECOVERED
        self.assertEqual(oo.classify_opportunity_outcome(_init("CAUTION", 48, fading=True),
                                                         _sub("WATCH", 62)), oo.RECOVERED)

    def test_score_increase_exactly_threshold_strengthened(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("WATCH", 70), _sub("WATCH", 73)),
                         oo.STRENGTHENED)

    def test_score_decrease_exactly_threshold_weakened(self):
        self.assertEqual(oo.classify_opportunity_outcome(_init("STRONG", 80), _sub("STRONG", 77)),
                         oo.WEAKENED)

    def test_version_change_with_absence_is_version_changed(self):
        # absent ticker but incompatible universe version -> VERSION_CHANGED, not DROPPED
        self.assertEqual(
            oo.classify_opportunity_outcome(_init("WATCH", 70, version="1.0"),
                                            _sub(None, None, present=False, version="2.0")),
            oo.VERSION_CHANGED)


class HorizonTests(unittest.TestCase):
    def test_pending(self):
        o = _obs(when=NOW - dt.timedelta(hours=1))
        r = oo.evaluate_observation_at_horizon(o, "H120", [], now=NOW)
        self.assertEqual(r["data_status"], oo.PENDING)

    def test_unavailable_no_snapshot(self):
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT", [], now=NOW)
        self.assertEqual(r["data_status"], oo.UNAVAILABLE)

    def test_rejects_source_snapshot(self):
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT",
                                               [_snap(SRC, [_opp("NVDA", "WATCH", 72)])], now=NOW)
        self.assertEqual(r["data_status"], oo.UNAVAILABLE)

    def test_picks_earliest_later(self):
        snaps = [_snap(SRC + dt.timedelta(hours=30), [_opp("NVDA", "WATCH", 60)]),
                 _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)])]
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT", snaps, now=NOW)
        self.assertEqual(r["evaluation_time"], SRC + dt.timedelta(hours=2))
        self.assertEqual(r["outcome_classification"], oo.STRENGTHENED)
        self.assertEqual(r["status_transition"], "WATCH→STRONG")

    def test_no_leakage_pre_cutoff_ignored(self):
        snaps = [_snap(SRC - dt.timedelta(hours=5), [_opp("NVDA", "STRONG", 90)]),
                 _snap(SRC + dt.timedelta(hours=10), [_opp("NVDA", "WATCH", 70)]),
                 _snap(SRC + dt.timedelta(hours=80), [_opp("NVDA", "WATCH", 71)])]
        r = oo.evaluate_observation_at_horizon(_obs(), "H72", snaps, now=NOW)
        self.assertEqual(r["evaluation_time"], SRC + dt.timedelta(hours=80))

    def test_invalid_snapshot_absence_is_unavailable_not_dropped(self):
        bad = {"snapshot_time": SRC + dt.timedelta(hours=2), "opportunities": "broken"}
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT", [bad], now=NOW)
        self.assertEqual(r["data_status"], oo.UNAVAILABLE)

    def test_valid_empty_snapshot_is_dropped(self):
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT",
                                               [_snap(SRC + dt.timedelta(hours=2), [])], now=NOW)
        self.assertEqual(r["outcome_classification"], oo.DROPPED)
        self.assertFalse(r["still_present"])

    def test_version_change_null_delta(self):
        snaps = [_snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 95, version="2.0")])]
        r = oo.evaluate_observation_at_horizon(_obs(), "NEXT", snaps, now=NOW)
        self.assertEqual(r["outcome_classification"], oo.VERSION_CHANGED)
        self.assertIsNone(r["score_delta"])

    def test_out_of_order_and_duplicates_deterministic(self):
        a = _snap(SRC + dt.timedelta(hours=10), [_opp("NVDA", "WATCH", 60)])
        b = _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)])
        dup = _snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "CAUTION", 40)])
        r1 = oo.evaluate_observation_at_horizon(_obs(), "NEXT", [a, b, dup], now=NOW)
        r2 = oo.evaluate_observation_at_horizon(_obs(), "NEXT", [dup, a, b], now=NOW)
        self.assertEqual(r1["evaluation_time"], r2["evaluation_time"])
        self.assertEqual(r1["evaluation_time"], SRC + dt.timedelta(hours=2))


class ReconstructionTests(unittest.TestCase):
    def test_reversed_db_order_identical(self):
        rows = [
            {"observation_id": 2, "ticker": "NVDA", "snapshot_time": SRC, "score": 60,
             "status": "WATCH", "score_version": "1.0", "signals": ["gainer"], "fading": False},
            {"observation_id": 1, "ticker": "NVDA", "snapshot_time": SRC, "score": 80,
             "status": "STRONG", "score_version": "1.0", "signals": ["breakout"], "fading": False},
        ]
        a = oo._reconstruct_snapshots(rows)
        b = oo._reconstruct_snapshots(list(reversed(rows)))
        # deterministic: lowest observation_id (1 -> STRONG 80) wins regardless of order
        self.assertEqual(a, b)
        self.assertEqual(a[0]["opportunities"][0]["score"], 80)

    def test_source_dedupe_one_logical_observation(self):
        rows = [
            _obs(oid=5, score=70), _obs(oid=3, score=70),  # same ticker/time/version
        ]
        deduped = oo._dedupe_source_observations(rows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["observation_id"], 3)  # lowest id wins

    def test_absent_from_incompatible_version_universe(self):
        obs = _obs(version="1.0")  # NVDA
        snaps = [_snap(SRC + dt.timedelta(hours=2),
                       [_opp("AMD", "STRONG", 80, version="2.0")])]  # NVDA absent; universe v2.0
        r = oo.evaluate_observation_at_horizon(obs, "NEXT", snaps, now=NOW)
        self.assertEqual(r["outcome_classification"], oo.VERSION_CHANGED)
        self.assertIsNone(r["score_delta"])


class BandBucketTests(unittest.TestCase):
    def test_score_bands_boundaries(self):
        # bands aligned to canonical thresholds: STRONG>=75, CAUTION<50
        self.assertEqual(oo.score_band(75), "75+")
        self.assertEqual(oo.score_band(74.99), "60-74")
        self.assertEqual(oo.score_band(60), "60-74")
        self.assertEqual(oo.score_band(59.99), "50-59")
        self.assertEqual(oo.score_band(50), "50-59")
        self.assertEqual(oo.score_band(49.99), "<50")
        self.assertEqual(oo.score_band(100), "75+")

    def test_score_bands_malformed(self):
        self.assertIsNone(oo.score_band(None))
        self.assertIsNone(oo.score_band("abc"))
        self.assertIsNone(oo.score_band(-5))
        self.assertIsNone(oo.score_band(101))

    def test_signal_buckets(self):
        self.assertEqual(oo.signal_count_bucket(1), "1")
        self.assertEqual(oo.signal_count_bucket(2), "2")
        self.assertEqual(oo.signal_count_bucket(5), "3+")
        self.assertEqual(oo.signal_count_bucket(0), "0")       # zero kept distinct
        self.assertEqual(oo.signal_count_bucket(None), "0")

    def test_signal_normalization_dedupes(self):
        self.assertEqual(oo.normalized_signals(["breakout", "Breakout", "volume ", "volume"]),
                         ["breakout", "volume"])
        # frozen signal count uses the de-duplicated set, never the raw length
        o = _obs()
        o["signals"] = ["breakout", "breakout", "gainer"]
        r = oo.evaluate_observation_at_horizon(
            o, "NEXT", [_snap(SRC + dt.timedelta(hours=2), [_opp("NVDA", "STRONG", 81)])], now=NOW)
        self.assertEqual(r["initial_signal_count"], 2)


class MaturationTests(unittest.TestCase):
    def test_idempotent_and_immutable(self):
        # Subsequent observations come from the SAME frozen store, so the later
        # state must be another frozen observation (oid=2 at +2h, STRONG 81).
        persisted = {}

        def _persist(res):
            key = (res["opportunity_observation_id"], res["evaluation_horizon"])
            if key in persisted:
                return False
            persisted[key] = dict(res)
            return True

        def _go(obs, existing):
            with (
                mock.patch("db.signal_outcomes.fetch_opportunity_observations", return_value=obs),
                mock.patch("db.opportunity_outcomes.persisted_outcome_keys", return_value=existing),
                mock.patch("db.opportunity_outcomes.persist_opportunity_outcome", side_effect=_persist),
            ):
                return oo.mature_opportunity_outcomes(now=NOW)

        run1 = [_obs(oid=1, when=SRC, score=72, status="WATCH"),
                _obs(oid=2, when=SRC + dt.timedelta(hours=2), score=81, status="STRONG")]
        m1 = _go(run1, set())
        self.assertEqual(m1["matured"], 1)             # oid=1 NEXT -> STRONG 81
        self.assertEqual(persisted[(1, "NEXT")]["subsequent_score"], 81)
        self.assertEqual(persisted[(1, "NEXT")]["outcome_classification"], oo.STRENGTHENED)

        # Rerun: a newer, STRONGER later observation (oid=3 @ +6h, 95) appears, and
        # (1,NEXT) is already persisted -> it must NOT be recomputed/rewritten.
        run2 = run1 + [_obs(oid=3, when=SRC + dt.timedelta(hours=6), score=95, status="STRONG")]
        _go(run2, {(1, "NEXT")})
        self.assertEqual(persisted[(1, "NEXT")]["subsequent_score"], 81)  # immutable: still +2h/81

    def test_ineligible_skipped(self):
        obs = [_obs(oid=2, version="9.9")]
        with (
            mock.patch("db.signal_outcomes.fetch_opportunity_observations", return_value=obs),
            mock.patch("db.opportunity_outcomes.persisted_outcome_keys", return_value=set()),
            mock.patch("db.opportunity_outcomes.persist_opportunity_outcome") as p,
        ):
            m = oo.mature_opportunity_outcomes(now=NOW)
        self.assertEqual(m["skipped"], 1)
        p.assert_not_called()


class AggregationTests(unittest.TestCase):
    def _summary(self, overall, status, band, signal, regime, horizon, min_sample=10):
        cur = mock.MagicMock()
        cur.fetchall.side_effect = [overall, status, band, signal, regime, horizon]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        from db import opportunity_outcomes as db_oo
        with mock.patch.object(db_oo, "get_neon_conn", return_value=conn):
            return db_oo.get_opportunity_outcome_summary(min_sample=min_sample), cur

    def test_read_only_and_denominator_excludes_version_changed(self):
        q, cur = self._summary(
            overall=[("PERSISTED", 50), ("STRENGTHENED", 20), ("WEAKENED", 10), ("VERSION_CHANGED", 5)],
            status=[("STRONG", "PERSISTED", 50), ("STRONG", "WEAKENED", 10)],
            band=[("80+", "PERSISTED", 40)],
            signal=[("3+", "STRENGTHENED", 20)],
            regime=[("UNKNOWN", "PERSISTED", 50)],
            horizon=[("NEXT", "PERSISTED", 50)])
        self.assertEqual(q["matured"], 85)
        self.assertEqual(q["comparable"], 80)  # excludes 5 VERSION_CHANGED
        # follow-through = strengthened+persisted (RECOVERED is separate)
        self.assertEqual(q["by_status"][0]["follow_through_rate"], 50 / 60)
        for c in cur.execute.call_args_list:
            self.assertNotRegex(c[0][0].upper(), r"\b(INSERT|UPDATE|DELETE)\b")

    def test_small_sample_insufficient(self):
        q, _ = self._summary(
            overall=[("PERSISTED", 6)], status=[("STRONG", "PERSISTED", 6)],
            band=[("80+", "PERSISTED", 6)], signal=[("2", "PERSISTED", 6)],
            regime=[("UNKNOWN", "PERSISTED", 6)], horizon=[("NEXT", "PERSISTED", 6)])
        self.assertEqual(q["by_status"][0]["assessment"], "INSUFFICIENT_SAMPLE")
        self.assertIsNone(q["by_status"][0]["follow_through_rate"])


if __name__ == "__main__":
    unittest.main()
