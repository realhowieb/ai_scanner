import datetime as dt
import unittest

from analytics import post_rescore_audit as a


def row(i=1, **kw):
    return {"id": i, "source": "opportunity", "source_event_id": 1789383600,
            "signal_type": "hsf_opportunity", "ticker": "ABC",
            "fired_at": dt.datetime(2026, 9, 14, 11, tzinfo=dt.timezone.utc),
            "outcome_computed_at": a.REPAIR_START + dt.timedelta(minutes=1),
            "return_1d": .01, "return_3d": -.02, "return_5d": .03,
            "raw_signal": {"hsf_score": 80}, "benchmark_return_5d": .02, **kw}


class PostRescoreTests(unittest.TestCase):
    def test_cohort_window_half_open(self):
        self.assertTrue(a.recovered(row(outcome_computed_at=a.REPAIR_START)))
        self.assertFalse(a.recovered(row(outcome_computed_at=a.REPAIR_END)))
        self.assertFalse(a.recovered(row(outcome_computed_at=None)))

    def test_before_is_copy_and_only_masks_repair(self):
        original = row()
        valid = row(2, outcome_computed_at=a.REPAIR_START - dt.timedelta(days=1))
        before = a.before_repair([original, valid])
        self.assertIsNone(before[0]["return_5d"])
        self.assertEqual(before[1]["return_5d"], .03)
        self.assertEqual(original["return_5d"], .03)
        self.assertEqual(before[0]["raw_signal"], original["raw_signal"])

    def test_dedup_happens_before_label_selection(self):
        first = row(return_5d=None)
        later = row(2, fired_at=first["fired_at"] + dt.timedelta(hours=1))
        self.assertEqual(a.signal_days([later, first]), [first])

    def test_open_or_premature_labels_are_not_evaluation_rows(self):
        now = a.REPAIR_END
        self.assertTrue(a.usable(row(), 5, now))
        self.assertFalse(a.usable(row(outcome_computed_at=row()["fired_at"]), 5, now))
        self.assertFalse(a.usable(row(return_5d=float("nan")), 5, now))

    def test_single_class_auc_and_non_probability_score(self):
        result = a.metrics([row(hsf_score=80)], 5, a.REPAIR_END, "hsf_score")
        self.assertIsNone(result["roc_auc"])
        self.assertNotIn("brier", result)
        self.assertEqual(result["support"], "UNDER_SAMPLED")

    def test_missing_probability_not_filled(self):
        self.assertEqual(a.metrics([row(prebreakout_prob=None)], 5, a.REPAIR_END, "prebreakout_prob")["n"], 0)

    def test_finite_numbers_only(self):
        for v in (None, "bad", float("nan"), float("inf")):
            self.assertIsNone(a.number(v))
