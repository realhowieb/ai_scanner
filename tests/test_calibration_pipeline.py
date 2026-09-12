"""Run 20D — HSF calibration data-pipeline integrity."""
import unittest
from unittest import mock

from analytics import hsf_calibration as hc
from db import signal_outcomes as so
from ui import opportunities as op

_BANNED_OUTCOME_KEYS = {
    "return_1d", "return_3d", "return_5d", "mfe_1d", "mfe_5d", "mae_5d",
    "positive", "hit", "outcome_computed_at", "future_return", "future_label",
}


def _opp():
    return op.build_opportunities(
        {"top_setups": [("NVDA", 74)], "golden": ["NVDA"], "gainers": [("NVDA", 2.4)],
         "picks": [{"symbol": "NVDA", "prob": 78}]}, top_n=1)[0]


class FreezePayloadTests(unittest.TestCase):
    def test_freeze_payload_has_score_version_components_and_no_outcome_fields(self):
        captured = {}

        def fake_freeze_signal(**kw):
            captured.update(kw)
            return True

        with mock.patch.object(so, "freeze_signal", side_effect=fake_freeze_signal):
            import datetime as dt
            ok = so.freeze_opportunity(dt.datetime(2026, 9, 12, tzinfo=dt.timezone.utc), _opp())
        self.assertTrue(ok)
        self.assertEqual(captured["source"], "opportunity")
        self.assertEqual(captured["signal_type"], "hsf_opportunity")
        raw, ind = captured["raw_signal"], captured["indicators"]
        # Signal-time fields present:
        self.assertEqual(raw["score_version"], op.HSF_SCORE_VERSION)
        for k in ("signals_component", "model_component", "momentum_component", "fading_penalty"):
            self.assertIn(k, raw["score_components"])
        self.assertIn("signals", ind)
        self.assertIn("n_signals", ind)
        # NO future/outcome fields anywhere in the frozen payload:
        self.assertFalse(_BANNED_OUTCOME_KEYS & set(raw))
        self.assertFalse(_BANNED_OUTCOME_KEYS & set(ind))

    def test_freeze_is_idempotent_by_snapshot_epoch(self):
        # source_event_id is the snapshot epoch, so re-freezing the same snapshot
        # targets the same UNIQUE row (ON CONFLICT DO NOTHING). Two different
        # times -> two different event ids (genuinely separate observations).
        seen = []

        def fake(**kw):
            seen.append(kw["source_event_id"])
            return True

        import datetime as dt
        with mock.patch.object(so, "freeze_signal", side_effect=fake):
            so.freeze_opportunity(dt.datetime(2026, 9, 12, 12, 0, tzinfo=dt.timezone.utc), _opp())
            so.freeze_opportunity(dt.datetime(2026, 9, 12, 12, 0, tzinfo=dt.timezone.utc), _opp())
            so.freeze_opportunity(dt.datetime(2026, 9, 12, 13, 0, tzinfo=dt.timezone.utc), _opp())
        self.assertEqual(seen[0], seen[1])       # same snapshot -> same key
        self.assertNotEqual(seen[0], seen[2])    # later snapshot -> new key

    def test_freeze_no_op_without_time_or_ticker(self):
        self.assertFalse(so.freeze_opportunity(None, _opp()))
        self.assertFalse(so.freeze_opportunity("t", {"ticker": ""}))


class FreezeOwnershipTests(unittest.TestCase):
    def test_cron_freeze_latest_reuses_canonical_build_and_freeze(self):
        from analytics import opportunity_freeze as ofz
        data = {"snapshot_time": "T", "top_setups": [("NVDA", 74)], "golden": ["NVDA"]}
        with (
            mock.patch("ui.market_brief._compute_brief", return_value=data),
            mock.patch("db.signal_outcomes.freeze_opportunities", return_value=3) as fz,
        ):
            n = ofz.freeze_latest_opportunities()
        self.assertEqual(n, 3)
        self.assertTrue(fz.called)

    def test_cron_freeze_safe_when_no_data_or_error(self):
        from analytics import opportunity_freeze as ofz
        with mock.patch("ui.market_brief._compute_brief", return_value=None):
            self.assertEqual(ofz.freeze_latest_opportunities(), 0)
        with mock.patch("ui.market_brief._compute_brief", side_effect=RuntimeError("x")):
            self.assertEqual(ofz.freeze_latest_opportunities(), 0)


class PipelineIntegrityTests(unittest.TestCase):
    def _row(self, **kw):
        base = {"ticker": "T", "raw_signal": {"hsf_score": 80, "score_version": "1.0",
                "score_components": {"signals_component": 24, "model_component": 20,
                                     "momentum_component": 5, "fading_penalty": 0}},
                "indicators": {"n_signals": 3, "signals": ["breakout"]},
                "setup_score": 70, "prebreakout_prob": 78,
                "return_5d": None, "mfe_5d": None, "mae_5d": None, "outcome_computed_at": None}
        base.update(kw)
        return base

    def test_pending_is_not_a_failure(self):
        rec = hc.normalize_row(self._row())  # pending
        self.assertFalse(rec["matured"])
        self.assertIsNone(rec["positive"])   # NOT counted as negative

    def test_matured_recognized_and_positive_from_mfe(self):
        import datetime as dt
        rec = hc.normalize_row(self._row(mfe_5d=0.06, outcome_computed_at=dt.datetime(2026, 9, 20)))
        self.assertTrue(rec["matured"])
        self.assertTrue(rec["positive"])

    def test_old_row_without_version_is_unknown_not_v1(self):
        row = self._row()
        row["raw_signal"] = dict(row["raw_signal"])
        row["raw_signal"].pop("score_version")
        rec = hc.normalize_row(row)
        self.assertIsNone(rec["score_version"])
        ds = hc.build_calibration_dataset([row])
        self.assertIn("unknown", ds["versions"])   # legacy -> unknown, not '1.0'

    def test_data_quality_flags_missing_score_and_components(self):
        r_missing_score = hc.normalize_row(self._row())
        r_missing_score["hsf_score"] = None
        r_missing_comp = hc.normalize_row(self._row())
        for k in ("signals_component", "model_component", "momentum_component", "fading_penalty"):
            r_missing_comp[k] = None
        warns = hc.data_quality_checks([r_missing_score, r_missing_comp])
        self.assertTrue(any("missing hsf_score" in w for w in warns))
        self.assertTrue(any("missing score components" in w for w in warns))

    def test_invalid_score_flagged(self):
        rec = hc.normalize_row(self._row())
        rec["hsf_score"] = 150
        self.assertTrue(any("0-100" in w for w in hc.data_quality_checks([rec])))

    def test_empty_and_db_unavailable_do_not_fabricate(self):
        ds = hc.build_calibration_dataset([])
        self.assertEqual(ds["n_matured"], 0)
        self.assertEqual(hc.evaluate_calibration([])["brier"], None)
        # DB unavailable path (fetch returns []) -> empty, no crash.
        with mock.patch("db.signal_outcomes.fetch_opportunity_outcomes", return_value=[]):
            ds2 = hc.build_calibration_dataset()
        self.assertEqual(ds2["n_total"], 0)


if __name__ == "__main__":
    unittest.main()
