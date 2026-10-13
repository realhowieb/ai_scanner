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
    _RUNS = [{"id": 7, "created_at": "T7"}, {"id": 6, "created_at": "T6"}]

    def test_cron_freezes_latest_run_with_scanner_scores(self):
        from analytics import opportunity_freeze as ofz
        opps = [{"ticker": "NVDA", "score": 77}]
        with (
            mock.patch.object(ofz, "market_runs", return_value=self._RUNS),
            mock.patch.object(ofz, "scanner_opportunities", return_value=opps) as sc,
            mock.patch("db.signal_outcomes.freeze_opportunities", return_value=3) as fz,
        ):
            n = ofz.freeze_latest_opportunities()
        self.assertEqual(n, 3)
        sc.assert_called_once_with(7, ofz.FREEZE_TOP_N)
        fz.assert_called_once_with("T7", opps, signal_type=so.SCANNER_SIGNAL_TYPE)

    def test_scanner_opportunities_use_the_scanner_ranking(self):
        import pandas as pd

        from analytics import opportunity_freeze as ofz
        df = pd.DataFrame([
            {"Ticker": "NVDA", "IsBreakout": True, "EMACross": "Golden",
             "PreBreakoutProb%": 78, "BreakoutScore": 74, "PctChange": 2.4},
            {"Ticker": "ZZZ", "IsBreakout": False, "PctChange": 0.1},
        ])
        with mock.patch("ui.market_scans._run_df_uncached", return_value=df):
            opps = ofz.scanner_opportunities(7)
        from ui.results_intelligence import consolidate_scanner_results
        self.assertEqual(opps, consolidate_scanner_results(df.to_dict(orient="records")))

    def test_cron_freeze_safe_when_no_data_or_error(self):
        from analytics import opportunity_freeze as ofz
        with mock.patch.object(ofz, "market_runs", return_value=[]):
            self.assertEqual(ofz.freeze_latest_opportunities(), 0)
        with mock.patch.object(ofz, "market_runs", side_effect=RuntimeError("x")):
            self.assertEqual(ofz.freeze_latest_opportunities(), 0)
        with (
            mock.patch.object(ofz, "market_runs", return_value=self._RUNS),
            mock.patch.object(ofz, "scanner_opportunities", return_value=[]),
        ):
            self.assertEqual(ofz.freeze_latest_opportunities(), 0)

    def test_scanner_freeze_is_tagged_and_keyed_apart_from_brief_rows(self):
        import datetime as dt
        seen = []
        with mock.patch.object(so, "freeze_signal", side_effect=lambda **kw: seen.append(kw) or True):
            t = dt.datetime(2026, 10, 13, 13, 35, tzinfo=dt.timezone.utc)
            so.freeze_opportunity(t, _opp())
            so.freeze_opportunity(t, _opp(), signal_type=so.SCANNER_SIGNAL_TYPE)
        self.assertEqual([kw["signal_type"] for kw in seen], ["hsf_opportunity", "hsf_scanner"])
        self.assertEqual([kw["raw_signal"]["score_source"] for kw in seen], ["brief", "scanner"])
        self.assertEqual(seen[0]["source_event_id"], seen[1]["source_event_id"])

    def test_every_opportunity_reader_drops_brief_rows_superseded_by_scanner_rows(self):
        self.assertIn("hsf_scanner", so._SCANNER_SUPERSEDES_BRIEF)
        self.assertIn(so._SCANNER_SUPERSEDES_BRIEF, so._OUTCOME_ALL_SQL)
        self.assertIn(so._SCANNER_SUPERSEDES_BRIEF, so._OUTCOME_TICKER_SQL)
        for fn in (so.fetch_opportunity_outcomes, so.fetch_ticker_opportunity_history,
                   so.fetch_opportunity_observations):
            cur = mock.MagicMock()
            cur.fetchall.return_value = []
            conn = mock.MagicMock()
            conn.cursor.return_value = cur
            with mock.patch.object(so, "get_neon_conn", return_value=conn), \
                    mock.patch.object(so, "_ensure_schema"):
                fn("NVDA") if fn is so.fetch_ticker_opportunity_history else fn()
            sql = cur.execute.call_args[0][0]
            self.assertIn(so._SCANNER_SUPERSEDES_BRIEF, sql, fn.__name__)

    def test_brief_render_no_longer_freezes(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "ui" / "market_brief.py").read_text()
        self.assertNotIn("freeze_opportunities(", src)


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
