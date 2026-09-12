"""Run 21 — HSF Stock Intelligence: builder, lifecycle, read-only guarantees."""
import datetime as dt
import unittest
from unittest import mock

from ui import stock_intelligence as si


def _row(**kw):
    r = {"Ticker": "NVDA", "IsBreakout": True, "EMACross": "Golden",
         "PreBreakoutProb%": 78, "BreakoutScore": 74, "PctChange": 4.3, "Last": 120.0}
    r.update(kw)
    return r


def _hist():
    t0 = dt.datetime(2026, 9, 12, 9, 42, tzinfo=dt.timezone.utc)
    return [
        {"time": t0, "score": 58, "status": "WATCH", "score_version": "1.0", "signals": ["prebreakout"]},
        {"time": t0 + dt.timedelta(minutes=36), "score": 67, "status": "WATCH", "score_version": "1.0",
         "signals": ["prebreakout", "golden_cross"]},
        {"time": t0 + dt.timedelta(hours=1, minutes=21), "score": 78, "status": "STRONG", "score_version": "1.0",
         "signals": ["prebreakout", "golden_cross", "breakout"]},
    ]


class BuilderTests(unittest.TestCase):
    def test_full_data_reuses_canonical_score(self):
        from ui import opportunities as op
        intel = si.build_stock_intelligence("NVDA", current_row=_row(), history=_hist(),
                                            regime="RISK-ON", calibration_records=[], earnings_days=1)
        self.assertTrue(intel["has_opportunity"])
        self.assertEqual(intel["score_version"], op.HSF_SCORE_VERSION)
        self.assertIn("confirming signals", " ".join(intel["reasons"]))
        self.assertEqual(intel["market_regime"], "RISK-ON")
        self.assertIsNotNone(intel["score_components"])

    def test_ticker_only_is_safe(self):
        intel = si.build_stock_intelligence("XYZ")
        self.assertFalse(intel["has_opportunity"])
        self.assertIsNone(intel["hsf_score"])
        self.assertEqual(intel["lifecycle"], [])

    def test_breakout_score_only_and_prebreakout_only(self):
        b = si.build_stock_intelligence("A", current_row={"Ticker": "A", "BreakoutScore": 60})
        self.assertTrue(b["has_opportunity"])
        p = si.build_stock_intelligence("B", current_row={"Ticker": "B", "PreBreakoutProb%": 70})
        self.assertTrue(p["has_opportunity"])
        self.assertEqual(p["model"]["prebreakout_prob"], 70)

    def test_falls_back_to_history_when_no_row(self):
        intel = si.build_stock_intelligence("NVDA", history=_hist())
        self.assertTrue(intel["has_opportunity"])
        self.assertTrue(intel["from_history"])
        self.assertEqual(intel["hsf_score"], 78)   # latest observation
        self.assertIsNone(intel["price"])          # no live price

    def test_hsf_score_is_not_a_probability(self):
        intel = si.build_stock_intelligence("NVDA", current_row=_row())
        # Model prob and HSF score are separate fields, never conflated.
        self.assertNotEqual(intel["hsf_score"], intel["model"]["prebreakout_prob"])

    def test_fading_and_earnings_produce_risks_and_watch(self):
        intel = si.build_stock_intelligence(
            "F", current_row={"Ticker": "F", "IsBreakout": True, "BreakoutScore": 55, "PctChange": -3.0},
            earnings_days=1)
        self.assertTrue(any("Earnings" in r for r in intel["risks"]))
        self.assertTrue(any("Earnings" in w for w in intel["watch_next"]))

    def test_missing_calibration_and_regime_are_safe(self):
        intel = si.build_stock_intelligence("NVDA", current_row=_row(), calibration_records=None, regime=None)
        self.assertIsNone(intel["historical_context"])
        self.assertIsNone(intel["market_regime"])

    def test_watch_next_has_no_execution_language(self):
        intel = si.build_stock_intelligence("NVDA", current_row=_row())
        joined = " ".join(intel["watch_next"]).lower()
        for banned in ("buy", "sell", "$", "stop", "entry", "target"):
            self.assertNotIn(banned, joined)


class LifecycleTests(unittest.TestCase):
    def test_first_rising_and_transition_with_signal_add(self):
        life = si.reconstruct_lifecycle(_hist())
        self.assertEqual(life[0]["label"], "HSF history begins")
        self.assertTrue(life[0]["first"])
        self.assertEqual(life[1]["movement"], "RISING")
        self.assertIn("golden_cross", life[1]["signals_added"])
        self.assertEqual(life[2]["transition"], ("WATCH", "STRONG"))
        self.assertIn("breakout", life[2]["signals_added"])

    def test_weakening_and_strong_to_watch(self):
        h = _hist()
        h.append({"time": h[-1]["time"] + dt.timedelta(hours=1), "score": 70, "status": "WATCH",
                  "score_version": "1.0", "signals": ["prebreakout"]})
        life = si.reconstruct_lifecycle(h)
        self.assertEqual(life[-1]["movement"], "FALLING")
        self.assertEqual(life[-1]["transition"], ("STRONG", "WATCH"))
        self.assertIn("breakout", life[-1]["signals_removed"])

    def test_single_observation(self):
        self.assertEqual(len(si.reconstruct_lifecycle(_hist()[:1])), 1)

    def test_out_of_order_and_no_history(self):
        self.assertTrue(si.reconstruct_lifecycle(list(reversed(_hist())))[0]["first"])
        self.assertEqual(si.reconstruct_lifecycle(None), [])
        self.assertEqual(si.reconstruct_lifecycle([]), [])

    def test_version_change_blocks_numeric_movement(self):
        t0 = dt.datetime(2026, 9, 12, tzinfo=dt.timezone.utc)
        h = [{"time": t0, "score": 70, "status": "WATCH", "score_version": "1.0", "signals": ["breakout"]},
             {"time": t0 + dt.timedelta(hours=1), "score": 86, "status": "STRONG", "score_version": "1.1",
              "signals": ["breakout"]}]
        life = si.reconstruct_lifecycle(h)
        self.assertEqual(life[1]["movement"], "VERSION_CHANGED")
        self.assertNotIn("delta", life[1])

    def test_malformed_and_legacy_records_safe(self):
        t0 = dt.datetime(2026, 9, 12, tzinfo=dt.timezone.utc)
        h = [{"time": t0, "score": None, "signals": []},                 # malformed (no score)
             {"time": None, "score": 70, "signals": []},                 # malformed (no time)
             {"time": t0 + dt.timedelta(hours=1), "score": 72, "status": "WATCH", "signals": []}]  # legacy (no version)
        life = si.reconstruct_lifecycle(h)
        self.assertEqual(len(life), 1)  # only the one valid row survives


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Col:
    def button(self, *a, **k):
        return False


class _FakeSt:
    def __init__(self):
        self.session_state = {}

    def __getattr__(self, _name):
        def _noop(*a, **k):
            return None
        return _noop

    def columns(self, n, **k):
        return [_Col() for _ in range(n if isinstance(n, int) else len(n))]

    def expander(self, *a, **k):
        return _Ctx()

    def button(self, *a, **k):
        return False


class ReadOnlyAndNoRebuildTests(unittest.TestCase):
    def test_render_is_read_only_and_no_rebuild_no_claude(self):
        fake = _FakeSt()
        spies = {n: mock.MagicMock() for n in
                 ("freeze_opps", "freeze_one", "freeze_sig", "save", "brief", "claude", "create_alert")}
        spies["freeze_opps"].return_value = 0
        with (
            mock.patch.object(si, "st", fake),
            mock.patch("db.signal_outcomes.fetch_ticker_opportunity_history", return_value=_hist()) as fetch,
            mock.patch("db.signal_outcomes.freeze_opportunities", spies["freeze_opps"]),
            mock.patch("db.signal_outcomes.freeze_opportunity", spies["freeze_one"]),
            mock.patch("db.signal_outcomes.freeze_signal", spies["freeze_sig"]),
            mock.patch("db.opportunity_snapshots.save_opportunity_snapshot", spies["save"]),
            mock.patch("ui.market_brief._brief_cached", spies["brief"]),
            mock.patch("ui.market_brief._calibration_records_cached", return_value=[]),
            mock.patch("ui.ai_confidence_explain._earnings_days_for", return_value=None),
            mock.patch("db.alerts.create_alert", spies["create_alert"], create=True),
            mock.patch("ui.ai.ask_claude", spies["claude"]),
        ):
            si.render_stock_intelligence("NVDA", current_row=_row())
        # Only a small read-only history query; no writes, no brief rebuild, no Claude.
        self.assertEqual(fetch.call_count, 1)
        spies["freeze_opps"].assert_not_called()
        spies["freeze_one"].assert_not_called()
        spies["freeze_sig"].assert_not_called()
        spies["save"].assert_not_called()
        spies["brief"].assert_not_called()
        spies["claude"].assert_not_called()
        spies["create_alert"].assert_not_called()

    def test_render_empty_ticker_and_db_unavailable_are_safe(self):
        fake = _FakeSt()
        with mock.patch.object(si, "st", fake):
            si.render_stock_intelligence("")  # empty -> info, no crash
        with (
            mock.patch.object(si, "st", _FakeSt()),
            mock.patch("db.signal_outcomes.fetch_ticker_opportunity_history", side_effect=RuntimeError("db")),
            mock.patch("ui.market_brief._calibration_records_cached", return_value=[]),
            mock.patch("ui.ai_confidence_explain._earnings_days_for", return_value=None),
        ):
            si.render_stock_intelligence("NVDA", current_row=_row())  # DB down -> still renders


class StateIsolationTests(unittest.TestCase):
    """Section 43 — mandatory: A -> B -> A must not leak B into A."""

    _A = {"ticker": "AMD", "score": 60, "score_version": "1.0", "status": "WATCH",
          "primary_setup": "PreBreakout", "signals": ["prebreakout"], "n_signals": 1,
          "fading": False, "breakout_score": None, "prob": 70, "chg_pct": 1.0, "last": 100.0,
          "score_components": {"signals_component": 12, "model_component": 26,
                               "momentum_component": 0, "fading_penalty": 0}}
    _B = {"ticker": "NVDA", "score": 85, "score_version": "1.0", "status": "STRONG",
          "primary_setup": "Breakout", "signals": ["breakout", "golden_cross"], "n_signals": 2,
          "fading": False, "breakout_score": 80, "prob": None, "chg_pct": 3.0, "last": 120.0,
          "score_components": {"signals_component": 24, "model_component": 30,
                               "momentum_component": 6, "fading_penalty": 0}}

    def test_a_b_a_no_leak(self):
        a1 = si.build_stock_intelligence("AMD", current_opp=self._A)
        b = si.build_stock_intelligence("NVDA", current_opp=self._B)
        a2 = si.build_stock_intelligence("AMD", current_opp=self._A)
        self.assertEqual((a1["hsf_score"], a2["hsf_score"], b["hsf_score"]), (60, 60, 85))
        self.assertEqual(a2["signals"], ["prebreakout"])
        self.assertNotIn("breakout", a2["signals"])
        self.assertNotIn("golden_cross", a2["signals"])
        self.assertIsNone(a2["model"]["breakout_score"])
        self.assertEqual(a2["model"]["prebreakout_prob"], 70)

    def test_mismatched_opp_is_rejected(self):
        # B's opp handed to AMD must be ignored (no leak) -> falls back to none.
        intel = si.build_stock_intelligence("AMD", current_opp=self._B)
        self.assertFalse(intel["has_opportunity"])

    def test_build_does_not_mutate_caller_opp(self):
        snap = dict(self._A)
        si.build_stock_intelligence("AMD", current_opp=self._A, history=_hist_amd())
        self.assertEqual(self._A, snap)  # movement/annotations don't mutate input


def _hist_amd():
    import datetime as _dt
    t0 = _dt.datetime(2026, 9, 12, 9, 30, tzinfo=_dt.timezone.utc)
    return [{"time": t0, "score": 55, "status": "WATCH", "score_version": "1.0", "signals": ["prebreakout"]}]


class RequiredLifecycleTest(unittest.TestCase):
    """Section 44 — the exact required history."""

    def _h(self):
        import datetime as _dt
        base = _dt.datetime(2026, 9, 12, 9, 30, tzinfo=_dt.timezone.utc)

        def step(m, s, st_, sig):
            return {"time": base + dt.timedelta(minutes=m), "score": s,
                    "status": st_, "score_version": "1.0", "signals": sig}
        return [
            step(0, 58, "WATCH", ["prebreakout"]),
            step(30, 62, "WATCH", ["prebreakout"]),
            step(60, 70, "WATCH", ["prebreakout", "golden_cross"]),
            step(90, 78, "STRONG", ["prebreakout", "golden_cross", "breakout"]),
            step(120, 74, "WATCH", ["golden_cross", "breakout"]),
        ]

    def test_full_lifecycle(self):
        life = si.reconstruct_lifecycle(self._h())
        self.assertEqual(life[0]["label"], "HSF history begins")
        self.assertEqual(life[1]["movement"], "RISING")            # +4
        self.assertEqual(life[2]["signals_added"], ["golden_cross"])
        self.assertEqual(life[3]["transition"], ("WATCH", "STRONG"))
        self.assertIn("breakout", life[3]["signals_added"])
        self.assertEqual(life[4]["transition"], ("STRONG", "WATCH"))
        self.assertEqual(life[4]["movement"], "FALLING")           # -4
        self.assertEqual(life[4]["signals_removed"], ["prebreakout"])

    def test_movement_threshold_boundaries(self):
        import datetime as _dt
        t0 = _dt.datetime(2026, 9, 12, tzinfo=_dt.timezone.utc)

        def two(delta):
            return si.reconstruct_lifecycle([
                {"time": t0, "score": 70, "status": "WATCH", "score_version": "1.0", "signals": []},
                {"time": t0 + dt.timedelta(hours=1), "score": 70 + delta, "status": "WATCH",
                 "score_version": "1.0", "signals": []}])[1]["movement"]
        self.assertEqual(two(2), "UNCHANGED")
        self.assertEqual(two(3), "RISING")
        self.assertEqual(two(4), "RISING")
        self.assertEqual(two(0), "UNCHANGED")
        self.assertEqual(two(-2), "UNCHANGED")
        self.assertEqual(two(-3), "FALLING")


class HSFConsistencyTests(unittest.TestCase):
    """Section 4/5 — Stock Intel reuses the CANONICAL score, not a copy."""

    def test_score_equals_canonical(self):
        from ui import opportunities as op
        row = _row()
        intel = si.build_stock_intelligence("NVDA", current_row=row)
        # Same inputs through the canonical scorer -> same score.
        from ui.results_intelligence import _row_to_signal_fields
        f = _row_to_signal_fields(row)
        pos = [s for s in op._POSITIVE_SIGNALS if s in f["signals"]]
        canonical = op.build_opportunity_score(
            n_signals=len(pos), breakout_score=f["breakout_score"], prob=f["prob"],
            chg_pct=f["chg_pct"], fading=f["fading"])
        self.assertEqual(intel["hsf_score"], canonical)

    def test_components_sum_to_score(self):
        intel = si.build_stock_intelligence("NVDA", current_row=_row())
        c = intel["score_components"]
        raw = (c["signals_component"] + c["model_component"]
               + c["momentum_component"] - c["fading_penalty"])
        self.assertEqual(intel["hsf_score"], int(round(max(0, min(100, raw)))))


if __name__ == "__main__":
    unittest.main()
