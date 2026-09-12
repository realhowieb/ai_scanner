"""Run 20B — runtime/regression guards for Scanner intelligence.

Renders render_scanner_intelligence headlessly (a fake Streamlit) and proves the
read-only, cached, no-brief-rebuild, no-Claude guarantees from Runs 19/20A.
"""
import unittest
from unittest import mock

import pandas as pd

import ui.results_intelligence as ri


class _FakeCol:
    def button(self, *a, **k):
        return False

    def markdown(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass


class _ColConfig:
    def __getattr__(self, _name):
        return lambda *a, **k: None


class FakeStreamlit:
    """Minimal headless Streamlit: widgets return deterministic values, and
    session_state persists across renders so caching can be exercised."""

    def __init__(self):
        self.session_state = {}
        self.column_config = _ColConfig()

    # passive outputs
    def markdown(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass

    def dataframe(self, *a, **k):
        pass

    def switch_page(self, *a, **k):
        pass

    # widgets -> deterministic
    def radio(self, _label, options, **k):
        return self.session_state.get(k.get("key"), options[0])

    def selectbox(self, _label, options, **k):
        return self.session_state.get(k.get("key"), options[0] if options else None)

    def text_input(self, *a, **k):
        return self.session_state.get(k.get("key"), "")

    def button(self, *a, **k):
        return False

    def columns(self, n, **k):
        return [_FakeCol() for _ in range(n if isinstance(n, int) else len(n))]


def _rows():
    return [
        {"Ticker": "NVDA", "IsBreakout": True, "EMACross": "Golden",
         "PreBreakoutProb%": 78, "BreakoutScore": 74, "PctChange": 2.4, "GapPct": 1.0, "Last": 120.0},
        {"Ticker": "AMD", "IsBreakout": True, "EMACross": None,
         "PreBreakoutProb%": 55, "BreakoutScore": 60, "PctChange": 1.0, "GapPct": 0.2, "Last": 95.0},
        {"Ticker": "ZZZ", "IsBreakout": False, "PctChange": 0.1},  # non-qualifying
    ]


class ScannerRuntimeGuards(unittest.TestCase):
    def _render(self, df, fake, *, spies):
        with (
            mock.patch.object(ri, "st", fake),
            mock.patch("db.opportunity_snapshots.load_previous_opportunity_snapshot", spies["load"]),
            mock.patch("db.opportunity_snapshots.save_opportunity_snapshot", spies["save"]),
            mock.patch("db.signal_outcomes.freeze_opportunities", spies["freeze_opps"]),
            mock.patch("db.signal_outcomes.freeze_opportunity", spies["freeze_one"]),
            mock.patch("db.signal_outcomes.freeze_signal", spies["freeze_sig"]),
            mock.patch("ui.market_brief._brief_cached", spies["brief"]),
            mock.patch("ui.market_brief._calibration_records_cached", return_value=[]),
            mock.patch("ui.ai.ask_claude", spies["claude"]),
        ):
            ri.render_scanner_intelligence(df, key_prefix="results")

    def _spies(self, *, prev=None):
        return {
            "load": mock.MagicMock(return_value=prev),
            "save": mock.MagicMock(return_value=True),
            "freeze_opps": mock.MagicMock(return_value=0),
            "freeze_one": mock.MagicMock(return_value=True),
            "freeze_sig": mock.MagicMock(return_value=True),
            "brief": mock.MagicMock(return_value={}),
            "claude": mock.MagicMock(return_value=("x", None)),
        }

    def test_render_performs_no_db_writes(self):
        fake, spies = FakeStreamlit(), self._spies()
        self._render(pd.DataFrame(_rows()), fake, spies=spies)
        spies["save"].assert_not_called()
        spies["freeze_opps"].assert_not_called()
        spies["freeze_one"].assert_not_called()
        spies["freeze_sig"].assert_not_called()

    def test_render_does_not_rebuild_market_brief_or_call_claude(self):
        fake, spies = FakeStreamlit(), self._spies()
        self._render(pd.DataFrame(_rows()), fake, spies=spies)
        spies["brief"].assert_not_called()   # regime read from session, not rebuilt
        spies["claude"].assert_not_called()  # no automatic LLM per row

    def test_one_db_read_then_cached_and_no_reconsolidate(self):
        fake, spies = FakeStreamlit(), self._spies()
        df = pd.DataFrame(_rows())
        with mock.patch.object(ri, "consolidate_scanner_results",
                               wraps=ri.consolidate_scanner_results) as cons:
            self._render(df, fake, spies=spies)       # first render: miss
            self._render(df, fake, spies=spies)       # second render: same df -> hit
        self.assertEqual(spies["load"].call_count, 1)  # DB read only on the miss
        self.assertEqual(cons.call_count, 1)           # consolidated once

    def test_view_switch_does_not_recompute(self):
        fake, spies = FakeStreamlit(), self._spies()
        df = pd.DataFrame(_rows())
        with mock.patch.object(ri, "consolidate_scanner_results",
                               wraps=ri.consolidate_scanner_results) as cons:
            self._render(df, fake, spies=spies)                 # Top ranked
            fake.session_state["results_intel_view"] = "Developing"
            self._render(df, fake, spies=spies)                 # switch view
            fake.session_state["results_intel_view"] = "Fading"
            self._render(df, fake, spies=spies)
        self.assertEqual(cons.call_count, 1)          # never reconsolidated
        self.assertEqual(spies["load"].call_count, 1)  # no extra DB reads
        spies["save"].assert_not_called()

    def test_changed_result_set_invalidates_and_reads_again(self):
        fake, spies = FakeStreamlit(), self._spies()
        with mock.patch.object(ri, "consolidate_scanner_results",
                               wraps=ri.consolidate_scanner_results) as cons:
            self._render(pd.DataFrame(_rows()), fake, spies=spies)
            changed = _rows()
            changed[0]["BreakoutScore"] = 20  # intelligence field changed
            self._render(pd.DataFrame(changed), fake, spies=spies)
        self.assertEqual(cons.call_count, 2)
        self.assertEqual(spies["load"].call_count, 2)

    def test_db_unavailable_still_renders(self):
        fake = FakeStreamlit()
        spies = self._spies()
        spies["load"].side_effect = RuntimeError("db down")
        # Should not raise; renders current-only.
        self._render(pd.DataFrame(_rows()), fake, spies=spies)

    def test_failure_conditions_render_safely(self):
        for df in (
            pd.DataFrame(),                                   # empty
            pd.DataFrame([_rows()[0]]),                       # single row
            pd.DataFrame(_rows() + _rows()),                  # duplicate tickers
            pd.DataFrame([{"Ticker": "X"}]),                  # missing model/signals
            pd.DataFrame([{"Foo": 1}]),                       # malformed (no ticker col)
        ):
            self._render(df, FakeStreamlit(), spies=self._spies())

    def test_regime_from_session_used_not_rebuilt(self):
        fake, spies = FakeStreamlit(), self._spies()
        fake.session_state["_last_market_regime"] = "RISK-ON"
        self._render(pd.DataFrame(_rows()), fake, spies=spies)
        spies["brief"].assert_not_called()

    def test_previous_snapshot_enables_movement_counts(self):
        prev = {"opportunities": [
            {"ticker": "NVDA", "score": 70, "status": "WATCH", "score_version": "1.0"}]}
        fake, spies = FakeStreamlit(), self._spies(prev=prev)
        # renders with movement; still no writes
        self._render(pd.DataFrame(_rows()), fake, spies=spies)
        spies["save"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
