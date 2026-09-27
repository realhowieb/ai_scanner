"""Run 85E — historical research needs Pro on the Scanner and Stock Intelligence too.

Run 85D gated historical context on Market Brief (can_track_record). The same
"70–79 range: X% positive outcome · n" lines also showed to Free accounts in the
Scanner's opportunity cards and Stock Intelligence's Historical research panel.
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None
CTX = {"sufficient": True, "bucket": "70-79", "positive_rate": 0.3, "n": 10, "confidence": "low"}


def _flags(tier):
    from ui.app_session import compute_entitlements

    return compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=False)


def _all_text(at):
    parts = []
    for coll in (at.markdown, at.caption, at.info, at.expander):
        parts += [str(getattr(e, "value", "") or getattr(e, "label", "")) for e in coll]
    return "\n".join(parts)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ScannerHistoricalTests(unittest.TestCase):
    def test_scanner_cards_show_history_only_with_pro(self):
        import test_run83b_scanner_state as scanner

        for tier, visible in (("basic", False), ("pro", True), ("premium", True)):
            with mock.patch("analytics.hsf_calibration.historical_context", return_value=CTX), \
                 mock.patch("ui.market_brief._calibration_records_cached", return_value=[{"x": 1}]):
                at = scanner.run_app(tier=tier, session={"entitlements": dict(_flags(tier))})
            self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
            text = _all_text(at)
            self.assertIn("### HSF Opportunities", text, tier)
            self.assertEqual("📊 Historical context" in text, visible, tier)


SCRIPT = '''
import streamlit as st
from ui.stock_intelligence import _render_historical
_render_historical({
    "ticker": "MXL",
    "historical_context": {"sufficient": True, "bucket": "70-79", "positive_rate": 0.3, "n": 10, "confidence": "low"},
    "history_summary": {"observations": 4, "matured": 3, "positive": 1},
    "outcome_cohort": {"available": True, "follow_through_rate": 0.5, "status": "WATCH", "score_band": "70-79",
                       "horizon": "H24", "comparable": 30, "evidence_strength": "EARLY"},
})
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class StockIntelligenceHistoricalTests(unittest.TestCase):
    def render(self, tier):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["entitlements"] = dict(_flags(tier))
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
        return _all_text(at)

    def test_free_sees_only_the_pro_note(self):
        text = self.render("basic")
        for hidden in ("positive-outcome rate", "4 prior HSF observation(s)", "persisted or strengthened"):
            self.assertNotIn(hidden, text)
        self.assertIn("📚 Historical research · Pro adds historical research", text)

    def test_pro_and_premium_see_the_research(self):
        for tier in ("pro", "premium"):
            text = self.render(tier)
            self.assertIn("positive-outcome rate 30%", text, tier)
            self.assertIn("4 prior HSF observation(s)", text, tier)
            self.assertIn("persisted or strengthened", text, tier)
            self.assertNotIn("Pro adds historical research", text, tier)


class SourceTests(unittest.TestCase):
    def test_every_historical_context_display_is_gated(self):
        for rel in ("ui/results_intelligence.py", "ui/stock_intelligence.py", "ui/market_brief.py"):
            src = (ROOT / rel).read_text()
            self.assertIn("can_track_record", src, rel)


if __name__ == "__main__":
    unittest.main()
