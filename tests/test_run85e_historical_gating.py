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
HAS_ST = True
CTX = {"sufficient": True, "bucket": "70-79", "positive_rate": 0.3, "n": 10, "confidence": "low"}


def _flags(tier):
    from ui.app_session import compute_entitlements

    return compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=False)


def _all_text(at):
    parts = []
    for coll in (at.markdown, at.caption, at.info, at.expander):
        parts += [str(getattr(e, "value", "") or getattr(e, "label", "")) for e in coll]
    return "\n".join(parts)


SCRIPT = '''
from utils.st_compat import st
from ui.stock_intelligence import _render_historical
_render_historical({
    "ticker": "MXL",
    "historical_context": {"sufficient": True, "bucket": "70-79", "positive_rate": 0.3, "n": 10, "confidence": "low"},
    "history_summary": {"observations": 4, "matured": 3, "positive": 1},
    "outcome_cohort": {"available": True, "follow_through_rate": 0.5, "status": "WATCH", "score_band": "70-79",
                       "horizon": "H24", "comparable": 30, "evidence_strength": "EARLY"},
})
'''


class SourceTests(unittest.TestCase):
    def test_every_historical_context_display_is_gated(self):
        for rel in ("ui/results_intelligence.py", "ui/stock_intelligence.py", "ui/market_brief.py"):
            src = (ROOT / rel).read_text()
            self.assertIn("can_track_record", src, rel)


if __name__ == "__main__":
    unittest.main()
