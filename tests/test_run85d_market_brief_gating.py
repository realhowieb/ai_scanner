"""Run 85D — Market Brief follows the same entitlements as the Scanner.

Live review: Market Brief showed Free accounts Premium's PreBreakout candidate
list, Claude-written AI text, and Pro's historical research, plus a small-sample
"how yesterday's brief did" performance scoreboard. Now:
  PreBreakout candidates → can_early_breakout (Premium)
  AI narrative / AI take → can_ai_notes (Premium)
  historical outcomes     → can_track_record (Pro)
  scoreboard              → removed for everyone
HSF Scores are computed identically for every plan; only display is gated.
"""
import datetime as dt
import importlib.util
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True

DATA = {
    "gappers": [{"ticker": "AKAM", "gap_pct": 12.8, "chg_pct": 3.1}, {"ticker": "GRAL", "gap_pct": -3.0, "chg_pct": 1.9}],
    "golden": ["VIAV", "AAON"],
    "top_setups": [("MXL", 52.3), ("GRAL", 51.6), ("AKAM", 49.7)],
    "picks": [{"symbol": "TWLO", "prob": 20.2, "last": 275.7}, {"symbol": "RBRK", "prob": 19.1, "last": 109.9},
              {"symbol": "MXL", "prob": 18.0, "last": 30.0}],
    "earnings_today": [],
    "market_close": [("SPY", 771.21, 0.51), ("QQQ", 745.04, 0.53)],
    "gainers": [("MXL", 9.7), ("VIAV", 7.9)],
    "losers": [("TWLO", -8.0), ("RBRK", -3.5)],
    "breadth": (76, 22),
    "sectors": [("Industrials", 0.9), ("Energy", -0.8)],
    "snapshot_time": dt.datetime(2026, 9, 27, 5, 45),
}

SCRIPT = '''
import runpy, streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
st.switch_page = lambda *a, **k: st.stop()
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "pages" / "brief.py")


def _flags(tier):
    from ui.app_session import compute_entitlements

    return compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=False)


def text(at):
    parts = []
    for coll in (at.markdown, at.caption, at.info, at.subheader, at.metric, at.expander):
        for e in coll:
            parts.append(str(getattr(e, "value", "") or getattr(e, "label", "")))
    return "\n".join(parts)


class EmailAndSourceTests(unittest.TestCase):
    def test_emailed_brief_includes_picks_only_for_premium(self):
        import ui.market_brief as mb

        for tier, expect in (("pro", []), ("premium", DATA["picks"])):
            fake_st = mock.MagicMock()
            fake_st.session_state = {"entitlements": dict(_flags(tier))}
            fake_st.button.return_value = True
            with mock.patch.object(mb, "st", fake_st), \
                 mock.patch.object(mb, "_watchlist_rows", return_value=[]), \
                 mock.patch("scheduler.morning_digest._compose", return_value=("h", "t")) as compose, \
                 mock.patch("ui.email_utils.send_digest_email", return_value=True):
                mb._render_email_button("u@example.com", DATA)
            self.assertEqual(compose.call_args.args[4], expect, tier)

    def test_scheduled_morning_email_gates_picks_to_premium(self):
        src = (ROOT / "scheduler" / "morning_digest.py").read_text()
        self.assertIn('user_picks = picks if has_min_tier(tier_key, "premium") else []', src)
        self.assertIn("email, watch_rows, gappers, earnings_hits, user_picks, notes=notes,", src)

    def test_scoreboard_code_removed(self):
        src = (ROOT / "ui" / "market_brief.py").read_text()
        for gone in ("def _render_yesterday", "def _yesterday_performance", '"yesterday"'):
            self.assertNotIn(gone, src)


if __name__ == "__main__":
    unittest.main()
