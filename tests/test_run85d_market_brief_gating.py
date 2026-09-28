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
HAS_ST = importlib.util.find_spec("streamlit") is not None

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


def render(tier):
    """Render Market Brief for `tier`; returns (AppTest, claude_calls)."""
    from streamlit.testing.v1 import AppTest

    calls = []

    def fake_claude(**kw):
        calls.append(kw.get("feature"))
        return "AI TEXT", None

    ctx = {"sufficient": True, "bucket": "70-79", "positive_rate": 0.3, "n": 10, "confidence": "low"}
    outcomes = {"completed": 5, "pending": 1, "hit_rate": 0.4, "avg_winner": 3.0, "avg_loser": -2.0}
    at = AppTest.from_string(SCRIPT, default_timeout=120)
    for k, v in {"username": "realtest123@example.com", "tier_key": tier, "is_admin": False,
                 "tier": SimpleNamespace(key=tier, name=tier.upper()), "entitlements": dict(_flags(tier))}.items():
        at.session_state[k] = v
    with mock.patch("ui.market_brief._brief_cached", mock.Mock(return_value=DATA)), \
         mock.patch("analytics.hsf_calibration.historical_context", return_value=ctx), \
         mock.patch("db.signal_outcomes.summarize_recent_outcomes", return_value=outcomes), \
         mock.patch("db.signal_outcomes.summarize_outcomes_by_type", return_value=[]), \
         mock.patch("ui.ai.is_configured", return_value=True), \
         mock.patch("ui.ai.ask_claude", side_effect=fake_claude), \
         mock.patch("market_data.get_latest_quotes", return_value={}), \
         mock.patch("market_data.build_day_trader_metrics", return_value=[]):
        at.run()
    return at, calls


def text(at):
    parts = []
    for coll in (at.markdown, at.caption, at.info, at.subheader, at.metric, at.expander):
        for e in coll:
            parts.append(str(getattr(e, "value", "") or getattr(e, "label", "")))
    return "\n".join(parts)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class MarketBriefGatingTests(unittest.TestCase):
    def test_free_sees_no_premium_or_pro_research(self):
        at, calls = render("basic")
        self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
        t = text(at)
        self.assertNotIn("### 🧠 PreBreakout picks", t)
        self.assertNotRegex(t, r"TWLO · \$275")                      # candidate list not shown
        self.assertIn("PreBreakout candidates", t)                   # one-line Premium note instead
        self.assertNotIn("AI take", t)
        self.assertNotIn("📊 Historical context", t)
        self.assertNotIn("flagged-signal outcomes", t)
        self.assertEqual(calls, [])                                   # no Claude spend on Free
        self.assertNotRegex(t, r"prebreakout \+|\+ prebreakout")       # Standouts tags hidden

    def test_pro_gets_historical_research_but_not_premium(self):
        at, calls = render("pro")
        self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
        t = text(at)
        self.assertIn("📊 Historical context", t)
        self.assertIn("flagged-signal outcomes", t)
        self.assertNotIn("### 🧠 PreBreakout picks", t)
        self.assertNotIn("AI take", t)
        self.assertEqual(calls, [])

    def test_premium_gets_everything(self):
        at, calls = render("premium")
        self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
        t = text(at)
        self.assertIn("### 🧠 PreBreakout picks", t)
        self.assertIn("AI take", t)
        self.assertIn("📊 Historical context", t)
        self.assertIn("opportunity_ai_take", calls)
        self.assertIn("market_brief_narrative", calls)

    def test_scoreboard_is_gone_for_everyone(self):
        for tier in ("basic", "pro", "premium"):
            t = text(render(tier)[0])
            self.assertNotIn("Yesterday's brief", t, tier)
            self.assertNotIn("how it did", t, tier)

    def test_hsf_scores_are_identical_for_every_plan(self):
        scores = {}
        for tier in ("basic", "pro", "premium"):
            scores[tier] = re.findall(r"(\w+) — HSF Score (\d+)", text(render(tier)[0]))
        self.assertTrue(scores["basic"])
        self.assertEqual(scores["basic"], scores["pro"])
        self.assertEqual(scores["basic"], scores["premium"])

    def test_predictive_edge_note_appears_once(self):
        t = text(render("premium")[0])
        self.assertLessEqual(t.count("not a predictive edge"), 1)


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
