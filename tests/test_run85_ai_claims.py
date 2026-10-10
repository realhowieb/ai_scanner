"""Run 85 — Premium AI claim accuracy.

R84-1: template-built notes were labelled "AI Notes" and sold as "AI setup notes".
R84-2: the AI Scan Summary ranked by BreakoutScore (HSF Score absent) and its
output gave trade instructions ("before entry", "position size") and inferred
"institutional accumulation", despite its prompt.

These tests check semantic invariants, not big string snapshots.
"""
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import ai_guardrails as g

ROOT = Path(__file__).resolve().parents[1]

# Lines Claude actually produced on the live app during Run 84.
LIVE_OUTPUT = """1. MXL (Score: 52.3)
Highest breakout score with clean 2.1% gap and strong 47.7% uptrend over 20 days
Solid 1.36x relative volume and $9.5M daily liquidity support continuation
2. AKAM (Score: 49.7)
Exceptional 12.8% gap signals institutional accumulation
1.49x relative volume and $19.4M daily liquidity adequate for position size
Risk Caveat: confirm with intraday support holds before entry."""


class GuardrailRuleTests(unittest.TestCase):
    def test_rules_state_the_responsibility_boundary_and_prohibitions(self):
        r = g.RULES.lower()
        for must in ("hsf's own scoring system calculated the hsf score", "never say you calculated, ranked",
                     "buy, sell, hold", "enter, exit", "size a position", "price targets",
                     "do not predict", "not personalized investment advice", "do not invent data"):
            self.assertIn(must, r)

    def test_analysis_features_get_rules_json_and_admin_features_do_not(self):
        for feature in ("scan_summary", "ticker_deepdive", "results_chat", "scan_diff", "watchlist_digest",
                        "alert_line", "ai_confidence_explain", "market_brief_narrative", "opportunity_ai_take", None):
            self.assertTrue(g.applies(feature), feature)
        for feature in ("nl_screener", "nl_scan_setup", "admin_triage"):
            self.assertFalse(g.applies(feature), feature)

    def test_every_feature_in_the_code_is_classified(self):
        used = set()
        for path in (ROOT / "ui").glob("*.py"):
            used |= set(re.findall(r'feature="([a-z_]+)"', path.read_text()))
        exempt = used & g.EXEMPT_FEATURES
        self.assertEqual(exempt, {"nl_screener", "nl_scan_setup", "admin_triage"})   # nothing else opts out


class ScrubTests(unittest.TestCase):
    def test_live_trade_instructions_are_removed_and_facts_kept(self):
        out, removed = g.scrub(LIVE_OUTPUT)
        self.assertEqual(removed, 3)
        low = out.lower()
        for gone in ("institutional accumulation", "position size", "before entry"):
            self.assertNotIn(gone, low)
        # Whole lines are removed (never rewritten), so a fact on the same line as an
        # instruction goes with it; every other line is kept verbatim.
        for kept in ("mxl", "2.1% gap", "47.7% uptrend", "akam", "support continuation"):
            self.assertIn(kept, low)
        self.assertIn(g.REMOVED_NOTE, out)

    def test_ordinary_market_wording_is_not_flagged(self):
        clean = ["The sell-off in semis eased", "Buyers outnumbered sellers", "Support holds near the 20-day average",
                 "Relative volume is 2.1x", "Short interest data is not available", "Momentum is fading"]
        self.assertEqual(g.find_action_language("\n".join(clean)), [])

    def test_instruction_forms_are_flagged(self):
        for bad in ("Buy on a pullback", "Consider selling half: sell into strength", "Enter above $50",
                    "Exit if it closes below the gap", "Set a stop-loss at $48", "Price target $70",
                    "This will break out this week", "A guaranteed winner", "Size your position carefully",
                    "Short it on the bounce", "Go short below $40"):
            self.assertTrue(g.find_action_language(bad), bad)


class _FakeAnthropic(types.ModuleType):
    """Minimal anthropic stand-in: records calls; can raise provider errors."""

    class AuthenticationError(Exception): ...
    class RateLimitError(Exception): ...
    class APITimeoutError(Exception): ...
    class APIError(Exception): ...

    def __init__(self, reply="", raise_=None):
        super().__init__("anthropic")
        self.calls, self._reply, self._raise = [], reply, raise_
        fake = self

        class Messages:
            def create(self, **kw):
                fake.calls.append(kw)
                if fake._raise:
                    raise fake._raise
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=fake._reply)])

        class Anthropic:
            def __init__(self, **kw):
                self.messages = Messages()

        self.Anthropic = Anthropic


def _ask(fake, feature, **kw):
    import config
    import ui.ai as ai

    with mock.patch.dict(sys.modules, {"anthropic": fake}), \
         mock.patch.object(config, "AI_ENABLED", True), \
         mock.patch.object(config, "ANTHROPIC_API_KEY", "test-key-not-real"):
        return ai.ask_claude(system="BASE PROMPT", user="data", feature=feature, **kw)


class AskClaudeIntegrationTests(unittest.TestCase):
    def test_summary_call_carries_rules_and_output_is_checked(self):
        fake = _FakeAnthropic(reply=LIVE_OUTPUT)
        text, err = _ask(fake, "scan_summary")
        self.assertIsNone(err)
        self.assertTrue(fake.calls[0]["system"].startswith("BASE PROMPT"))
        self.assertIn(g.RULES, fake.calls[0]["system"])
        self.assertNotIn("before entry", text)

    def test_json_screener_call_is_untouched(self):
        payload = '{"universe":"NASDAQ","explanation":"buyers on volume"}'
        fake = _FakeAnthropic(reply=payload)
        text, _ = _ask(fake, "nl_screener")
        self.assertEqual(fake.calls[0]["system"], "BASE PROMPT")
        self.assertEqual(text, payload)

    def test_provider_failures_return_errors_not_substitute_text(self):
        cases = [(_FakeAnthropic(raise_=_FakeAnthropic.AuthenticationError("bad key")), "invalid"),
                 (_FakeAnthropic(raise_=_FakeAnthropic.RateLimitError("quota")), "rate-limited"),
                 (_FakeAnthropic(raise_=_FakeAnthropic.APITimeoutError("slow")), "timed out"),
                 (_FakeAnthropic(reply=""), "empty")]
        for fake, expect in cases:
            text, err = _ask(fake, "scan_summary")
            self.assertIsNone(text, expect)
            self.assertIn(expect, (err or "").lower(), expect)

    def test_no_credentials_means_no_call(self):
        import config
        import ui.ai as ai

        fake = _FakeAnthropic(reply="x")
        with mock.patch.dict(sys.modules, {"anthropic": fake}), \
             mock.patch.object(config, "AI_ENABLED", True), mock.patch.object(config, "ANTHROPIC_API_KEY", None):
            text, err = ai.ask_claude(system="s", user="u", feature="scan_summary")
        self.assertIsNone(text)
        self.assertEqual(fake.calls, [])
        self.assertIn("not configured", err.lower())


class SummaryInputTests(unittest.TestCase):
    def test_summary_sees_hsf_score_in_hsf_order(self):
        from ui import ai_summary

        df = pd.DataFrame([
            {"Ticker": "BIGBREAKOUT", "BreakoutScore": 60, "IsBreakout": True, "PctChange": 0.5, "Last": 5.0},
            {"Ticker": "STRONGHSF", "BreakoutScore": 45, "IsBreakout": True, "PctChange": 6.0, "Last": 20.0,
             "VolRel20": 4.0, "GapPct": 5.0, "Trend20D%": 30.0},
        ])
        table = ai_summary._build_table_text(df)
        header, *rows = table.strip().splitlines()
        self.assertIn("HSF Score", header.split(","))
        from ui.headline_score import add_hsf_score_column, rank_hsf_opportunities
        expected_first = rank_hsf_opportunities(add_hsf_score_column(df))["Ticker"].iloc[0]
        self.assertTrue(rows[0].startswith(expected_first), (rows, expected_first))

    def test_prompts_explain_in_hsf_order_and_do_not_select(self):
        from ui import ai_summary

        for prompt in (ai_summary._SYSTEM_PROMPT, ai_summary._TICKER_SYSTEM_PROMPT):
            low = prompt.lower()
            self.assertIn("hsf score", low)
            self.assertIn("use only the scan data provided", low)
            self.assertIn("not investment advice", low)
            for phrase in ("buy, sell, hold", "entry, exit", "position-size", "price-target", "institutional activity"):
                self.assertIn(phrase, low)
            self.assertNotIn("identify the 3-5 strongest", low)       # the model no longer picks winners
        self.assertIn("already ordered by hsf score", ai_summary._SYSTEM_PROMPT.lower())

    def test_chat_context_includes_hsf_score_and_boundaries(self):
        from ui import ai_chat

        self.assertIn("HSF Score", ai_chat._CHAT_COLS)
        low = ai_chat._SYSTEM.lower()
        for phrase in ("including hsf score", "controls ordering", "do not rank",
                       "buy, sell, hold", "entry, exit", "position-size", "price-target",
                       "institutional activity", "not investment advice"):
            self.assertIn(phrase, low)


class LabellingTests(unittest.TestCase):
    def test_template_notes_are_not_labelled_ai(self):
        results = (ROOT / "ui" / "results.py").read_text()
        self.assertNotIn('st.subheader("AI Notes")', results)
        self.assertEqual(results.count('st.subheader("📝 Setup notes")'), 4)
        self.assertEqual(results.count("not AI-generated"), 4)
        self.assertNotIn("AI notes", results)

    def test_template_note_text_never_claims_ai(self):
        from ui.ai_notes import generate_ai_note

        rows = [pd.Series({"Ticker": "MXL", "BreakoutScore": 52.3, "GapPct": 2.1, "Trend20D%": 47.7,
                           "VolRel20": 1.36, "DollarVol20": 9.5e6, "Volatility20D%": 5.8}),
                pd.Series({"Ticker": "X", "Last": 10.0, "Change": 0.1, "% Change": 1.0}),
                pd.Series({"Ticker": "EMPTY"}), None]
        for row in rows:
            note = generate_ai_note(row)
            self.assertNotRegex(note, r"\bAI\b", note)

    def test_real_ai_surfaces_say_so_and_who_ranks(self):
        summary = (ROOT / "ui" / "ai_summary.py").read_text()
        self.assertIn("Claude (AI) explains the top results in HSF Score order", summary)
        self.assertIn("HSF's scoring system ", summary)
        self.assertIn("does the ranking; this summary describes it", summary)
        chat = (ROOT / "ui" / "ai_chat.py").read_text()
        self.assertIn("not investment advice", chat)
        self.assertNotIn("least risky", chat)
        panel = (ROOT / "web" / "src" / "features" / "AIScanPanel.tsx").read_text()
        self.assertIn("Claude uses only this scan", panel)
        self.assertIn("HSF Score controls ranking", panel)
        self.assertIn("does not recommend trades", panel)

    def test_summary_failure_shows_the_error_not_template_text(self):
        summary = (ROOT / "ui" / "ai_summary.py").read_text()
        self.assertNotIn("generate_ai_note", summary)
        self.assertIn('st.warning(err or "Could not generate summary.")', summary)

    def test_no_copy_claims_ai_calculates_or_ranks_hsf(self):
        pat = re.compile(r"\b(AI|Claude)\b[^\"'\n]{0,30}\b(ranks|ranked|picks|chooses|selects|calculates|"
                         r"computes|scores|decides|recommends)\b", re.IGNORECASE)
        hits = []
        for path in [*(ROOT / "ui").glob("*.py"), *(ROOT / "pages").glob("*.py"), ROOT / "app.py"]:
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pat.search(line) and not line.lstrip().startswith("#"):
                    hits.append(f"{path.name}:{i}: {line.strip()[:100]}")
        self.assertEqual(hits, [])

    def test_screener_examples_are_expressible_filters(self):
        # The natural-language screener can only set universe/price/volume/gap/session
        # filters; example chips must not promise sectors or trade types it can't find.
        src = (ROOT / "ui" / "ai_screener.py").read_text()
        for promise in ("AI Stocks", "Swing Trades", "swing trade setups"):
            self.assertNotIn(promise, src)

    def test_three_step_scanner_is_not_called_an_ai_scanner(self):
        self.assertNotIn("3-Step AI Scanner", (ROOT / "ui" / "three_step_scanner.py").read_text())


class PackagingAndTierTests(unittest.TestCase):
    def test_premium_ai_claim_names_only_real_ai_features(self):
        from ui import pricing

        label = next(lbl for lbl, flag in pricing.ROWS if flag == "can_ai_notes")
        self.assertIn("AI scan summaries and results chat", label)
        self.assertIn("setup notes", label)
        self.assertNotIn("AI setup notes", label)
        self.assertNotIn("AI setup notes", pricing.benefits_markdown() + pricing.plans_html())
        billing = (ROOT / "pages" / "billing.py").read_text()
        self.assertNotIn("AI research", billing)
        self.assertIn("AI scan summaries and chat", billing)

    def test_premium_still_has_its_distinct_value(self):
        from ui import pricing

        adds = pricing.plan_highlights()["premium"]
        self.assertTrue(any("AI scan summaries" in a for a in adds))
        self.assertTrue(any("Early Breakout" in a for a in adds))
        self.assertEqual(pricing.PRICES, {"basic": "Free", "pro": "$25/mo", "premium": "$40/mo"})

    def test_ai_gate_is_premium_and_admin_only(self):
        from types import SimpleNamespace

        from ui.app_session import FEATURE_MIN_TIER, compute_entitlements

        self.assertEqual(FEATURE_MIN_TIER["can_ai_notes"], "premium")
        for tier, want in (("basic", False), ("pro", False), ("premium", True)):
            flags = compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=False)
            self.assertEqual(bool(flags.get("can_ai_notes")), want, tier)
        admin = compute_entitlements(tier_obj=SimpleNamespace(key="basic", name="BASIC"), is_admin=True)
        self.assertTrue(admin.get("can_ai_notes"))


if __name__ == "__main__":
    unittest.main()
