"""Run 85C — one shared account / plan card on every authenticated page.

The card (ui.account_card) owns name, plan, plan summary, upgrade CTA,
"Compare all plans" and "Log out". Scanner, every sub-page sidebar and the
phone menu render it; changing pages must never change what it shows.
"""
import importlib.util
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from ui.account_card import account_name, plan_card

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None

CTA = {"basic": "Upgrade to Pro", "pro": "Upgrade to Premium", "premium": None, "admin": None}
LABEL = {"basic": "Free", "pro": "Pro", "premium": "Premium", "admin": "Admin"}

# Phrases in the card copy → the entitlement that makes each one true.
CLAIMS = {
    "email delivery": "can_email_alerts", "interactive results": "can_export_csv", "csv export": "can_export_csv",
    "exports": "can_export_csv", "nasdaq and combined scans": "can_scan_nasdaq", "earnings": "can_earnings",
    "history": "can_scan_history", "ai scan summaries and results chat": "can_ai_notes",
    "setup notes": "can_ai_notes", "early breakout research": "can_early_breakout",
    "full-market custom scans": "can_full_universe", "paper trading": "can_paper_trade",
    "the live day trader": "can_day_trader",
}


def _flags(tier):
    from ui.app_session import compute_entitlements

    return compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=tier == "admin")


class PlanCardTests(unittest.TestCase):
    def test_label_headline_and_cta_per_tier(self):
        for tier in ("basic", "pro", "premium"):
            c = plan_card(tier)
            self.assertEqual(c["label"], LABEL[tier])
            self.assertEqual(c["headline"], f"You're on {LABEL[tier]}")
            self.assertEqual(c["cta_label"], CTA[tier], tier)
            self.assertEqual(c["compare"], "Compare all plans")
        admin = plan_card("basic", is_admin=True)
        self.assertEqual((admin["label"], admin["cta_label"], admin["compare"]), ("Admin", None, None))

    def test_free_copy_is_exactly_the_approved_copy(self):
        c = plan_card("basic")
        self.assertEqual(c["summary"], "Discover today's market opportunities with HSF Score and basic Stock Intelligence.")
        self.assertEqual(c["cta_note"], "Pro adds monitoring and investigation: 5 alerts, email delivery, "
                                        "interactive results, the live Day Trader, exports and history.")

    def test_every_claim_is_backed_by_the_entitlements_of_that_tier(self):
        from ui.app_session import ALERT_LIMIT_BY_TIER

        checks = [("pro", plan_card("pro")["summary"]), ("premium", plan_card("premium")["summary"]),
                  ("pro", plan_card("basic")["cta_note"]), ("premium", plan_card("pro")["cta_note"])]
        for tier, text in checks:
            low = text.lower()
            flags = _flags(tier)
            for phrase, flag in CLAIMS.items():
                if phrase in low:
                    self.assertTrue(flags.get(flag), (tier, phrase))
            m = re.search(r"(\d+) alerts", text)
            self.assertIsNotNone(m, text)
            self.assertEqual(int(m.group(1)), ALERT_LIMIT_BY_TIER[tier], (tier, text))

    def test_no_stale_or_inaccurate_wording(self):
        for tier in ("basic", "pro", "premium"):
            blob = " ".join(str(v) for v in plan_card(tier).values() if v)
            self.assertNotRegex(blob, r"\bBasic\b")
            self.assertNotIn("AI setup notes", blob)       # Run 85: notes are not AI

    def test_account_name(self):
        self.assertEqual(account_name("", "realtest123@example.com"), "realtest123")
        self.assertEqual(account_name("Howard", "h@example.com"), "Howard")
        self.assertEqual(account_name(None, ""), "Account")


class ArchitectureTests(unittest.TestCase):
    def test_card_is_presentation_only(self):
        src = (ROOT / "ui" / "account_card.py").read_text()
        for lookup in ("from db", "import db", "resolve_user_tier", "load_user_map", "session_tier_state",
                       "get_neon_conn", "compute_entitlements"):
            self.assertNotIn(lookup, src, lookup)

    def test_every_page_path_uses_the_shared_card(self):
        nav = (ROOT / "ui" / "nav.py").read_text()
        self.assertIn("render_account_card(key_suffix=key_suffix", nav)
        profile = (ROOT / "ui" / "app_user_profile.py").read_text()
        self.assertIn("render_account_card(key_suffix=\"sidebar\")", profile)
        self.assertNotIn('"Log out", key="logout_button"', profile)

    def test_no_other_sidebar_plan_or_upgrade_card_rendering(self):
        hits = []
        for path in [*(ROOT / "ui").glob("*.py"), *(ROOT / "pages").glob("*.py"), ROOT / "app.py"]:
            if path.name in ("account_card.py", "app_runtime.py"):
                continue          # the card itself; app_runtime keeps the unused legacy card for import safety
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"sidebar\.markdown\(f?\"\*\*Plan:\*\*|You're on (Free|Pro|Premium)\"", line):
                    hits.append(f"{path.name}:{i}")
        self.assertEqual(hits, [])


CAPTURE = '''
import runpy, streamlit as st
from streamlit.delta_generator import DeltaGenerator
links = st.session_state.setdefault("_links", [])
def _link(*a, **k):
    links.append(k.get("label") or (a[1] if len(a) > 1 else ""))
st.page_link = _link
DeltaGenerator.page_link = lambda self, *a, **k: _link(*a, **k)
st.switch_page = lambda *a, **k: st.stop()
runpy.run_path(%r, run_name="__main__")
'''

PAGES = {"Today": "pages/today.py", "Market Brief": "pages/brief.py", "Day Trader": "pages/day_trader.py",
         "Stock Intelligence": "pages/stock.py", "How HSF works": "pages/methodology.py",
         "My Stocks": "pages/watchlists.py", "Journal": "pages/journal.py", "Settings": "pages/settings.py",
         "Billing": "pages/billing.py", "Alerts": "pages/alerts.py", "Kalshi BTC": "pages/kalshi.py"}


def card_signature(at):
    """What the sidebar account card shows: its markdown, captions and buttons, in order."""
    sb = at.sidebar
    md = [m.value for m in sb.markdown if not m.value.startswith("<")]
    card_md = [m for m in md if m.startswith(("### 👤", "**👤", "**Plan:**", "**You're on", "**Admin access"))]
    caps = [c.value for c in sb.caption if "alerts" in c.value or "Discover" in c.value or "enabled" in c.value]
    buttons = [b.label for b in sb.button]
    return tuple(card_md), tuple(caps), tuple(b for b in buttons if b.startswith(("Upgrade", "Log out")))


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CrossPageCardTests(unittest.TestCase):
    def render_page(self, page, tier):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(CAPTURE % str(ROOT / page), default_timeout=120)
        for k, v in {"username": "realtest123@example.com", "display_name": "realtest123", "tier_key": tier,
                     "tier": SimpleNamespace(key=tier, name=tier.upper()), "is_admin": tier == "admin",
                     "entitlements": dict(_flags(tier))}.items():
            at.session_state[k] = v
        at.run()
        return at

    def expected(self, tier):
        c = plan_card(tier, is_admin=tier == "admin")
        buttons = ((c["cta_label"],) if c["cta_label"] else ()) + ("Log out",)
        return c, buttons

    def test_same_card_on_every_page_for_every_tier(self):
        for tier in ("basic", "pro", "premium", "admin"):
            c, buttons = self.expected(tier)
            sigs = {}
            for name, page in PAGES.items():
                at = self.render_page(page, tier)
                self.assertFalse(at.exception, (name, tier, [str(e.value)[:200] for e in at.exception]))
                sig = card_signature(at)
                sigs[name] = sig
                self.assertIn(f"**Plan:** `{c['label']}`", sig[0], (name, tier))
                self.assertEqual(sig[2], buttons, (name, tier))
                links = at.session_state["_links"]
                self.assertEqual("Compare all plans" in links, bool(c["compare"]), (name, tier))
            self.assertEqual(len(set(sigs.values())), 1, (tier, sigs))

    def test_scanner_renders_the_identical_card(self):
        import test_run83b_scanner_state as scanner

        for tier in ("basic", "pro", "premium"):
            c, buttons = self.expected(tier)
            at = scanner.run_app(tier=tier, session={"display_name": "realtest123"})
            self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
            sig = card_signature(at)
            self.assertIn(f"**Plan:** `{c['label']}`", sig[0], tier)
            self.assertIn(f"**{c['headline']}**", sig[0], tier)
            self.assertEqual(sig[2], buttons, tier)
            page = self.render_page("pages/today.py", tier)
            self.assertEqual(card_signature(page)[0][1:], sig[0][1:], tier)   # same card (name source aside)

    def test_card_never_touches_the_database(self):
        from unittest import mock

        from streamlit.testing.v1 import AppTest

        script = '''
import streamlit as st
from ui.account_card import render_account_card
with st.sidebar:
    render_account_card()
'''
        boom = mock.Mock(side_effect=AssertionError("account card must not look anything up"))
        at = AppTest.from_string(script, default_timeout=60)
        at.session_state["username"] = "realtest123@example.com"
        at.session_state["tier_key"] = "pro"
        with mock.patch("auth.tier_sync.resolve_user_tier", boom), \
             mock.patch("ui.user_lookup.load_user_map", boom), mock.patch("db.engine.get_neon_conn", boom):
            at.run()
        boom.assert_not_called()
        self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
        self.assertIn("**Plan:** `Pro`", [m.value for m in at.sidebar.markdown])


if __name__ == "__main__":
    unittest.main()
