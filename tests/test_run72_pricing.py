"""Run 72 (P1-15) — public plans before sign-up, derived from real entitlements.

The Billing table, the benefits text and the landing "Plans" cards come from
ui.pricing, whose cells are derived from FEATURE_MIN_TIER / ALERT_LIMIT_BY_TIER.
Signed-out visitors see the plans (no "isn't linked to an email" message and no
signed-in navigation).
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True

# Claims we must never make on a pricing surface.
PROHIBITED = ("guarantee", "risk-free", "risk free", "can't lose", "profit", "win rate")


class DerivedPricingTests(unittest.TestCase):
    def test_every_row_matches_the_entitlement_map(self):
        from ui.app_session import FEATURE_MIN_TIER, TIER_ORDER
        from ui.pricing import ALERTS, ROWS, TIERS, included

        for label, flag in ROWS:
            if flag in (None, ALERTS):
                continue
            self.assertIn(flag, FEATURE_MIN_TIER, label)
            for tier in TIERS:
                want = TIER_ORDER[tier] >= TIER_ORDER[FEATURE_MIN_TIER[flag]]
                self.assertEqual(included(flag, tier), want, (label, tier))

    def test_admin_only_and_unknown_features_are_never_sold(self):
        from ui import pricing

        orig = dict(pricing.FEATURE_MIN_TIER)
        try:
            pricing.FEATURE_MIN_TIER["__admin_thing"] = "admin"
            for tier in pricing.TIERS:
                self.assertFalse(pricing.included("__admin_thing", tier))
                self.assertFalse(pricing.included("__missing", tier))
        finally:
            pricing.FEATURE_MIN_TIER.clear()
            pricing.FEATURE_MIN_TIER.update(orig)

    def test_table_shows_prices_and_alert_limits(self):
        from ui.app_session import ALERT_LIMIT_BY_TIER
        from ui.pricing import pricing_markdown

        md = pricing_markdown()
        for s in ("| Feature | Starter | Pro | Premium |", "$25/mo", "$40/mo"):
            self.assertIn(s, md)
        alert_row = next(line for line in md.splitlines() if line.startswith("| Alerts"))
        for tier in ("basic", "pro", "premium"):
            self.assertIn(f" {ALERT_LIMIT_BY_TIER[tier]} ", alert_row)

    def test_highlights_list_only_what_each_plan_adds(self):
        from ui.pricing import ALERTS, ROWS, included, plan_highlights

        h = plan_highlights()
        for label, flag in ROWS:
            if flag in (None, ALERTS):
                continue
            owners = [t for t in ("basic", "pro", "premium") if label in h[t]]
            self.assertLessEqual(len(owners), 1, label)       # added once, at its first tier
            if owners:
                self.assertTrue(included(flag, owners[0]))

    def test_no_prohibited_claims(self):
        from ui.pricing import benefits_markdown, plans_html, pricing_markdown

        text = (pricing_markdown() + benefits_markdown() + plans_html()).lower()
        for bad in PROHIBITED:
            self.assertNotIn(bad, text)

    def test_plan_cards(self):
        from ui.pricing import plans_html

        out = plans_html()
        cards = out.count('class="hsf-plan"') + out.count('class="hsf-plan featured"')
        self.assertEqual(cards, 3)
        self.assertEqual(out.count("Recommended"), 1)   # P2-21: no "Most popular" claim before customers
        self.assertNotIn("Most popular", out)
        self.assertIn("$0", out)
        self.assertIn("Everything in Starter", out)
        self.assertIn("Everything in Pro", out)


class WiringTests(unittest.TestCase):


    def test_nav_is_skipped_when_signed_out(self):
        import inspect

        from ui import nav

        src = inspect.getsource(nav.render_sidebar_nav)
        self.assertLess(src.index('st.session_state.get("username")'), src.index("render_top_menu()"))


SCRIPT = '''
import runpy
from utils.st_compat import st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
runpy.run_path(%r, run_name="__main__")
''' % str(ROOT / "pages" / "billing.py")


if __name__ == "__main__":
    unittest.main()
