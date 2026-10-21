"""Run 85B — the same account shows the same plan name on every page.

Live testing: one account showed "Plan: Basic" on Stock Intelligence and
"Plan: Free" / "You're on Free" on Scanner. Internally the entry tier is
`basic`; customers see it as **Free**. All displays now go through
ui.plan_labels.plan_label; entitlement code keeps comparing internal keys.
"""
import importlib.util
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from ui.plan_labels import PLAN_LABELS, plan_label

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True
EXPECTED = {"basic": "Starter", "pro": "Pro", "premium": "Premium", "admin": "Admin"}


class PlanLabelTests(unittest.TestCase):
    def test_internal_keys_map_to_customer_names(self):
        for key, label in EXPECTED.items():
            self.assertEqual(plan_label(key), label)
            self.assertEqual(plan_label(key.upper()), label)
        self.assertEqual(plan_label("free"), "Starter")

    def test_tier_objects_and_missing_values(self):
        self.assertEqual(plan_label(SimpleNamespace(key="basic", name="BASIC")), "Starter")
        self.assertEqual(plan_label(SimpleNamespace(key="premium", name="PREMIUM")), "Premium")
        self.assertEqual(plan_label(SimpleNamespace(name="Free")), "Starter")
        for missing in (None, "", "unknown-tier"):
            self.assertEqual(plan_label(missing), "Starter")

    def test_admin_role_wins(self):
        self.assertEqual(plan_label("basic", is_admin=True), "Admin")

    def test_no_label_is_basic(self):
        self.assertNotIn("Basic", PLAN_LABELS.values())

    def test_pricing_uses_the_same_names(self):
        from ui.pricing import TIER_NAMES

        self.assertEqual(TIER_NAMES, {"basic": "Starter", "pro": "Pro", "premium": "Premium"})

    def test_entitlements_still_keyed_on_internal_basic(self):
        from ui.app_session import FEATURE_MIN_TIER, TIER_ORDER

        self.assertIn("basic", TIER_ORDER)
        self.assertEqual(FEATURE_MIN_TIER["can_scan_sp500"], "basic")
        self.assertNotIn("free", TIER_ORDER)


PAGE_SCRIPT = '''
import runpy, streamlit as st
from streamlit.delta_generator import DeltaGenerator
st.page_link = lambda *a, **k: None
DeltaGenerator.page_link = lambda self, *a, **k: None
st.switch_page = lambda *a, **k: st.stop()
runpy.run_path(%r, run_name="__main__")
'''

SUB_PAGES = {
    "Today": "pages/today.py", "Market Brief": "pages/brief.py", "Day Trader": "pages/day_trader.py",
    "Stock Intelligence": "pages/stock.py", "How HSF Works": "pages/methodology.py",
    "My Stocks": "pages/watchlists.py", "Journal": "pages/journal.py", "Settings": "pages/settings.py",
    "Billing": "pages/billing.py",
}


def _sidebar_plan(at):
    texts = [m.value for m in at.sidebar.markdown]
    for t in texts:
        m = re.search(r"\*\*Plan:\*\*\s*`?([A-Za-z]+)`?", t)
        if m:
            return m.group(1)
    return None


def _visible_text(at):
    parts = []
    for coll in (at.markdown, at.sidebar.markdown, at.caption, at.info, at.warning, at.success, at.subheader):
        parts += [str(getattr(e, "value", "")) for e in coll]
    return "\n".join(parts)


if __name__ == "__main__":
    unittest.main()
