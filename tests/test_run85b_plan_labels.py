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
HAS_ST = importlib.util.find_spec("streamlit") is not None
EXPECTED = {"basic": "Free", "pro": "Pro", "premium": "Premium", "admin": "Admin"}


class PlanLabelTests(unittest.TestCase):
    def test_internal_keys_map_to_customer_names(self):
        for key, label in EXPECTED.items():
            self.assertEqual(plan_label(key), label)
            self.assertEqual(plan_label(key.upper()), label)
        self.assertEqual(plan_label("free"), "Free")

    def test_tier_objects_and_missing_values(self):
        self.assertEqual(plan_label(SimpleNamespace(key="basic", name="BASIC")), "Free")
        self.assertEqual(plan_label(SimpleNamespace(key="premium", name="PREMIUM")), "Premium")
        self.assertEqual(plan_label(SimpleNamespace(name="Free")), "Free")
        for missing in (None, "", "unknown-tier"):
            self.assertEqual(plan_label(missing), "Free")

    def test_admin_role_wins(self):
        self.assertEqual(plan_label("basic", is_admin=True), "Admin")

    def test_no_label_is_basic(self):
        self.assertNotIn("Basic", PLAN_LABELS.values())

    def test_pricing_uses_the_same_names(self):
        from ui.pricing import TIER_NAMES

        self.assertEqual(TIER_NAMES, {"basic": "Free", "pro": "Pro", "premium": "Premium"})

    def test_entitlements_still_keyed_on_internal_basic(self):
        from ui.app_session import FEATURE_MIN_TIER, TIER_ORDER

        self.assertIn("basic", TIER_ORDER)
        self.assertEqual(FEATURE_MIN_TIER["can_scan_sp500"], "basic")
        self.assertNotIn("free", TIER_ORDER)


class SourceTests(unittest.TestCase):
    """No page formats a tier key into customer text on its own."""

    def test_plan_displays_use_the_helper(self):
        for rel in ("ui/nav.py", "ui/app_user_profile.py", "pages/billing.py", "pages/settings.py"):
            self.assertIn("plan_label(", (ROOT / rel).read_text(), rel)

    def test_no_raw_tier_formatting_in_customer_copy(self):
        bad = re.compile(r"tier_key[\"')\s]*(or \"basic\"\)?)?\s*\)?\.(title|upper|capitalize)\(\)"
                         r"|prev_key\.upper\(\)|current_key\.title\(\)")
        hits = []
        for path in [*(ROOT / "ui").glob("*.py"), *(ROOT / "pages").glob("*.py"), ROOT / "app.py",
                     ROOT / "auth" / "tiering.py"]:
            for i, line in enumerate(path.read_text().splitlines(), 1):
                customer_text = 'f"' in line or "f'" in line or "st." in line   # displayed, not internal
                if bad.search(line) and customer_text:
                    hits.append(f"{path.name}:{i}: {line.strip()[:90]}")
        self.assertEqual(hits, [])

    def test_no_basic_plan_wording_in_customer_copy(self):
        pat = re.compile(r"(You['’]re (currently )?on|Upgrade from|Plan:)\s*\**`?Basic|Basic plan", re.IGNORECASE)
        hits = []
        for path in [*(ROOT / "ui").glob("*.py"), *(ROOT / "pages").glob("*.py"), ROOT / "app.py"]:
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pat.search(line) and not line.lstrip().startswith("#"):
                    hits.append(f"{path.name}:{i}: {line.strip()[:90]}")
        self.assertEqual(hits, [])


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


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CrossPageTests(unittest.TestCase):
    def render(self, page, tier):
        from streamlit.testing.v1 import AppTest

        from ui.app_session import compute_entitlements

        is_admin = tier == "admin"
        flags = compute_entitlements(tier_obj=SimpleNamespace(key=tier, name=tier.upper()), is_admin=is_admin)
        at = AppTest.from_string(PAGE_SCRIPT % str(ROOT / page), default_timeout=120)
        for k, v in {"username": "realtest123@example.com", "tier_key": tier, "is_admin": is_admin,
                     "tier": SimpleNamespace(key=tier, name=tier.upper()), "entitlements": dict(flags),
                     "display_name": "realtest123"}.items():
            at.session_state[k] = v
        at.run()
        return at

    def test_every_page_shows_the_same_plan_name(self):
        for tier, label in EXPECTED.items():
            seen = {}
            for name, page in SUB_PAGES.items():
                at = self.render(page, tier)
                self.assertFalse(at.exception, (name, tier, [str(e.value)[:200] for e in at.exception]))
                seen[name] = _sidebar_plan(at)
                if tier == "basic":
                    self.assertNotRegex(_visible_text(at), r"\bBasic\b", (name, "customer-visible 'Basic'"))
            self.assertEqual(set(seen.values()), {label}, (tier, seen))

    def test_scanner_sidebar_matches(self):
        # The Scanner sidebar comes from the real app.py (after the Today landing).
        import test_run83b_scanner_state as scanner

        for tier, label in EXPECTED.items():
            session = {}
            if tier == "admin":
                import auth.tiering as tiering
                saved = tiering.ADMIN_USERS
                tiering.ADMIN_USERS = set(saved) | {scanner.USER}
                self.addCleanup(setattr, tiering, "ADMIN_USERS", saved)
            at = scanner.run_app(tier=tier, session=session)
            self.assertFalse(at.exception, [str(e.value)[:200] for e in at.exception])
            self.assertEqual(_sidebar_plan(at), label, tier)
            if tier == "basic":
                self.assertNotRegex(_visible_text(at), r"\bBasic\b")


if __name__ == "__main__":
    unittest.main()
