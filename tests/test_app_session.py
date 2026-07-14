import unittest
from types import SimpleNamespace

from ui.app_session import (
    compute_entitlements,
    is_admin_user,
    normalize_admin_users,
    tier_key,
)


class AppSessionTests(unittest.TestCase):
    def test_normalize_admin_users_accepts_lists_sets_and_dicts(self):
        self.assertEqual(normalize_admin_users([" Admin ", "HOWARD"]), {"admin", "howard"})
        self.assertEqual(normalize_admin_users({"Root": {}, "Ops": {}}), {"root", "ops"})

    def test_tier_key_prefers_key_then_name_then_string(self):
        self.assertEqual(tier_key(SimpleNamespace(key="Pro", name="Premium")), "pro")
        self.assertEqual(tier_key(SimpleNamespace(name="Premium")), "premium")
        self.assertEqual(tier_key(" Admin "), "admin")
        self.assertEqual(tier_key(None), "basic")

    def test_is_admin_user_checks_config_and_tier_shape(self):
        self.assertTrue(is_admin_user(" HOWARD ", None, admin_users={"howard"}))
        self.assertTrue(is_admin_user("user", SimpleNamespace(key="Admin"), admin_users=set()))
        self.assertTrue(is_admin_user("user", SimpleNamespace(name="Admin"), admin_users=set()))
        self.assertFalse(is_admin_user("user", SimpleNamespace(key="Premium"), admin_users=set()))

    def test_compute_entitlements_grants_all_for_admin(self):
        flags = compute_entitlements(tier_obj=SimpleNamespace(key="basic"), is_admin=True)

        self.assertTrue(flags)
        self.assertTrue(all(flags.values()))

    def test_compute_entitlements_uses_tier_order_fallback(self):
        flags = compute_entitlements(
            tier_obj=SimpleNamespace(key="pro"),
            is_admin=False,
            has_min_tier_fn=None,
        )

        self.assertTrue(flags["can_scan_sp500"])
        self.assertTrue(flags["can_scan_nasdaq"])
        self.assertTrue(flags["can_export_csv"])
        self.assertTrue(flags["can_track_record"])
        self.assertFalse(flags["can_early_breakout"])
        self.assertFalse(flags["can_admin_panel"])


if __name__ == "__main__":
    unittest.main()
