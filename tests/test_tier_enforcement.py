"""Tests for tier access control and auth rate limiting.

These cover the billing-critical paths: correct blocking of features by tier,
correct admin promotion/demotion, and login rate limiting behavior.
"""
from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock, patch

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

# ---------------------------------------------------------------------------
# Tier comparison helpers
# ---------------------------------------------------------------------------

from ui.app_session import compute_entitlements, is_admin_user, FEATURE_MIN_TIER
from auth.tiering import has_min_tier, require_min_tier, get_user_tier, Tier, TIER_ORDER


class TestHasMinTier(unittest.TestCase):
    def test_basic_cannot_access_pro_features(self):
        self.assertFalse(has_min_tier("basic", "pro"))

    def test_pro_can_access_basic_features(self):
        self.assertTrue(has_min_tier("pro", "basic"))

    def test_pro_cannot_access_premium_features(self):
        self.assertFalse(has_min_tier("pro", "premium"))

    def test_premium_can_access_pro_features(self):
        self.assertTrue(has_min_tier("premium", "pro"))

    def test_admin_can_access_all_tiers(self):
        for tier in ("basic", "pro", "premium", "admin"):
            self.assertTrue(has_min_tier("admin", tier), f"admin should access {tier}")

    def test_unknown_tier_treated_as_basic(self):
        self.assertFalse(has_min_tier("unknown_tier", "pro"))

    def test_tier_object_accepted(self):
        basic_tier = get_user_tier("demo_basic", {"demo_basic": {"tier": "basic"}})
        self.assertFalse(has_min_tier(basic_tier, "pro"))
        pro_tier = get_user_tier("demo_pro", {"demo_pro": {"tier": "pro"}})
        self.assertTrue(has_min_tier(pro_tier, "pro"))


class TestRequireMinTier(unittest.TestCase):
    def test_returns_true_when_tier_sufficient(self):
        self.assertTrue(require_min_tier("premium", "pro", "Test Feature"))

    def test_returns_false_and_shows_warning_when_insufficient(self):
        # require_min_tier does a lazy `import streamlit as st` inside the function
        # so we patch the streamlit module directly.
        import sys
        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            result = require_min_tier("basic", "pro", "NASDAQ Scan")
            self.assertFalse(result)
            mock_st.warning.assert_called_once()
            warning_text = mock_st.warning.call_args[0][0]
            self.assertIn("NASDAQ Scan", warning_text)
            self.assertIn("Pro", warning_text)


class TestGetUserTier(unittest.TestCase):
    def test_basic_user_gets_basic_tier(self):
        users = {"alice": {"tier": "basic"}}
        tier = get_user_tier("alice", users)
        self.assertEqual(tier.key, "basic")
        self.assertFalse(tier.can_premarket)
        self.assertFalse(tier.can_ai_notes)
        self.assertFalse(tier.can_export_csv)

    def test_pro_user_gets_pro_capabilities(self):
        users = {"bob": {"tier": "pro"}}
        tier = get_user_tier("bob", users)
        self.assertEqual(tier.key, "pro")
        self.assertTrue(tier.can_premarket)
        self.assertTrue(tier.can_export_csv)
        self.assertFalse(tier.can_ai_notes)

    def test_premium_user_gets_all_capabilities(self):
        users = {"carol": {"tier": "premium"}}
        tier = get_user_tier("carol", users)
        self.assertEqual(tier.key, "premium")
        self.assertTrue(tier.can_ai_notes)
        self.assertTrue(tier.can_premarket)
        self.assertTrue(tier.can_export_csv)

    def test_unknown_user_defaults_to_basic(self):
        tier = get_user_tier("nobody", {})
        self.assertEqual(tier.key, "basic")

    def test_invalid_tier_key_defaults_to_basic(self):
        users = {"eve": {"tier": "superduper"}}
        tier = get_user_tier("eve", users)
        self.assertEqual(tier.key, "basic")

    def test_neon_tier_overrides_local_fallback(self):
        # Neon users_map takes priority over USERS_DB
        neon_users = {"demo_basic": {"tier": "premium"}}
        tier = get_user_tier("demo_basic", neon_users)
        self.assertEqual(tier.key, "premium")


# ---------------------------------------------------------------------------
# Entitlement computation
# ---------------------------------------------------------------------------

class TestComputeEntitlements(unittest.TestCase):
    def _tier(self, key: str) -> Tier:
        return get_user_tier("u", {"u": {"tier": key}})

    def test_admin_gets_all_features(self):
        flags = compute_entitlements(tier_obj=self._tier("basic"), is_admin=True)
        self.assertTrue(all(flags.values()), f"Admin should have all features: {flags}")

    def test_basic_blocked_from_nasdaq(self):
        flags = compute_entitlements(tier_obj=self._tier("basic"), is_admin=False)
        self.assertFalse(flags.get("can_scan_nasdaq"))
        self.assertFalse(flags.get("can_export_csv"))
        self.assertFalse(flags.get("can_ai_notes"))

    def test_pro_gets_nasdaq_not_ai_notes(self):
        flags = compute_entitlements(tier_obj=self._tier("pro"), is_admin=False)
        self.assertTrue(flags.get("can_scan_nasdaq"))
        self.assertTrue(flags.get("can_export_csv"))
        self.assertFalse(flags.get("can_ai_notes"))

    def test_premium_gets_ai_notes(self):
        flags = compute_entitlements(tier_obj=self._tier("premium"), is_admin=False)
        self.assertTrue(flags.get("can_ai_notes"))

    def test_non_admin_blocked_from_admin_features(self):
        flags = compute_entitlements(tier_obj=self._tier("premium"), is_admin=False)
        self.assertFalse(flags.get("can_diagnostics"))
        self.assertFalse(flags.get("can_admin_panel"))

    def test_all_features_covered(self):
        flags = compute_entitlements(tier_obj=self._tier("basic"), is_admin=False)
        for feature in FEATURE_MIN_TIER:
            self.assertIn(feature, flags, f"Missing feature flag: {feature}")


# ---------------------------------------------------------------------------
# Admin user detection
# ---------------------------------------------------------------------------

class TestIsAdminUser(unittest.TestCase):
    def test_username_in_admin_set(self):
        self.assertTrue(is_admin_user("alice", None, admin_users={"alice", "bob"}))

    def test_username_not_in_admin_set(self):
        self.assertFalse(is_admin_user("carol", None, admin_users={"alice", "bob"}))

    def test_admin_tier_object_grants_admin(self):
        # TIERS_CONFIG has no "admin" entry so get_user_tier returns basic;
        # construct a Tier with key="admin" directly to test the tier-object path.
        admin_tier = Tier(
            key="admin", name="Admin", features=[], max_results=999,
            is_premium=True, can_premarket=True, can_afterhours=True,
            can_unusual_volume=True, can_export_csv=True, can_ai_notes=True,
        )
        self.assertTrue(is_admin_user("x", admin_tier, admin_users=set()))

    def test_case_insensitive_username_check(self):
        self.assertTrue(is_admin_user("ALICE", None, admin_users={"alice"}))

    def test_empty_username_not_admin(self):
        self.assertFalse(is_admin_user("", None, admin_users={"alice"}))

    def test_none_username_not_admin(self):
        self.assertFalse(is_admin_user(None, None, admin_users={"alice"}))


# ---------------------------------------------------------------------------
# Login rate limiting
# ---------------------------------------------------------------------------

def _mock_conn_with_cursor(fetchone_return):
    """Build a mock psycopg2-style connection whose cursor returns fetchone_return."""
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = fetchone_return
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


@unittest.skipUnless(_PANDAS_AVAILABLE, "pandas required for db.users import")
class TestLoginRateLimiting(unittest.TestCase):
    def test_no_rate_limit_when_db_unavailable(self):
        from db.users import is_login_rate_limited
        with patch("db.users.get_neon_conn", return_value=None):
            self.assertFalse(is_login_rate_limited("alice"))

    def test_rate_limited_after_max_attempts(self):
        from db.users import is_login_rate_limited, _LOGIN_MAX_ATTEMPTS
        mock_conn, _ = _mock_conn_with_cursor((_LOGIN_MAX_ATTEMPTS,))
        with patch("db.users.get_neon_conn", return_value=mock_conn):
            with patch("db.users.ensure_neon_login_attempts_schema"):
                self.assertTrue(is_login_rate_limited("alice"))

    def test_not_rate_limited_below_threshold(self):
        from db.users import is_login_rate_limited, _LOGIN_MAX_ATTEMPTS
        mock_conn, _ = _mock_conn_with_cursor((_LOGIN_MAX_ATTEMPTS - 1,))
        with patch("db.users.get_neon_conn", return_value=mock_conn):
            with patch("db.users.ensure_neon_login_attempts_schema"):
                self.assertFalse(is_login_rate_limited("alice"))


# ---------------------------------------------------------------------------
# DB admin grant/revoke
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PANDAS_AVAILABLE, "pandas required for db.users import")
class TestAdminDbOperations(unittest.TestCase):
    def test_grant_admin_updates_db(self):
        from db.users import grant_admin
        mock_conn, mock_cur = _mock_conn_with_cursor(None)
        mock_cur.rowcount = 1
        with patch("db.users.get_neon_conn", return_value=mock_conn):
            with patch("db.users.ensure_neon_users_schema"):
                result = grant_admin("alice")
        self.assertTrue(result)

    def test_grant_admin_returns_false_when_db_unavailable(self):
        from db.users import grant_admin
        with patch("db.users.get_neon_conn", return_value=None):
            self.assertFalse(grant_admin("alice"))

    def test_revoke_admin_returns_false_when_db_unavailable(self):
        from db.users import revoke_admin
        with patch("db.users.get_neon_conn", return_value=None):
            self.assertFalse(revoke_admin("alice"))

    def test_is_admin_from_db_returns_false_when_db_unavailable(self):
        from db.users import is_admin_from_db
        with patch("db.users.get_neon_conn", return_value=None):
            self.assertFalse(is_admin_from_db("alice"))

    def test_is_admin_from_db_checks_is_admin_flag(self):
        from db.users import is_admin_from_db
        mock_conn, _ = _mock_conn_with_cursor((True, "basic"))  # is_admin=True, tier=basic
        with patch("db.users.get_neon_conn", return_value=mock_conn):
            with patch("db.users.ensure_neon_users_schema"):
                self.assertTrue(is_admin_from_db("alice"))

    def test_is_admin_from_db_checks_tier_admin(self):
        from db.users import is_admin_from_db
        mock_conn, _ = _mock_conn_with_cursor((False, "admin"))  # is_admin=False but tier=admin
        with patch("db.users.get_neon_conn", return_value=mock_conn):
            with patch("db.users.ensure_neon_users_schema"):
                self.assertTrue(is_admin_from_db("alice"))


if __name__ == "__main__":
    unittest.main()
