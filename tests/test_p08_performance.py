"""P0-8 — per-rerun cost: single-user lookup and cached daily research reads."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = True


class UserLookupTests(unittest.TestCase):
    def _lookup(self, rec):
        from ui import user_lookup

        with mock.patch("db.users.get_user_by_username", return_value=rec) as g:
            out = user_lookup._lookup("  Alice@Example.com ")
        return out, g


    def test_legacy_tier_fallback_behaves_as_before(self):
        from auth.tiering import get_user_tier

        tier = get_user_tier("alice@example.com", {"alice@example.com": {"name": "A", "tier": "pro"}})
        self.assertIn("pro", str(getattr(tier, "key", tier)).lower())


if __name__ == "__main__":
    unittest.main()
