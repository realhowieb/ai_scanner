"""P0-8 — per-rerun cost: single-user lookup and cached daily research reads."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None


class UserLookupTests(unittest.TestCase):
    def _lookup(self, rec):
        from ui import user_lookup

        with mock.patch("db.users.get_user_by_username", return_value=rec) as g:
            out = user_lookup._lookup("  Alice@Example.com ")
        return out, g

    def test_single_user_map_has_load_users_shape_without_password(self):
        out, g = self._lookup({"username": "alice@example.com", "full_name": "Alice", "tier": "pro",
                               "is_admin": False, "is_active": True})
        g.assert_called_once_with("alice@example.com")
        self.assertEqual(out, {"alice@example.com": {"name": "Alice", "tier": "pro"}})
        self.assertNotIn("password", out["alice@example.com"])

    def test_missing_tier_defaults_to_basic(self):
        out, _ = self._lookup({"username": "a", "full_name": None, "tier": None, "is_active": True})
        self.assertEqual(out["alice@example.com"]["tier"], "basic")

    def test_inactive_or_unknown_users_are_absent_like_load_users(self):
        self.assertEqual(self._lookup({"tier": "pro", "is_active": False})[0], {})
        self.assertEqual(self._lookup(None)[0], {})

    def test_blank_username_makes_no_query(self):
        from ui import user_lookup

        with mock.patch("db.users.get_user_by_username") as g:
            self.assertEqual(user_lookup._lookup("  "), {})
        g.assert_not_called()

    def test_legacy_tier_fallback_behaves_as_before(self):
        from auth.tiering import get_user_tier

        tier = get_user_tier("alice@example.com", {"alice@example.com": {"name": "A", "tier": "pro"}})
        self.assertIn("pro", str(getattr(tier, "key", tier)).lower())

    def test_scanner_no_longer_loads_every_user(self):
        src = (ROOT / "app.py").read_text()
        main = src[src.index("def main():"):]
        self.assertIn("users_map = load_user_map(username)", main)
        self.assertNotIn("load_users()", main)


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CachedResearchReadsTests(unittest.TestCase):
    def test_track_record_summaries_query_once_per_cache_window(self):
        from ui import track_record as tr

        tr._load_summary_rows.clear()
        with mock.patch("db.track_record.load_latest_track_record",
                        return_value={"horizon_days": 5, "avg_return": 0.01}) as q:
            tr._load_summary_rows()
            tr._load_summary_rows()          # a second rerun
        self.assertEqual(q.call_count, len(tr.DEFAULT_HORIZONS) * len(tr.RANKING_LABELS))
        tr._load_summary_rows.clear()

    def test_heatmap_and_leaderboard_reads_are_cached(self):
        from ui import strategy_lab as sl
        from ui import track_record as tr

        for fn in (tr._daily_excess_cached, sl._horizons_cached, sl._leaderboard_cached):
            self.assertTrue(hasattr(fn, "clear"), fn)

        sl._leaderboard_cached.clear()
        with mock.patch("db.signal_leaderboard.load_leaderboard", return_value=[{"signal": "x"}]) as q:
            sl._leaderboard_cached(5, "close")
            sl._leaderboard_cached(5, "close")
            sl._leaderboard_cached(5, "open")     # different key → its own read
        self.assertEqual(q.call_count, 2)
        sl._leaderboard_cached.clear()


if __name__ == "__main__":
    unittest.main()
