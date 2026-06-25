import unittest

from scan.options import (
    DEFAULT_MARKET,
    DEFAULT_PROFILE,
    DEFAULT_STRATEGY,
    apply_admin_caps,
    build_scan_run_options,
    normalize_market,
    normalize_profile,
    normalize_strategy,
    profile_options,
)


class ScanOptionsTests(unittest.TestCase):
    def test_normalizes_unknown_selection_values_to_defaults(self):
        self.assertEqual(normalize_market("bad"), DEFAULT_MARKET)
        self.assertEqual(normalize_strategy("bad"), DEFAULT_STRATEGY)
        self.assertEqual(normalize_profile("bad"), DEFAULT_PROFILE)

    def test_admin_caps_expand_without_lowering_custom_large_values(self):
        self.assertEqual(apply_admin_caps(2000, 4000, 150, is_admin=False), (2000, 4000, 150))
        self.assertEqual(apply_admin_caps(2000, 4000, 150, is_admin=True), (100000, 150000, 10000))
        self.assertEqual(apply_admin_caps(200000, 300000, 20000, is_admin=True), (200000, 300000, 20000))

    def test_profile_options_use_existing_defaults(self):
        self.assertEqual(profile_options("aggressive", {}).min_gap, 0.0)
        self.assertFalse(profile_options("aggressive", {}).unusual_volume)
        self.assertEqual(profile_options("regular", {}).min_gap, 1.0)
        self.assertTrue(profile_options("regular", {}).unusual_volume)
        self.assertEqual(profile_options("conservative", {}).min_gap, 3.0)
        self.assertTrue(profile_options("conservative", {}).unusual_volume)

    def test_build_scan_run_options_reads_sidebar_values(self):
        values = {
            "min_price": 2.5,
            "max_price": 55,
            "top_n": 25,
            "premarket": True,
            "afterhours": True,
            "min_gap_pct": 4.5,
            "unusual_vol": False,
        }

        opts = build_scan_run_options("regular", values, is_admin=False)

        self.assertEqual(opts.min_price, 2.5)
        self.assertEqual(opts.max_price, 55)
        self.assertEqual(opts.top_n, 25)
        self.assertTrue(opts.premarket)
        self.assertTrue(opts.afterhours)
        self.assertEqual(opts.min_gap, 4.5)
        self.assertFalse(opts.unusual_volume)


if __name__ == "__main__":
    unittest.main()
