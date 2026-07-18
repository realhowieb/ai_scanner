"""data/prices missing-symbol TTL cache + skipped-dedup pure logic."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS, "data.prices imports pandas")
class MissingCacheTests(unittest.TestCase):
    def _cfg(self):
        from data.prices import PriceFetchConfig

        return PriceFetchConfig(tickers=["AAA"], period="60d", interval="1d")

    def setUp(self):
        import data.prices as dp

        dp._MISSING_PRICE_CACHE.clear()

    def test_remembered_symbol_is_returned_while_fresh(self):
        import data.prices as dp

        cfg = self._cfg()
        dp._remember_missing("aaa", cfg, "delisted")
        self.assertEqual(dp._recent_missing_reason("AAA", cfg), "delisted")

    def test_expired_entry_is_evicted(self):
        import data.prices as dp

        cfg = self._cfg()
        key = dp._missing_cache_key("AAA", cfg)
        # Plant an entry older than the TTL.
        old_ts = 0.0  # epoch — definitely older than the 900s window
        dp._MISSING_PRICE_CACHE[key] = ("stale", old_ts)
        self.assertIsNone(dp._recent_missing_reason("AAA", cfg))
        self.assertNotIn(key, dp._MISSING_PRICE_CACHE)  # evicted on read

    def test_cache_key_varies_by_period_and_interval(self):
        import data.prices as dp
        from data.prices import PriceFetchConfig

        c1 = PriceFetchConfig(tickers=["X"], period="60d", interval="1d")
        c2 = PriceFetchConfig(tickers=["X"], period="1y", interval="1d")
        self.assertNotEqual(dp._missing_cache_key("X", c1), dp._missing_cache_key("X", c2))

    def test_dedupe_skipped_collapses_and_uppercases(self):
        import data.prices as dp

        out = dp._dedupe_skipped([("aaa", "delisted"), ("AAA", "delisted"), ("bbb", "timeout")])
        self.assertEqual(out, [("AAA", "delisted"), ("BBB", "timeout")])


if __name__ == "__main__":
    unittest.main()
