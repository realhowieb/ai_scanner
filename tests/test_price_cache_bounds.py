"""data/prices frame cache stays bounded: expired entries are swept and a size cap applies."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS, "data.prices imports pandas")
class PriceCacheBoundsTests(unittest.TestCase):
    def setUp(self):
        import data.prices as dp

        self.dp = dp
        self.cache: dict = {}
        dp._last_sweep.pop(id(self.cache), None)

    def test_expired_entries_are_swept_on_write(self):
        self.cache["OLD|2y|1d|0"] = ("frame", 0.0)  # written long ago
        self.dp._cache_put(self.cache, "NEW|2y|1d|0", "frame", ttl_s=60.0)
        self.assertEqual(list(self.cache), ["NEW|2y|1d|0"])

    def test_cap_drops_the_oldest_writes(self):
        with mock.patch.object(self.dp, "_PRICE_CACHE_MAX_ENTRIES", 3), \
             mock.patch.object(self.dp, "_SWEEP_EVERY_S", 1e9):
            for i in range(5):
                self.dp._cache_put(self.cache, f"S{i}", "frame", ttl_s=3600.0)
            self.assertEqual(list(self.cache), ["S2", "S3", "S4"])
            self.dp._cache_put(self.cache, "S2", "fresh", ttl_s=3600.0)  # rewrite moves it to newest
            self.dp._cache_put(self.cache, "S5", "frame", ttl_s=3600.0)
            self.assertEqual(list(self.cache), ["S4", "S2", "S5"])

    def test_zero_means_no_cap(self):
        with mock.patch.object(self.dp, "_PRICE_CACHE_MAX_ENTRIES", 0):
            for i in range(50):
                self.dp._cache_put(self.cache, f"S{i}", "frame", ttl_s=3600.0)
        self.assertEqual(len(self.cache), 50)

    def test_size_is_reported(self):
        self.assertEqual(self.dp.price_cache_size(), len(self.dp._PRICE_CACHE))


if __name__ == "__main__":
    unittest.main()
