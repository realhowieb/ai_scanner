"""Page-speed caching: stale-while-revalidate, one load per key, and the short
account-row cache (perf thread, 2026-10-08)."""
import importlib.util
import threading
import time
import unittest
from unittest import mock

from api.today import TTLCache


class StaleWhileRevalidateTests(unittest.TestCase):
    def test_expired_value_is_served_while_one_background_reload_runs(self):
        cache = TTLCache(8)
        cache.get("k", lambda: "old", ttl_s=0.01, stale_s=60)
        time.sleep(0.02)
        started, release = threading.Event(), threading.Event()
        calls = []

        def slow():
            calls.append(1)
            started.set()
            release.wait(2)
            return "new"

        self.assertEqual(cache.get("k", slow, ttl_s=60, stale_s=60), "old")
        self.assertTrue(started.wait(2))
        self.assertEqual(cache.get("k", slow, ttl_s=60, stale_s=60), "old")  # no second reload
        release.set()
        for _ in range(100):
            if cache.get("k", slow, ttl_s=60, stale_s=60) == "new":
                break
            time.sleep(0.01)
        self.assertEqual(cache.get("k", slow, ttl_s=60, stale_s=60), "new")
        self.assertEqual(len(calls), 1)

    def test_past_the_stale_window_it_loads_in_the_foreground(self):
        cache = TTLCache(8)
        cache.get("k", lambda: "old", ttl_s=0.01, stale_s=0.01)
        time.sleep(0.03)
        self.assertEqual(cache.get("k", lambda: "new", ttl_s=60), "new")

    def test_a_failed_background_reload_keeps_the_stale_value(self):
        cache = TTLCache(8)
        cache.get("k", lambda: "old", ttl_s=0.01, stale_s=60)
        time.sleep(0.02)

        def boom():
            raise RuntimeError("market data down")

        self.assertEqual(cache.get("k", boom, ttl_s=60, stale_s=60), "old")
        time.sleep(0.05)
        self.assertEqual(cache.get("k", boom, ttl_s=60, stale_s=60), "old")

    def test_concurrent_misses_share_one_load(self):
        cache = TTLCache(8)
        calls, out = [], []

        def slow():
            calls.append(1)
            time.sleep(0.1)
            return "v"

        threads = [threading.Thread(target=lambda: out.append(cache.get("k", slow, ttl_s=60))) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(out, ["v"] * 5)
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("fastapi", "jwt", "bcrypt")),
                     "needs fastapi, PyJWT and bcrypt")
class AccountCacheTests(unittest.TestCase):
    def setUp(self):
        from api import main

        self.main = main
        main._account_cache.clear()
        self.addCleanup(main._account_cache.clear)

    def test_the_row_is_reused_briefly_and_never_carries_the_password(self):
        row = {"username": "a@example.com", "password": "hash", "tier": "pro", "is_active": True}
        with mock.patch("api.store.get_account", return_value=dict(row)) as get:
            first = self.main._recent_account("A@example.com")
            second = self.main._recent_account("a@example.com")
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("password", first)
        self.assertEqual(second["tier"], "pro")

    def test_it_expires_and_forget_account_drops_it(self):
        with mock.patch("api.store.get_account", return_value={"username": "a@example.com", "tier": "pro"}) as get:
            self.main._recent_account("a@example.com")
            self.main.forget_account("a@example.com")
            self.main._recent_account("a@example.com")
            with mock.patch.object(self.main, "ACCOUNT_CACHE_S", 0):
                self.main._account_cache.clear()
                self.main._recent_account("a@example.com")
                self.main._recent_account("a@example.com")
        self.assertEqual(get.call_count, 4)


if __name__ == "__main__":
    unittest.main()
