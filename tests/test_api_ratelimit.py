"""Rate quotas survive key pressure and expire using their own windows."""
import unittest
from unittest.mock import patch

try:
    from fastapi import HTTPException

    from api import ratelimit

    DEPS = True
except ImportError:
    DEPS = False


@unittest.skipUnless(DEPS, "needs FastAPI")
class RateLimitTests(unittest.TestCase):
    def setUp(self):
        ratelimit.reset()
        self.addCleanup(ratelimit.reset)

    @patch("api.ratelimit._MAX_KEYS", 1)
    def test_key_pressure_preserves_long_window_quota(self):
        for _ in range(5):
            ratelimit.check("signup", "existing", now=0)
        with self.assertRaises(HTTPException) as pressure:
            ratelimit.check("login", "new", now=601)
        self.assertEqual(pressure.exception.status_code, 429)
        self.assertEqual(len(ratelimit._hits), 1)
        with self.assertRaises(HTTPException):
            ratelimit.check("signup", "existing", now=601)

    @patch("api.ratelimit._MAX_KEYS", 1)
    def test_expired_keys_free_capacity_at_window_boundary(self):
        ratelimit.check("login", "old", now=0)
        ratelimit.check("signup", "new", now=600)
        self.assertEqual(set(ratelimit._hits), {("signup", "new")})

    @patch("api.ratelimit._MAX_KEYS", 1)
    def test_existing_key_keeps_working_when_capacity_is_full(self):
        ratelimit.check("login", "existing", now=0)
        ratelimit.check("login", "existing", now=1)
        with self.assertRaises(HTTPException):
            ratelimit.check("login", "new", now=1)
        self.assertEqual(len(ratelimit._hits), 1)

    def test_retry_after_rounds_up_until_request_can_succeed(self):
        for _ in range(5):
            ratelimit.check("signup", "user", now=0)
        with self.assertRaises(HTTPException) as caught:
            ratelimit.check("signup", "user", now=3598.5)
        self.assertEqual(caught.exception.headers["Retry-After"], "2")
        ratelimit.check("signup", "user", now=3600.5)
