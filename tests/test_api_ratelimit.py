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


@unittest.skipUnless(DEPS, "needs FastAPI")
class ProxyClientIpTests(unittest.TestCase):
    """The web server forwards each visitor's address with a shared secret, so one busy
    website doesn't share a single sign-in limit across all of its visitors."""

    def _req(self, headers):
        from starlette.requests import Request

        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": ("10.0.0.1", 1)})

    def test_forwarded_address_needs_the_secret(self):
        h = {"X-Forwarded-For": "198.51.100.7", "X-HSF-Client-IP": "203.0.113.9", "X-HSF-Proxy-Key": "s3cret"}
        with patch.dict("os.environ", {"HSF_PROXY_SECRET": "s3cret"}):
            self.assertEqual(ratelimit.client_ip(self._req(h)), "203.0.113.9")
            self.assertEqual(ratelimit.client_ip(self._req({**h, "X-HSF-Proxy-Key": "wrong"})), "198.51.100.7")
            self.assertEqual(ratelimit.client_ip(self._req({**h, "X-HSF-Client-IP": "nope"})), "198.51.100.7")
        with patch.dict("os.environ", {"HSF_PROXY_SECRET": ""}):
            self.assertEqual(ratelimit.client_ip(self._req(h)), "198.51.100.7")
