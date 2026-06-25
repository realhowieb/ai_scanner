"""Tests for the password reset token logic in db/password_reset.py."""
from __future__ import annotations

import hashlib
import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


def _make_conn(rows=None, rowcount=0):
    """Return a mock psycopg connection whose cursor fetchone returns rows in order."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    if rows is not None:
        cur.fetchone.side_effect = rows
    cur.rowcount = rowcount
    return conn, cur


class TokenHashTest(unittest.TestCase):
    def test_hash_is_deterministic(self):
        from db.password_reset import _hash
        self.assertEqual(_hash("abc"), _hash("abc"))

    def test_hash_differs_for_different_tokens(self):
        from db.password_reset import _hash
        self.assertNotEqual(_hash("abc"), _hash("xyz"))

    def test_hash_is_sha256_hex(self):
        from db.password_reset import _hash
        result = _hash("test")
        expected = hashlib.sha256(b"test").hexdigest()
        self.assertEqual(result, expected)


class RateLimitTest(unittest.TestCase):
    def test_under_limit_not_rate_limited(self):
        from db.password_reset import _rate_limited
        cur = MagicMock()
        cur.fetchone.return_value = (2,)
        self.assertFalse(_rate_limited(cur, "user@example.com"))

    def test_at_limit_is_rate_limited(self):
        from db.password_reset import _RESET_MAX_PER_HOUR, _rate_limited
        cur = MagicMock()
        cur.fetchone.return_value = (_RESET_MAX_PER_HOUR,)
        self.assertTrue(_rate_limited(cur, "user@example.com"))

    def test_over_limit_is_rate_limited(self):
        from db.password_reset import _RESET_MAX_PER_HOUR, _rate_limited
        cur = MagicMock()
        cur.fetchone.return_value = (_RESET_MAX_PER_HOUR + 5,)
        self.assertTrue(_rate_limited(cur, "user@example.com"))


class CreateResetTokenTest(unittest.TestCase):
    def test_returns_none_when_no_db(self):
        from db import password_reset as pr
        with patch.object(pr, "get_neon_conn", None):
            result = pr.create_reset_token("u@example.com")
        self.assertIsNone(result)

    def test_returns_none_when_rate_limited(self):
        from db import password_reset as pr
        conn, cur = _make_conn(rows=[(3,)])  # rate-limit check returns 3
        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                result = pr.create_reset_token("u@example.com")
        self.assertIsNone(result)

    def test_returns_token_string_when_allowed(self):
        from db import password_reset as pr
        conn, cur = _make_conn(rows=[(0,)])  # rate-limit check returns 0
        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                result = pr.create_reset_token("u@example.com")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)


class ConsumeResetTokenTest(unittest.TestCase):
    def test_returns_none_when_no_db(self):
        from db import password_reset as pr
        with patch.object(pr, "get_neon_conn", None):
            result = pr.consume_reset_token("sometoken")
        self.assertIsNone(result)

    def test_returns_none_for_invalid_token(self):
        from db import password_reset as pr
        conn, cur = _make_conn(rows=[None])
        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                result = pr.consume_reset_token("badtoken")
        self.assertIsNone(result)

    def test_returns_username_and_marks_used_for_valid_token(self):
        from db import password_reset as pr
        conn, cur = _make_conn(rows=[(42, "u@example.com")])
        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                result = pr.consume_reset_token("validtoken")
        self.assertEqual(result, "u@example.com")
        # Verify UPDATE was called to mark the token used
        update_calls = [str(c) for c in cur.execute.call_args_list if "used = TRUE" in str(c)]
        self.assertTrue(len(update_calls) >= 1, "Token should be marked used")

    def test_consumed_token_cannot_be_reused(self):
        """Second call with same token finds no row (already marked used)."""
        from db import password_reset as pr
        # First call: valid row returned, second call: None
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value = cur
        cur.fetchone.side_effect = [(1, "u@example.com"), None]

        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                first = pr.consume_reset_token("t")
        # Second call needs a fresh conn mock returning None
        conn2, cur2 = _make_conn(rows=[None])
        with patch.object(pr, "get_neon_conn", return_value=conn2):
            with patch.object(pr, "_ensure_schema"):
                second = pr.consume_reset_token("t")
        self.assertEqual(first, "u@example.com")
        self.assertIsNone(second)


class DictRowCompatTest(unittest.TestCase):
    """Production uses psycopg dict_row — rows are dicts, not tuples.

    These guard against the KeyError: 0 / wrong-unpacking bug where row[0]
    or `a, b = row` was used on a dict cursor.
    """

    def test_rate_limited_handles_dict_row(self):
        from db.password_reset import _RESET_MAX_PER_HOUR, _rate_limited
        cur = MagicMock()
        cur.fetchone.return_value = {"count": _RESET_MAX_PER_HOUR}
        self.assertTrue(_rate_limited(cur, "u@example.com"))
        cur.fetchone.return_value = {"count": 0}
        self.assertFalse(_rate_limited(cur, "u@example.com"))

    def test_consume_token_handles_dict_row(self):
        from db import password_reset as pr
        conn, cur = _make_conn(rows=[{"id": 42, "username": "u@example.com"}])
        with patch.object(pr, "get_neon_conn", return_value=conn):
            with patch.object(pr, "_ensure_schema"):
                result = pr.consume_reset_token("validtoken")
        self.assertEqual(result, "u@example.com")


if __name__ == "__main__":
    unittest.main()
