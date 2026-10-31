"""db/sign_in_links.py: one-time "email me a sign-in link" tokens."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock, patch


def _conn(fetch):
    conn, cur = MagicMock(), MagicMock()
    conn.cursor.return_value = cur
    cur.fetchone.side_effect = fetch
    return conn, cur


class SignInLinkTokenTests(unittest.TestCase):
    def setUp(self):
        import db.sign_in_links as m

        self.m = m
        patch.object(m, "_ensure_schema", lambda conn: None).start()
        self.addCleanup(patch.stopall)

    def test_create_stores_only_the_hash_and_replaces_older_links(self):
        conn, cur = _conn([(0,)])
        with patch.object(self.m, "get_neon_conn", return_value=conn):
            token = self.m.create_token("pro@example.com")
        self.assertTrue(token and len(token) > 30)
        sql = [c.args[0] for c in cur.execute.call_args_list]
        self.assertTrue(any(s.startswith("UPDATE sign_in_link_tokens SET used = TRUE WHERE username") for s in sql))
        insert = next(c for c in cur.execute.call_args_list if c.args[0].startswith("INSERT"))
        self.assertEqual(insert.args[1][1], hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn(token, str(cur.execute.call_args_list))
        conn.commit.assert_called()
        conn.close.assert_called()

    def test_create_refuses_after_three_an_hour(self):
        conn, cur = _conn([(self.m.MAX_PER_HOUR,)])
        with patch.object(self.m, "get_neon_conn", return_value=conn):
            self.assertIsNone(self.m.create_token("pro@example.com"))
        self.assertFalse(any(c.args[0].startswith("INSERT") for c in cur.execute.call_args_list))

    def test_consume_marks_used_in_one_statement(self):
        conn, cur = _conn([("pro@example.com",)])
        with patch.object(self.m, "get_neon_conn", return_value=conn):
            self.assertEqual(self.m.consume_token("tok"), "pro@example.com")
        sql = cur.execute.call_args_list[0].args[0]
        self.assertIn("used = FALSE", sql)
        self.assertIn("expires_at > NOW()", sql)
        self.assertIn("RETURNING username", sql)

    def test_consume_unknown_or_used_token(self):
        conn, _ = _conn([None])
        with patch.object(self.m, "get_neon_conn", return_value=conn):
            self.assertIsNone(self.m.consume_token("tok"))

    def test_dict_rows(self):
        conn, _ = _conn([{"username": "pro@example.com"}])
        with patch.object(self.m, "get_neon_conn", return_value=conn):
            self.assertEqual(self.m.consume_token("tok"), "pro@example.com")


if __name__ == "__main__":
    unittest.main()
