"""Production Neon connections return rows as dicts keyed by column name
(psycopg dict_row), so two unnamed aggregates (COUNT(*), COUNT(x)) share the key
"count" and collapse. These run the real queries with that row shape."""
import re
import sqlite3
import unittest

from db import referrals


def _pg_name(name):
    """Postgres names an unaliased aggregate after its function: COUNT(x) -> "count"."""
    m = re.match(r"\s*(\w+)\s*\(", name)
    return m.group(1).lower() if m else name


def _dict_row(cursor, row):  # like psycopg.rows.dict_row: later duplicate names overwrite
    return {_pg_name(d[0]): v for d, v in zip(cursor.description, row)}


class ReferralStatsDictRowTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = _dict_row
        referrals.set_connection_factory(lambda: self.conn)
        self.addCleanup(referrals.set_connection_factory, None)
        self.addCleanup(self.conn.close)

    def test_stats_reads_both_counts(self):
        code = referrals.get_or_create_code("inviter@example.com")
        self.assertTrue(referrals.record_signup(code, "friend1@example.com"))
        self.assertTrue(referrals.record_signup(code, "friend2@example.com"))
        referrals.reward_verified("friend1@example.com", days=30, cap_days=180)
        self.assertEqual(referrals.stats("inviter@example.com"), {"signed_up": 2, "verified": 1})

    def test_summary_works(self):
        from api import referrals as api_referrals

        out = api_referrals.summary("someone@example.com")
        self.assertEqual((out["signed_up"], out["verified"]), (0, 0))
        self.assertTrue(out["path"].startswith("/signup?ref="))


if __name__ == "__main__":
    unittest.main()
