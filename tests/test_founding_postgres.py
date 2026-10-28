"""Founding 100 memberships on real Postgres: many buyers at the same moment never get
more than 75 Founder Pro or 25 Founder Elite seats, a shared founder number, or a
second membership. Uses only an explicit throwaway URL (the billing Postgres job's
HSF_TEST_BILLING_PG_URL); each test gets its own schema."""
import concurrent.futures
import datetime as dt
import importlib.util
import os
import threading
import unittest
import uuid

from domain import founding as rules

URL = os.getenv("HSF_TEST_BILLING_PG_URL", "")
DRIVER = "psycopg2" if importlib.util.find_spec("psycopg2") else ("psycopg" if importlib.util.find_spec("psycopg") else None)
UTC = dt.timezone.utc
T0 = dt.datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
PRO, ELITE = rules.FOUNDER_PRO, rules.FOUNDER_ELITE


def _connect(schema):
    if DRIVER == "psycopg2":
        import psycopg2

        return psycopg2.connect(URL, options=f"-csearch_path={schema}")
    import psycopg

    return psycopg.connect(URL, options=f"-csearch_path={schema}")


@unittest.skipUnless(URL and DRIVER, "set HSF_TEST_BILLING_PG_URL to a throwaway Postgres")
class FoundingPostgresTest(unittest.TestCase):
    def setUp(self):
        from db import founding

        self.db = founding
        self.schema = "hsf_founding_test_" + uuid.uuid4().hex
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'CREATE SCHEMA "{self.schema}"')
        conn.close()
        self.addCleanup(self._drop)
        os.environ["AI_SCANNER_SCHEMA_ONCE"] = "0"
        self.addCleanup(lambda: os.environ.pop("AI_SCANNER_SCHEMA_ONCE", None))
        founding.set_connection_factory(lambda: _connect(self.schema))
        self.addCleanup(lambda: founding.set_connection_factory(None))
        founding.availability(T0)  # create the tables before the rush

    def _drop(self):
        conn = _connect("public")
        conn.autocommit = True
        conn.cursor().execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        conn.close()

    def _rush(self, n, fn):
        start = threading.Barrier(min(n, 32))

        def go(i):
            if i < 32:
                start.wait()
            return fn(i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            return list(pool.map(go, range(n)))

    def _buy(self, i, tier):
        try:
            res = self.db.reserve(f"user{i}@example.com", tier, now=T0)
        except self.db.SoldOut:
            return None
        out = self.db.fulfill(res["id"], payment_reference=f"pi_{tier}_{i}", amount_cents=rules.PRICE_CENTS[tier], now=T0)
        return out["membership"]["membership_number"] if out["outcome"] == "activated" else None

    def inv(self):
        conn = _connect(self.schema)
        try:
            cur = conn.cursor()
            cur.execute("SELECT tier, held, sold FROM founding_inventory ORDER BY tier")
            return {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        finally:
            conn.close()

    def test_150_buyers_at_once_fill_exactly_75_pro_and_25_elite(self):
        results = self._rush(150, lambda i: self._buy(i, ELITE if i % 3 == 0 else PRO))
        numbers = sorted(n for n in results if n is not None)
        self.assertEqual(numbers, list(range(1, 101)))
        self.assertEqual(self.inv(), {ELITE: (0, 25), PRO: (0, 75)})
        a = self.db.availability(T0)
        self.assertEqual((a["sold"], a["remaining"]), (100, 0))

    def test_concurrent_holds_never_exceed_capacity(self):
        def hold(i):
            try:
                return self.db.reserve(f"h{i}@example.com", ELITE, now=T0)["id"]
            except self.db.SoldOut:
                return None

        held = [r for r in self._rush(80, hold) if r is not None]
        self.assertEqual(len(held), 25)
        self.assertEqual(self.inv()[ELITE], (25, 0))

    def test_one_account_clicking_many_times_gets_one_hold(self):
        ids = self._rush(24, lambda _: self.db.reserve("same@example.com", PRO, now=T0)["id"])
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(self.inv()[PRO], (1, 0))

    def test_duplicate_webhooks_in_parallel_activate_once(self):
        res = self.db.reserve("a@example.com", PRO, now=T0)
        outs = self._rush(16, lambda _: self.db.fulfill(res["id"], payment_reference="pi_dup", amount_cents=49_900,
                                                         now=T0)["outcome"])
        self.assertEqual(outs.count("activated"), 1)
        self.assertEqual(outs.count("already_done"), 15)
        self.assertEqual(self.inv()[PRO], (0, 1))

    def test_upgrades_and_purchases_race_for_the_last_elite_seats(self):
        for i in range(10):
            self._buy(f"pro{i}", PRO)
        for i in range(20):
            self._buy(f"elite{i}", ELITE)

        def go(i):
            try:
                if i < 10:
                    res = self.db.reserve(f"userpro{i}@example.com", ELITE, kind="upgrade", now=T0)
                    out = self.db.fulfill(res["id"], payment_reference=f"pi_up{i}", amount_cents=50_000, now=T0)
                else:
                    res = self.db.reserve(f"new{i}@example.com", ELITE, now=T0)
                    out = self.db.fulfill(res["id"], payment_reference=f"pi_new{i}", amount_cents=99_900, now=T0)
                return out["outcome"]
            except self.db.SoldOut:
                return "sold_out"

        outs = self._rush(20, go)
        self.assertEqual(outs.count("upgraded") + outs.count("activated"), 5)
        held, sold = self.inv()[ELITE]
        self.assertEqual((held, sold), (0, 25))
        pro_sold = self.inv()[PRO][1]
        self.assertEqual(pro_sold, 10 - outs.count("upgraded"))
        conn = _connect(self.schema)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT membership_number) FROM founding_memberships WHERE status = 'active'")
            total, distinct = cur.fetchone()
        finally:
            conn.close()
        self.assertEqual(total, distinct)
        self.assertEqual(total, 30 + outs.count("activated"))  # an upgrade never adds a member

    def test_policy_versions_and_acceptances(self):
        from db import legal as ldb
        from domain import legal

        rows = self._rush(8, lambda _: ldb.sync(legal.load(env={})))  # several servers starting together
        self.assertTrue(all(r == rows[0] for r in rows))
        ids = [r["id"] for r in rows[0].values()]
        added = self._rush(6, lambda _: ldb.record_acceptance("a@example.com", ids, checkout_session_id="cs_pg",
                                                              tier=PRO, kind="purchase", reservation_id=1))
        self.assertEqual(sum(added), 3)
        self.assertEqual(len(ldb.acceptances(checkout_session_id="cs_pg")), 3)
        conn = _connect(self.schema)
        try:
            cur = conn.cursor()
            with self.assertRaises(Exception):  # unknown version id
                cur.execute("INSERT INTO user_policy_acceptances (user_id, policy_version_id, accepted_at, "
                            "checkout_session_id) VALUES ('x', 99999, 'n', 'cs_x')")
            conn.rollback()
        finally:
            conn.close()

    def test_constraints_hold_without_the_code(self):
        conn = _connect(self.schema)
        try:
            cur = conn.cursor()
            for stmt in ("UPDATE founding_inventory SET sold = 76 WHERE tier = 'founder_pro'",
                         "UPDATE founding_inventory SET held = 20, sold = 6 WHERE tier = 'founder_elite'",
                         "INSERT INTO founding_memberships (user_id, membership_number, tier, status, purchase_price_cents, "
                         "created_at, updated_at) VALUES ('x', 101, 'founder_pro', 'expired', 0, 'n', 'n')"):
                with self.assertRaises(Exception, msg=stmt):
                    cur.execute(stmt)
                conn.rollback()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
