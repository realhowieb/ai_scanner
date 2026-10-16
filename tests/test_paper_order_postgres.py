"""Optional real transaction checks, isolated in a fresh schema on a throwaway DB."""

import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

PG_URL = os.environ.get("HSF_TEST_PG_URL")


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class PaperOrderPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg import sql

        from db import paper_orders, trades

        self.schema = "paper_test_" + uuid.uuid4().hex
        with psycopg.connect(PG_URL) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(self.cleanup_schema)
        with self.connect() as conn:
            paper_orders.ensure_schema.__wrapped__(conn)
            trades._ensure_schema.__wrapped__(conn)

    def connect(self):
        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row

        conn = psycopg.connect(PG_URL, row_factory=dict_row)
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        return conn

    def cleanup_schema(self):
        import psycopg
        from psycopg import sql

        with psycopg.connect(PG_URL) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    def test_parallel_confirmation_and_out_of_order_fills_remain_single_records(self):
        from db import paper_orders, trades

        with patch.object(paper_orders, "_conn", side_effect=self.connect):
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(
                    pool.map(lambda _: paper_orders.reserve("u", "confirmation", "account", "AAPL", 2, {}), range(4))
                )
            self.assertEqual({r["client_order_id"] for r in rows}, {"confirmation"})
            self.assertIsNone(paper_orders.get("another-user", "confirmation"))
            self.assertEqual(paper_orders.get("u", "confirmation")["ticker"], "AAPL")
            paper_orders.save("u", "confirmation", state="submitted", result={"order_id": "order"})
            self.assertEqual(paper_orders.pending("u", "account")[0]["result"], {"order_id": "order"})

        with patch.object(trades, "_get_conn", side_effect=self.connect):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(
                    pool.map(
                        lambda qty: trades.reconcile_paper_fill("u", "order", "AAPL", 100.0, qty, {}), [2, 1, 2, 1]
                    )
                )
        with self.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM paper_order_requests").fetchone()["n"], 1)
            rows = conn.execute("SELECT shares, entry_price FROM user_trades").fetchall()
            self.assertEqual(rows, [{"shares": 2, "entry_price": 100.0}])


    def test_concurrent_delayed_responses_cannot_overwrite_a_completed_receipt(self):
        from db import paper_orders

        with patch.object(paper_orders, "_conn", side_effect=self.connect):
            paper_orders.reserve("u", "confirmation", "account", "AAPL", 2, {})
            full = {"order_id": "o1", "status": "filled", "filled_qty": "2"}
            paper_orders.save("u", "confirmation", state="synced", result=full)
            with ThreadPoolExecutor(max_workers=4) as pool:
                receipts = list(pool.map(lambda _: paper_orders.save(
                    "u", "confirmation", state="submitted",
                    result={"order_id": "o1", "status": "accepted", "filled_qty": "0"}), range(12)))
            self.assertTrue(all(receipt["state"] == "synced" and receipt["result"] == full for receipt in receipts))

    def test_journal_paging_and_direct_owned_lookup_reach_older_trades(self):
        from db import trades

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO user_trades (user_id,ticker,entry_price,shares) VALUES (%s,%s,%s,%s)",
                                [("u", "AAPL", 100, 2)] * 1002)
        with patch.object(trades, "_get_conn", side_effect=self.connect):
            self.assertEqual(len(trades.list_trades("u", limit=50, offset=1000)), 2)
            self.assertEqual(trades.get_trade("u", 1)["id"], 1)
            self.assertIsNone(trades.get_trade("someone-else", 1))
