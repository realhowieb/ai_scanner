"""Feature 2: Alpaca positions/orders client + durable order-event feed."""
from __future__ import annotations

import unittest
from unittest import mock


class _Cur:
    def __init__(self, fetchall=None):
        self._fetchall = fetchall or []
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return self._fetchall

    def close(self):
        pass


class _Conn:
    def __init__(self, fetchall=None):
        self.cur = _Cur(fetchall)

    def cursor(self, *a, **k):
        return self.cur

    def commit(self):
        pass

    def close(self):
        pass


class ClientTests(unittest.TestCase):
    def _resp(self, code, data):
        r = mock.MagicMock()
        r.status_code = code
        r.json.return_value = data
        return r

    def test_get_positions_maps_fields(self):
        import data.alpaca_trading as at

        req = mock.MagicMock()
        req.get.return_value = self._resp(200, [
            {"symbol": "CNTA", "qty": "20", "avg_entry_price": "12.45",
             "current_price": "13.00", "market_value": "260",
             "unrealized_pl": "11", "unrealized_plpc": "0.044"},
        ])
        with mock.patch.object(at, "requests", req):
            out = at.get_positions("K", "S")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["symbol"], "CNTA")
        self.assertEqual(out[0]["unrealized_plpc"], "0.044")

    def test_get_orders_none_on_http_error(self):
        import data.alpaca_trading as at

        req = mock.MagicMock()
        req.get.return_value = self._resp(500, {"message": "boom"})
        with mock.patch.object(at, "requests", req):
            self.assertIsNone(at.get_orders("K", "S"))


class EventStoreTests(unittest.TestCase):
    def test_sync_upserts_each_order_with_id(self):
        import db.paper_events as pe

        conn = _Conn()
        orders = [
            {"id": "o1", "symbol": "CNTA", "side": "buy", "qty": "20",
             "filled_qty": "20", "type": "market", "status": "filled",
             "filled_avg_price": "12.50", "submitted_at": "2026-07-22T13:00:00Z",
             "filled_at": "2026-07-22T13:00:01Z"},
            {"id": None, "symbol": "SKIP"},          # no id → skipped
        ]
        with mock.patch.object(pe, "_get_conn", return_value=conn):
            n = pe.sync_orders("U@x.com", orders)
        self.assertEqual(n, 1)
        inserts = [c for c in conn.cur.calls
                   if "INSERT INTO alpaca_paper_order_events" in c[0]]
        self.assertEqual(len(inserts), 1)
        self.assertEqual(inserts[0][1][0], "u@x.com")   # user lowercased
        self.assertEqual(inserts[0][1][1], "o1")

    def test_list_events_maps_tuples(self):
        import db.paper_events as pe

        tup = ("o1", "CNTA", "buy", 20.0, 20.0, "market", "filled", 12.5,
               None, None, None)
        conn = _Conn(fetchall=[tup])
        with mock.patch.object(pe, "_get_conn", return_value=conn):
            out = pe.list_events("u@x.com")
        self.assertEqual(out[0]["order_id"], "o1")
        self.assertEqual(out[0]["status"], "filled")

    def test_sync_empty_is_noop(self):
        import db.paper_events as pe
        with mock.patch.object(pe, "_get_conn",
                               side_effect=AssertionError("must not connect")):
            self.assertEqual(pe.sync_orders("u", []), 0)


if __name__ == "__main__":
    unittest.main()
