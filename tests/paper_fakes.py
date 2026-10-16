"""In-memory boundaries for paper-order recovery tests; no network or credentials."""

from copy import deepcopy
from threading import Lock


class MemoryOrders:
    def __init__(self):
        self.rows = {}
        self.lock = Lock()

    def reserve(self, user, cid, account_ref, ticker, qty, context):
        with self.lock:
            row = self.rows.setdefault(
                (user, cid),
                {
                    "user_id": user,
                    "client_order_id": cid,
                    "account_ref": account_ref,
                    "ticker": ticker,
                    "qty": qty,
                    "context": context,
                    "state": "pending",
                },
            )
            return deepcopy(row)

    def save(self, user, cid, *, state, result=None, error=None):
        with self.lock:
            row = self.rows[user, cid]
            row.update(state=state, error=error)
            if result:
                row["result"] = deepcopy(result)

    def pending(self, user, account_ref, limit=5):
        with self.lock:
            return [
                deepcopy(r)
                for r in self.rows.values()
                if r["user_id"] == user and r["state"] in ("pending", "submitted")
            ][:limit]


class PaperBroker:
    def __init__(self):
        self.orders = {}
        self.posts = []
        self.lock = Lock()

    def lookup(self, key, secret, cid):
        with self.lock:
            return deepcopy(self.orders.get((key, cid), {"ok": False, "not_found": True}))

    def submit(self, key, secret, ticker, qty, side="buy", *, client_order_id):
        with self.lock:
            if (key, client_order_id) in self.orders:
                return {"ok": False, "status_code": 422}
            self.posts.append(client_order_id)
            row = {
                "ok": True,
                "order_id": f"o{len(self.posts)}",
                "client_order_id": client_order_id,
                "symbol": ticker,
                "qty": str(qty),
                "side": side,
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
            self.orders[key, client_order_id] = row
            return deepcopy(row)
