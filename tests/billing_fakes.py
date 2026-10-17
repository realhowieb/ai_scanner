"""Transactional in-memory boundary for billing contracts; real SQL is tested in CI."""
import copy
import threading
from contextlib import contextmanager


class MemoryStore:
    def __init__(self, billing):
        self.b = billing
        self.lock = threading.RLock()
        self.snapshots = {}
        self.intents = {}
        self.events = set()
        self.failures = {}
        self.sync_errors = {}
        self.fail_mark = False

    @contextmanager
    def transaction(self, email):
        with self.lock:
            backup = copy.deepcopy((self.snapshots, self.intents, self.events))
            user = self.b._get_user_by_email(email)
            if not user:
                raise LookupError("account missing")
            tx = MemoryTransaction(self, email, {**user, "username": email})
            try:
                yield tx
            except Exception:
                self.snapshots, self.intents, self.events = backup
                raise

    def read(self, email):
        return self.snapshots.get(email, {}), self.intents.get(email, {})

    def processed(self, event_id):
        return event_id in self.events

    def failure(self, event_id, kind, code):
        self.failures[event_id] = {"event_type": kind, "code": code, "at": "2026-10-16T00:00:00Z"}

    def sync_failure(self, email, code):
        self.sync_errors[email] = code

    def diagnostics(self):
        return {"last_webhook_at": None, "failed_syncs": len(self.sync_errors), "webhook_failures": list(self.failures.values())}


class MemoryTransaction:
    def __init__(self, store, email, user):
        self.store, self.email, self.user = store, email, user
        self.snapshot, self.intent = store.read(email)

    def processed(self, event_id):
        return self.store.processed(event_id)

    def mark(self, event_id, kind):
        if self.store.fail_mark:
            raise RuntimeError("event ledger unavailable")
        self.store.events.add(event_id)
        self.store.failures.pop(event_id, None)

    def save_snapshot(self, state, customer, subscription, price):
        if not (self.user.get("is_admin") or self.user.get("tier") == "admin"):
            self.store.b._set_user_plan_by_email(email=self.email, tier=state["plan"], stripe_customer_id=customer,
                                              stripe_subscription_id=subscription, stripe_price_id=price)
        self.store.snapshots[self.email] = state
        self.store.sync_errors.pop(self.email, None)
        self.snapshot = state

    def save_intent(self, intent):
        self.intent = intent
        self.store.intents[self.email] = intent
