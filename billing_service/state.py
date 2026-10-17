"""Durable billing snapshots and per-account transactions (psycopg2)."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class PostgresStore:
    def __init__(self, connect):
        self.connect = connect
        self._ready = False
        self._schema_lock = threading.Lock()

    @contextmanager
    def connection(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self):
        with self._schema_lock:
            if self._ready:
                return
            with self.connection() as conn, conn.cursor() as cur:
                # Serialize additive bootstrap across API workers/instances.
                cur.execute("SELECT pg_advisory_xact_lock(724381206)")
                cur.execute(Path(__file__).with_name("schema.sql").read_text())
            self._ready = True

    @contextmanager
    def transaction(self, email):
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            # Same lock for webhook, sync and checkout on every API instance.
            cur.execute("SELECT username, tier, stripe_customer_id, is_admin FROM users WHERE username=%s FOR UPDATE", (email,))
            row = cur.fetchone()
            if not row:
                raise LookupError("billing account not found")
            user = dict(zip(("username", "tier", "stripe_customer_id", "is_admin"), row))
            cur.execute("INSERT INTO hsf_billing_state(username) VALUES (%s) ON CONFLICT DO NOTHING", (email,))
            cur.execute("SELECT snapshot, checkout_intent FROM hsf_billing_state WHERE username=%s", (email,))
            snapshot, intent = cur.fetchone()
            yield Transaction(conn, user, snapshot, intent)

    def read(self, email):
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT snapshot, checkout_intent FROM hsf_billing_state WHERE username=%s", (email,))
            row = cur.fetchone()
            return row if row else ({}, {})

    def processed(self, event_id):
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM stripe_processed_events WHERE event_id=%s", (event_id,))
            return cur.fetchone() is not None

    def failure(self, event_id, event_type, code):
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO hsf_billing_failures(event_id,event_type,error_code) VALUES (%s,%s,%s) "
                        "ON CONFLICT(event_id) DO UPDATE SET error_code=EXCLUDED.error_code,failed_at=NOW()",
                        (event_id, event_type, code))

    def sync_failure(self, email, code):
        with self.transaction(email) as tx, tx.conn.cursor() as cur:
            cur.execute("UPDATE hsf_billing_state SET last_sync_at=NOW(),last_sync_error=%s WHERE username=%s", (code, email))

    def diagnostics(self):
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT MAX(processed_at) FROM stripe_processed_events")
            last = cur.fetchone()[0]
            cur.execute("SELECT event_type,error_code,failed_at FROM hsf_billing_failures ORDER BY failed_at DESC LIMIT 20")
            failures = [{"event_type": r[0], "code": r[1], "at": r[2].isoformat()} for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) FROM hsf_billing_state WHERE last_sync_error IS NOT NULL")
            return {"last_webhook_at": last.isoformat() if last else None,
                    "failed_syncs": cur.fetchone()[0], "webhook_failures": failures}

    def claim_reconciliation(self, limit=100):
        """Lease stale accounts for an hour across all API workers/instances."""
        self.ensure_schema()
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT u.username FROM users u LEFT JOIN hsf_billing_state b ON b.username=u.username "
                        "WHERE (u.stripe_customer_id IS NOT NULL OR b.checkout_intent->>'session_id' IS NOT NULL) "
                        "AND NOT COALESCE(u.is_admin,FALSE) "
                        "AND lower(u.tier) <> 'admin' AND (u.tier IN ('pro','premium') OR "
                        "b.snapshot->>'status' IN ('past_due','incomplete','unpaid') OR b.checkout_intent <> '{}'::jsonb) "
                        "AND (b.last_sync_at IS NULL OR b.last_sync_at < NOW()-INTERVAL '1 hour') "
                        "ORDER BY b.last_sync_at NULLS FIRST LIMIT %s FOR UPDATE OF u SKIP LOCKED", (limit,))
            emails = [r[0] for r in cur.fetchall()]
            for email in emails:
                cur.execute("INSERT INTO hsf_billing_state(username,last_sync_at) VALUES (%s,NOW()) "
                            "ON CONFLICT(username) DO UPDATE SET last_sync_at=NOW()", (email,))
            return emails


class Transaction:
    def __init__(self, conn, user, snapshot, intent):
        self.conn, self.user, self.snapshot, self.intent = conn, user, snapshot, intent

    def processed(self, event_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM stripe_processed_events WHERE event_id=%s", (event_id,))
            return cur.fetchone() is not None

    def mark(self, event_id, event_type):
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO stripe_processed_events(event_id,event_type) VALUES (%s,%s)", (event_id, event_type))
            cur.execute("DELETE FROM hsf_billing_failures WHERE event_id=%s", (event_id,))

    def save_snapshot(self, snapshot, customer_id, subscription_id, price_id):
        with self.conn.cursor() as cur:
            # Never overwrite an admin's tier or permissions through billing.
            cur.execute("UPDATE users SET tier=%s,stripe_customer_id=COALESCE(%s,stripe_customer_id),"
                        "stripe_subscription_id=%s,stripe_price_id=%s,plan_updated_at=NOW() "
                        "WHERE username=%s AND NOT COALESCE(is_admin,FALSE) AND lower(tier) <> 'admin'",
                        (snapshot["plan"], customer_id, subscription_id, price_id, self.user["username"]))
            cur.execute("UPDATE hsf_billing_state SET snapshot=%s::jsonb,last_sync_at=NOW(),last_sync_error=NULL "
                        "WHERE username=%s", (json.dumps(snapshot), self.user["username"]))
        self.snapshot = snapshot

    def save_intent(self, intent):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE hsf_billing_state SET checkout_intent=%s::jsonb WHERE username=%s",
                        (json.dumps(intent), self.user["username"]))
        self.intent = intent
