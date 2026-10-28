"""Accounts whose billing sync keeps failing (Admin -> Billing diagnostics
"Accounts with sync errors"). Read-only. Run via the Diagnostics workflow.

Typical cause after the switch to live Stripe: the account still holds a
test-mode Stripe customer id, which the live key can't find. The repo is
public, so emails are masked and Stripe ids show only their last 6 chars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mask_email(email: str) -> str:
    name, _, domain = (email or "").partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def _tail(value) -> str:
    return f"...{str(value)[-6:]}" if value else "-"


def main() -> int:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        print("no Neon connection")
        return 1
    cur = conn.cursor()
    cur.execute(
        "SELECT u.username, u.tier, u.is_admin, u.stripe_customer_id, u.stripe_subscription_id, "
        "b.last_sync_error, b.last_sync_at, b.snapshot "
        "FROM hsf_billing_state b JOIN users u ON u.username = b.username "
        "WHERE b.last_sync_error IS NOT NULL ORDER BY b.last_sync_at DESC"
    )
    cols = ("username", "tier", "is_admin", "customer", "subscription", "error", "last_sync_at", "snapshot")
    rows = [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in (cur.fetchall() or [])]
    print(f"accounts with sync errors: {len(rows)}")
    for r in rows:
        snap = r.get("snapshot") or {}
        if isinstance(snap, str):
            snap = json.loads(snap or "{}")
        brief = {k: snap.get(k) for k in ("status", "plan", "interval", "test_mode", "livemode") if k in snap}
        print(f"- {_mask_email(r.get('username') or r.get('u.username', ''))} tier={r.get('tier')} "
              f"admin={r.get('is_admin')} customer={_tail(r.get('customer') or r.get('stripe_customer_id'))} "
              f"sub={_tail(r.get('subscription') or r.get('stripe_subscription_id'))} "
              f"error={r.get('error') or r.get('last_sync_error')} last_sync={r.get('last_sync_at')} snapshot={brief}")
    cur.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
