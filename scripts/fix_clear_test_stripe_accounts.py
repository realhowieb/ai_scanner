"""One-off cleanup after the 2026-10-28 switch to live Stripe.

Three test accounts were upgraded with Stripe TEST customers. The live key
can't find them, so the hourly billing check fails on each one ("Accounts with
sync errors: 3") and the accounts keep a paid plan nobody pays for.

For exactly those three test-mode customer ids, this script:
  - sets the plan back to Starter (tier 'basic'),
  - clears the stale Stripe customer, subscription and price ids,
  - deletes the account's billing-state row (rebuilt empty on next read).

It only touches a row that still has a sync error, a test-mode snapshot and one
of the listed customer ids, never an admin, and it rolls back unless exactly
the expected rows change. Safe to re-run: a second run changes nothing.
Run via the Diagnostics workflow with script=fix_clear_test_stripe_accounts.py.
The repo is public, so output masks emails.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stripe TEST-mode customers (confirmed in the test dashboard 2026-10-28).
TEST_CUSTOMERS = ("cus_VSFH8lGbDIlRUJ", "cus_VNGZzLVmlkWeDx", "cus_VRmhN7fycrNeYX")

MATCH = (
    "FROM users u JOIN hsf_billing_state b ON b.username = u.username "
    "WHERE u.stripe_customer_id = ANY(%s) AND b.last_sync_error IS NOT NULL "
    "AND b.snapshot->>'test_mode' = 'true' "
    "AND NOT COALESCE(u.is_admin, FALSE) AND lower(COALESCE(u.tier, '')) <> 'admin'"
)


def _mask(email: str) -> str:
    name, _, domain = (email or "").partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def main() -> int:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        print("no Neon connection")
        return 1
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT u.username, u.tier, u.stripe_customer_id {MATCH} FOR UPDATE OF u",
                        (list(TEST_CUSTOMERS),))
            rows = cur.fetchall() or []
            users = [r["username"] for r in rows]
            for r in rows:
                print(f"- {_mask(r['username'])} tier={r['tier']} -> basic, "
                      f"customer ...{r['stripe_customer_id'][-6:]} cleared")
            if not users:
                print("nothing to clean up (already done?)")
                conn.rollback()
                return 0
            cur.execute("UPDATE users SET tier = 'basic', stripe_customer_id = NULL, "
                        "stripe_subscription_id = NULL, stripe_price_id = NULL, plan_updated_at = NOW() "
                        "WHERE username = ANY(%s)", (users,))
            updated = cur.rowcount
            cur.execute("DELETE FROM hsf_billing_state WHERE username = ANY(%s)", (users,))
            deleted = cur.rowcount
            if updated != len(users) or deleted != len(users) or len(users) > len(TEST_CUSTOMERS):
                conn.rollback()
                print(f"unexpected row counts (users {updated}, state {deleted}, matched {len(users)}); rolled back")
                return 1
            cur.execute("SELECT COUNT(*) AS n FROM hsf_billing_state WHERE last_sync_error IS NOT NULL")
            remaining = cur.fetchone()["n"]
        conn.commit()
        print(f"cleaned {len(users)} account(s); accounts with sync errors now: {remaining}")
        return 0
    except Exception as exc:  # never echo connection details
        conn.rollback()
        print(f"failed, rolled back: {type(exc).__name__}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
