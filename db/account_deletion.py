"""Delete a customer account and its personal data (P2-82; App Store requirement).

Shared by the web (Settings) and the API (DELETE /v1/me). Rules:
  * admins can't delete themselves here;
  * an account on a paid plan with a Stripe subscription must cancel first
    (Billing > Manage subscription); it can be deleted once the subscription has
    ended and the billing webhook has moved it to Free. Stripe keeps its own
    billing records (invoices), which this never touches;
  * everything tied to the account is removed in one transaction: watchlists,
    alerts and alert history, journal, paper-trading keys and orders, settings
    and email preferences, saved scans, AI usage, sessions and sign-in records,
    verification/reset tokens, API refresh tokens, push devices and API scan jobs,
    then the user row. Research and market tables hold no account data and are
    untouched; acquisition_events only has a one-way hash and is kept.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# (table, account column). Child rows first; the users row is deleted last.
ACCOUNT_TABLES: Tuple[Tuple[str, str], ...] = (
    ("user_alerts", "user_id"), ("alert_events", "user_id"), ("alert_outcomes", "user_id"),
    ("user_trades", "user_id"), ("alpaca_paper_accounts", "user_id"), ("alpaca_paper_order_events", "user_id"),
    ("user_settings", "user_id"), ("hsf_email_prefs", "user_id"), ("hsf_alert_prefs", "user_id"),
    ("hsf_intelligence_alerts", "user_id"),
    ("email_verifications", "username"), ("password_reset_tokens", "username"), ("ai_usage", "username"),
    ("login_attempts", "username"), ("runs", "username"), ("scan_errors", "username"),
    ("auth_sessions", "username"), ("hsf_auth_tokens", "username"),
    ("api_refresh_tokens", "username"), ("api_push_devices", "username"), ("api_scan_jobs", "username"),
)
PAID = ("pro", "premium")


class DeletionBlocked(RuntimeError):
    """The account can't be deleted yet (message is safe to show)."""


def blocker(user: Dict[str, object]) -> Optional[str]:
    """Why this account can't be deleted now, or None."""
    if user.get("is_admin") or str(user.get("tier") or "").lower() == "admin":
        return "Admin accounts can't be deleted here."
    if str(user.get("tier") or "").lower() in PAID and user.get("stripe_subscription_id"):
        return ("Cancel your subscription first (Billing > Manage subscription). You can delete your "
                "account once it has ended.")
    return None


def delete_account(conn, username: str) -> Dict[str, int]:
    """Delete `username` and its data in one transaction; returns rows deleted per table.
    Raises DeletionBlocked (nothing changed) when blocker() applies or the account is gone."""
    from psycopg import sql

    user = (username or "").strip().lower()
    cur = conn.cursor()
    try:
        cur.execute("SELECT username, tier, is_admin, stripe_subscription_id FROM users "
                    "WHERE lower(username) = %s FOR UPDATE", (user,))
        row = cur.fetchone()
        if row is None:
            raise DeletionBlocked("Account not found.")
        cols = [d.name for d in cur.description]
        info = dict(row) if isinstance(row, dict) else dict(zip(cols, row))
        why = blocker(info)
        if why:
            raise DeletionBlocked(why)

        counts: Dict[str, int] = {}
        existing = _existing_tables(cur, ["watchlists", "watchlist_items", *[t for t, _ in ACCOUNT_TABLES]])
        if "watchlists" in existing:
            if "watchlist_items" in existing:
                cur.execute("DELETE FROM watchlist_items WHERE watchlist_id IN "
                            "(SELECT id FROM watchlists WHERE lower(user_id) = %s)", (user,))
                counts["watchlist_items"] = cur.rowcount or 0
            cur.execute("DELETE FROM watchlists WHERE lower(user_id) = %s", (user,))
            counts["watchlists"] = cur.rowcount or 0
        for table, column in ACCOUNT_TABLES:
            if table not in existing:
                continue
            cur.execute(sql.SQL("DELETE FROM {} WHERE lower({}) = %s").format(sql.Identifier(table),
                                                                              sql.Identifier(column)), (user,))
            counts[table] = cur.rowcount or 0
        cur.execute("DELETE FROM users WHERE lower(username) = %s", (user,))
        counts["users"] = cur.rowcount or 0
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _existing_tables(cur, names: List[str]) -> set:
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() "
                "AND table_name = ANY(%s)", (list(names),))
    out = set()
    for r in cur.fetchall():
        out.add(r["table_name"] if isinstance(r, dict) else r[0])
    return out
