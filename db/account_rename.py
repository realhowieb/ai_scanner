"""Change an account's email (its username) across every table (replaces P2-36).

Usernames double as email addresses, and about 30 tables store the username as
their account key. Changing someone's email used to mean hand-written SQL in all
of them (P2-28). This does it in one transaction:

  * preview_rename() counts the rows each table holds for the old address and
    lists anything that blocks the change, without writing;
  * rename_account() moves every row to the new address, signs the account out
    everywhere (sessions, refresh tokens and sign-in/reset/verification links
    issued to the old address are deleted) and marks the new address unverified
    unless the admin keeps it verified.

Tables whose account column has a foreign key to users(username) are found from
the database itself and moved through a temporary copy, since that key has no
ON UPDATE CASCADE. Audit logs (hsf_admin_events, founding_audit_log) keep the
address they recorded, and research tables (signal_outcomes) stay frozen.
acquisition_events only holds a one-way hash and is kept; hsf_referrals.referred
holds the hash of the invited address, so it is re-hashed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# (table, account column) whose rows move to the new address.
RENAME_TABLES: Tuple[Tuple[str, str], ...] = (
    ("watchlists", "user_id"), ("hsf_alert_rules", "user_id"),
    ("user_alerts", "user_id"), ("alert_events", "user_id"), ("alert_outcomes", "user_id"),
    ("user_trades", "user_id"), ("alpaca_paper_accounts", "user_id"), ("alpaca_paper_order_events", "user_id"),
    ("paper_order_requests", "user_id"),
    ("user_settings", "user_id"), ("hsf_email_prefs", "user_id"), ("hsf_alert_prefs", "user_id"),
    ("hsf_intelligence_alerts", "user_id"), ("hsf_alert_rule_events", "user_id"),
    ("ai_usage", "username"), ("login_attempts", "username"), ("runs", "username"), ("scan_errors", "username"),
    ("hsf_billing_state", "username"),
    ("hsf_workspace_preferences", "user_id"), ("hsf_push_deliveries", "user_id"),
    ("hsf_chat_destinations", "user_id"), ("hsf_chat_link_codes", "user_id"),
    ("api_push_devices", "username"), ("api_scan_jobs", "username"), ("hsf_scheduled_scans", "username"),
    ("hsf_premium_auto", "username"),
    ("hsf_referral_codes", "user_id"), ("hsf_plan_grants", "user_id"), ("hsf_referrals", "referrer"),
    ("hsf_lifecycle_emails", "user_id"), ("user_policy_acceptances", "user_id"),
    ("hsf_web_state", "user_id"), ("hsf_feedback", "user_id"), ("hsf_user_activity", "user_id"),
    ("founding_memberships", "user_id"), ("founding_reservations", "user_id"), ("founding_payments", "user_id"),
    ("founding_refund_requests", "user_id"), ("founding_member_prefs", "user_id"), ("founding_wall", "user_id"),
)

# Sign-in state and one-time links tied to the old address: deleted, not moved.
CLEAR_TABLES: Tuple[Tuple[str, str], ...] = (
    ("auth_sessions", "username"), ("hsf_auth_tokens", "username"), ("api_refresh_tokens", "username"),
    ("sign_in_link_tokens", "username"), ("password_reset_tokens", "username"), ("email_verifications", "username"),
)

# Tables with an account-looking column that deliberately stay as they are.
KEEP_TABLES = frozenset({
    "hsf_admin_events", "founding_audit_log",   # audit history keeps what it recorded
    "acquisition_events",                       # one-way hash only
    "hsf_marketing_subscribers",                # separate opt-in list, not tied to an account
    "signal_outcomes",                          # immutable research rows (fired signals)
})


class RenameBlocked(RuntimeError):
    """The change can't be made (message is safe to show the admin)."""


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _existing_tables(cur, names: List[str]) -> set:
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() "
                "AND table_name = ANY(%s)", (list(names),))
    return {r["table_name"] if isinstance(r, dict) else r[0] for r in cur.fetchall()}


def _fk_tables(cur) -> set:
    """(table, column) pairs whose column has a foreign key to users(username)."""
    cur.execute(
        """
        SELECT cl.relname AS t, att.attname AS c
        FROM pg_constraint con
        JOIN pg_class cl ON cl.oid = con.conrelid
        JOIN pg_class ref ON ref.oid = con.confrelid
        JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = ANY(con.conkey)
        JOIN pg_attribute refatt ON refatt.attrelid = con.confrelid AND refatt.attnum = ANY(con.confkey)
        WHERE con.contype = 'f' AND ref.relname = 'users' AND refatt.attname = 'username'
          AND cl.relnamespace = to_regnamespace(current_schema())
        """
    )
    return {((r["t"], r["c"]) if isinstance(r, dict) else (r[0], r[1])) for r in cur.fetchall()}


def _count(cur, table: str, column: str, value: str) -> int:
    from psycopg import sql

    cur.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE lower({}) = %s").format(
        sql.Identifier(table), sql.Identifier(column)), (value,))
    row = cur.fetchone()
    return int((list(row.values())[0] if isinstance(row, dict) else row[0]) or 0)


def _user(cur, username: str, lock: bool = False) -> Optional[Dict[str, Any]]:
    if lock:
        cur.execute("SELECT username, tier, is_admin, stripe_customer_id, COALESCE(email_verified, FALSE) AS email_verified "
                    "FROM users WHERE lower(username) = %s FOR UPDATE", (username,))
    else:
        cur.execute("SELECT username, tier, is_admin, stripe_customer_id, COALESCE(email_verified, FALSE) AS email_verified "
                    "FROM users WHERE lower(username) = %s", (username,))
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d.name for d in cur.description]
    return dict(row) if isinstance(row, dict) else dict(zip(cols, row))


def _check(cur, old: str, new: str, lock: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """The account row and every reason the change can't go ahead."""
    from db.user_admin import is_valid_email

    problems: List[str] = []
    if not old:
        raise RenameBlocked("Choose an account.")
    if not is_valid_email(new):
        raise RenameBlocked("The new email must be a valid address (it's where the account's emails go).")
    if new == old:
        raise RenameBlocked("The new email is the same as the current one.")
    user = _user(cur, old, lock=lock)
    if user is None:
        raise RenameBlocked("Account not found.")
    if _user(cur, new) is not None:
        problems.append(f"An account for {new} already exists.")
    existing = _existing_tables(cur, [t for t, _ in RENAME_TABLES])
    leftovers = [t for t, c in RENAME_TABLES if t in existing and _count(cur, t, c, new)]
    if leftovers:
        problems.append(f"Old rows for {new} are still stored in: {', '.join(leftovers)}. Remove them first.")
    return user, problems


def preview_rename(conn, old_email: Any, new_email: Any) -> Dict[str, Any]:
    """Rows per table that would move or be cleared, and any blockers. Writes nothing."""
    old, new = _norm(old_email), _norm(new_email)
    cur = conn.cursor()
    try:
        user, problems = _check(cur, old, new)
        tables = [t for t, _ in RENAME_TABLES + CLEAR_TABLES]
        existing = _existing_tables(cur, tables)
        move = {t: n for t, c in RENAME_TABLES if t in existing and (n := _count(cur, t, c, old))}
        clear = {t: n for t, c in CLEAR_TABLES if t in existing and (n := _count(cur, t, c, old))}
        return {"old_email": old, "new_email": new, "move": move, "clear": clear, "problems": problems,
                "stripe_customer": bool(user.get("stripe_customer_id")),
                "email_verified": bool(user.get("email_verified"))}
    finally:
        conn.rollback()
        cur.close()


def rename_account(conn, old_email: Any, new_email: Any, *, keep_verified: bool = False) -> Dict[str, Any]:
    """Move the account to `new_email` in one transaction; returns rows changed per table.
    Raises RenameBlocked (nothing changed) when a check fails."""
    from psycopg import sql

    from db.referrals import referred_key

    old, new = _norm(old_email), _norm(new_email)
    cur = conn.cursor()
    try:
        user, problems = _check(cur, old, new, lock=True)
        if problems:
            raise RenameBlocked(" ".join(problems))
        stored = str(user["username"])
        existing = _existing_tables(cur, [t for t, _ in RENAME_TABLES + CLEAR_TABLES] + ["hsf_referrals"])
        fks = {(t, c) for t, c in _fk_tables(cur)}
        moved: Dict[str, int] = {}
        cleared: Dict[str, int] = {}

        for table, column in CLEAR_TABLES:
            if table in existing and (table, column) not in fks:
                cur.execute(sql.SQL("DELETE FROM {} WHERE lower({}) = %s").format(
                    sql.Identifier(table), sql.Identifier(column)), (old,))
                if cur.rowcount:
                    cleared[table] = cur.rowcount

        # Rows behind a foreign key to users(username): park them, move the user, put them back.
        parked: List[Tuple[str, str, str]] = []
        for i, (table, column) in enumerate(sorted(fks)):
            temp = f"hsf_rename_park_{i}"
            cur.execute(sql.SQL("CREATE TEMP TABLE {} ON COMMIT DROP AS SELECT * FROM {} WHERE lower({}) = %s").format(
                sql.Identifier(temp), sql.Identifier(table), sql.Identifier(column)), (old,))
            cur.execute(sql.SQL("DELETE FROM {} WHERE lower({}) = %s").format(
                sql.Identifier(table), sql.Identifier(column)), (old,))
            parked.append((table, column, temp))

        cur.execute("UPDATE users SET username = %s, email_verified = (email_verified AND %s) WHERE username = %s",
                    (new, bool(keep_verified), stored))
        moved["users"] = cur.rowcount or 0

        for table, column, temp in parked:
            cur.execute(sql.SQL("UPDATE {} SET {} = %s").format(sql.Identifier(temp), sql.Identifier(column)), (new,))
            cur.execute(sql.SQL("INSERT INTO {} SELECT * FROM {}").format(sql.Identifier(table), sql.Identifier(temp)))
            if cur.rowcount:
                moved[table] = cur.rowcount

        for table, column in RENAME_TABLES:
            if table not in existing or (table, column) in fks:
                continue
            cur.execute(sql.SQL("UPDATE {} SET {} = %s WHERE lower({}) = %s").format(
                sql.Identifier(table), sql.Identifier(column), sql.Identifier(column)), (new, old))
            if cur.rowcount:
                moved[table] = moved.get(table, 0) + cur.rowcount

        if "hsf_referrals" in existing:   # the invited side is stored as a hash of the address
            cur.execute("UPDATE hsf_referrals SET referred = %s WHERE referred = %s", (referred_key(new), referred_key(old)))
            if cur.rowcount:
                moved["hsf_referrals (invited)"] = cur.rowcount
        conn.commit()
        return {"old_email": old, "new_email": new, "moved": moved, "cleared": cleared,
                "stripe_customer_id": user.get("stripe_customer_id")}
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
