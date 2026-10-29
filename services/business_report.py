"""Monday business email to the admins (api/ops_jobs.py, from 12:00 UTC).

Last 7 days, from the database only (no Stripe call):
    sign-ups, verified, took a watchlist or alert step, free Pro trials  (db.admin_analytics.load_signup_funnel)
    sign-ups by source                                                   (same)
    paying accounts now, new this week, cancellations scheduled, ended   (users + hsf_billing_state)
    monthly revenue now: active/past-due subscriptions' prices, yearly / 12 (hsf_billing_state snapshots)
    tips, come-back, trial-ended and billing emails sent                 (hsf_lifecycle_emails)

Sent with services.ops_notify (chat + email). Env: HSF_BUSINESS_EMAIL_ENABLED=0 stops it.
"""
from __future__ import annotations

import datetime as dt
import html
import json
from collections import Counter
from typing import Any, Dict, List, Optional

UTC = dt.timezone.utc
PAID_STATUSES = ("active", "trialing", "past_due")


def _rows(conn, sql: str, args: tuple = ()) -> List[tuple]:
    cur = conn.cursor()
    try:
        cur.execute(sql, args)
        rows = cur.fetchall() or []
    finally:
        cur.close()
    return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]


def revenue(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Monthly revenue (in the currency's main unit) from billing snapshots. Pure."""
    by_cur: Counter = Counter()
    paying = 0
    test = False
    for s in snapshots:
        if not isinstance(s, dict) or s.get("status") not in PAID_STATUSES or s.get("plan") not in ("pro", "premium"):
            continue
        paying += 1
        test = test or bool(s.get("test_mode"))
        amount = s.get("amount")
        if amount is None:
            continue
        monthly = float(amount) / (12.0 if s.get("interval") == "year" else 1.0) / 100.0
        by_cur[str(s.get("currency") or "usd").upper()] += monthly
    return {"paying": paying, "mrr": dict(by_cur), "test_mode": test}


def gather(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    from db.admin_analytics import load_signup_funnel
    from db.engine import get_neon_conn

    now = now or dt.datetime.now(UTC)
    since = now - dt.timedelta(days=7)
    out: Dict[str, Any] = {"errors": []}
    try:
        f = load_signup_funnel(days=7)
        acc = f.get("accounts") or []
        out["signups"] = len(acc)
        out["verified"] = sum(1 for a in acc if a.get("verified"))
        out["engaged"] = sum(1 for a in acc if a.get("engaged"))
        out["trials"] = sum(1 for a in acc if a.get("trial"))
        out["paid_from_signups"] = sum(1 for a in acc if a.get("has_subscription"))
        out["sources"] = Counter(v or "direct" for v in (f.get("sources") or {}).values()).most_common(5)
    except Exception as e:
        out["errors"].append(f"funnel: {type(e).__name__}")
    try:
        from db import founding_refunds

        out["founders"] = founding_refunds.weekly_summary(since)
    except Exception as e:
        out["errors"].append(f"founders: {type(e).__name__}")
    conn = get_neon_conn()
    if conn is None:
        out["errors"].append("database unavailable")
        return out
    try:
        try:
            snaps = []
            for (snap,) in _rows(conn, "SELECT snapshot FROM hsf_billing_state WHERE snapshot IS NOT NULL"):
                snaps.append(json.loads(snap) if isinstance(snap, str) else snap)
            out["revenue"] = revenue(snaps)
            out["cancel_scheduled"] = sum(1 for s in snaps if isinstance(s, dict) and s.get("cancel_at_period_end")
                                          and s.get("status") in PAID_STATUSES)
            out["past_due"] = sum(1 for s in snaps if isinstance(s, dict) and s.get("status") == "past_due")
        except Exception as e:
            conn.rollback()
            out["errors"].append(f"billing: {type(e).__name__}")
        try:
            out["new_paid"] = _rows(conn, "SELECT COUNT(*) FROM users WHERE lower(tier) IN ('pro','premium') "
                                          "AND stripe_subscription_id IS NOT NULL AND plan_updated_at >= %s",
                                    (since,))[0][0]
        except Exception as e:
            conn.rollback()
            out["errors"].append(f"users: {type(e).__name__}")
        try:
            sent = _rows(conn, "SELECT kind, COUNT(*) FROM hsf_lifecycle_emails WHERE sent_at >= %s "
                               "AND user_id <> '__ops__' GROUP BY kind", (since.isoformat(timespec="seconds"),))
            out["emails"] = {str(k): int(n) for k, n in sent}
        except Exception as e:
            conn.rollback()
            out["errors"].append(f"emails: {type(e).__name__}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def _founder_lines(f: Optional[Dict[str, Any]]) -> list:
    """Founding 100 lines, only once someone has paid (one-time payments, not MRR)."""
    if not f or not (f.get("active") or f.get("payments") or f.get("open_refund_requests")):
        return []
    active = f.get("active") or {}
    line = (f"Founders: {active.get('founder_pro', 0)} Founder Pro, {active.get('founder_elite', 0)} Founder Elite"
            f" · {f.get('payments', 0)} payments this week ${f.get('paid_cents', 0) / 100:,.2f}")
    if f.get("refunds"):
        line += f" · refunded ${f.get('refunded_cents', 0) / 100:,.2f}"
    if f.get("open_refund_requests"):
        line += f" · {f['open_refund_requests']} refund request(s) waiting"
    return [line]


def compose(d: Dict[str, Any], now: dt.datetime) -> tuple:
    """(subject, text, html inner). Pure."""
    rev = d.get("revenue") or {}
    mrr = ", ".join(f"{cur} {v:,.2f}" for cur, v in sorted((rev.get("mrr") or {}).items())) or "0"
    test = " (Stripe test mode)" if rev.get("test_mode") else ""
    subject = f"📈 HSF week: {d.get('signups', '?')} sign-ups, {rev.get('paying', '?')} paying, {mrr}/mo{test}"
    emails = d.get("emails") or {}
    lines = [
        f"Sign-ups: {d.get('signups', '?')} (verified {d.get('verified', '?')}, saved a watchlist or alert "
        f"{d.get('engaged', '?')}, free Pro trials {d.get('trials', '?')})",
        "Sources: " + (", ".join(f"{s} {n}" for s, n in d.get("sources") or []) or "none recorded"),
        f"Paying now: {rev.get('paying', '?')} · new this week {d.get('new_paid', '?')} · "
        f"cancellation scheduled {d.get('cancel_scheduled', '?')} · payment overdue {d.get('past_due', '?')}",
        f"Monthly revenue now: {mrr}{test}",
        *_founder_lines(d.get("founders")),
        "Account emails sent: " + (", ".join(f"{k} {n}" for k, n in sorted(emails.items())) or "none"),
    ]
    if d.get("errors"):
        lines.append("Couldn't load: " + "; ".join(d["errors"]))
    text = "\n".join(lines)
    inner = ("<h2 style='font-size:18px;margin:16px 0 8px'>HSF week ending "
             f"{html.escape(f'{now:%b} {now.day}')}</h2><ul style='padding-left:20px'>"
             + "".join(f"<li>{html.escape(x)}</li>" for x in lines) + "</ul>")
    return subject, text, inner


def run(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    from services.ops_notify import notify_ops

    now = now or dt.datetime.now(UTC)
    subject, text, inner = compose(gather(now), now)
    return notify_ops(subject, text, inner_html=inner, email=True)
