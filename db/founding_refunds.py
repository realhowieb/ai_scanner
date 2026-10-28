"""Founding 100 refunds: payments, refund requests, admin decisions and the audit log
(rules in domain/founding_refunds.py, policy in legal/founding_refunds.v1.txt).

Life of a request:
  requested -> approved -> initiated (Stripe refund created) -> completed (Stripe's
  charge.refunded event) ; or requested -> denied ; or approved/initiated -> failed.

* Members can always ask, for each of their payments; outside the 14 days the request
  is marked for review, never refused. One open request per payment.
* Only a confirmed refund event from the payment provider changes access (via
  db/founding.refund): a request or an approval never does.
* Refunded amounts accumulate on the payment and can never exceed what was paid.
* Every step is written to founding_audit_log (insert-only).

Shares db/founding.py's connections, transaction and advisory lock, so refunds and
seat changes never interleave.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from db import founding as db
from domain import founding_refunds as rules


class RefundError(db.FoundingError):
    code = "refund_error"


class AlreadyRequested(RefundError):
    code = "refund_already_requested"


class NothingToRefund(RefundError):
    code = "nothing_to_refund"


class BadState(RefundError):
    code = "refund_wrong_state"


def _audit(tx, actor: str, action: str, *, request_id: Optional[int] = None, ref: Optional[str] = None,
           now: dt.datetime, **detail: Any) -> None:
    tx.run("INSERT INTO founding_audit_log (at, actor, action, refund_request_id, payment_reference, detail) "
           "VALUES (%s, %s, %s, %s, %s, %s)",
           (db._iso(now), actor, action, request_id, ref, json.dumps(detail, sort_keys=True, default=str)))


def _payment_view(p: Dict[str, Any], now: dt.datetime) -> Dict[str, Any]:
    paid_at = db.parse_ts(p["paid_at"])
    deadline = rules.guarantee_deadline(paid_at)
    return {**p, "deadline": db._iso(deadline), "within_guarantee": now <= deadline,
            "refundable_cents": rules.refundable(p["amount_cents"], p["refunded_cents"])}


def payments(user_id: str, now: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
    """An account's founding payments, oldest first, with refund deadline and what's left."""
    now = db._utc(now)
    with db._transaction(write=False) as tx:
        rows = tx.rows("SELECT * FROM founding_payments WHERE user_id = %s ORDER BY id", (user_id.strip().lower(),))
    return [_payment_view(p, now) for p in rows]


def requests_for(user_id: str) -> List[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT r.*, p.kind, p.payment_reference, p.amount_cents AS paid_cents FROM founding_refund_requests r "
                       "JOIN founding_payments p ON p.id = r.payment_id WHERE r.user_id = %s ORDER BY r.id",
                       (user_id.strip().lower(),))


def all_requests(status: Optional[str] = None, limit: int = 200, *,
                 request_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """For admins: newest first, with the payment and membership facts needed to decide."""
    where, params = "", []
    if status:
        where, params = "WHERE r.status = %s", [status]
    if request_id is not None:
        where, params = "WHERE r.id = %s", [int(request_id)]
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT r.*, p.kind, p.tier, p.payment_reference, p.amount_cents AS paid_cents, p.refunded_cents, "
                       "p.paid_at, m.membership_number, m.former_number, m.status AS membership_status, m.activated_at, "
                       "m.expires_at, m.upgraded_at FROM founding_refund_requests r "
                       "JOIN founding_payments p ON p.id = r.payment_id "
                       "JOIN founding_memberships m ON m.id = p.membership_id "
                       f"{where} ORDER BY r.id DESC LIMIT %s", tuple(params + [int(limit)]))


def get_request(request_id: int) -> Optional[Dict[str, Any]]:
    rows = all_requests(request_id=request_id, limit=1)
    return rows[0] if rows else None


def history(limit: int = 200) -> List[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT * FROM founding_audit_log ORDER BY id DESC LIMIT %s", (int(limit),))


def request_refund(user_id: str, kind: str, *, reason: str = "", now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """A member asks for a refund of their purchase or upgrade payment. Never refused for
    being late: within_guarantee just records whether the 14 days covered it."""
    user_id, now = user_id.strip().lower(), db._utc(now)
    if kind not in rules.PAYMENT_KINDS:
        raise RefundError("Choose the purchase or the upgrade payment.")
    with db._transaction() as tx:
        p = tx.one("SELECT * FROM founding_payments WHERE user_id = %s AND kind = %s ORDER BY id DESC LIMIT 1",
                   (user_id, kind))
        if not p:
            raise NothingToRefund("There is no founding payment of that kind on this account.")
        if rules.refundable(p["amount_cents"], p["refunded_cents"]) <= 0:
            raise NothingToRefund("This payment has already been refunded in full.")
        if tx.one("SELECT id FROM founding_refund_requests WHERE payment_id = %s AND status IN ('requested', 'approved', "
                  "'initiated')", (p["id"],)):
            raise AlreadyRequested("A refund request for this payment is already open.")
        inside = rules.within_guarantee(db.parse_ts(p["paid_at"]), now)
        tx.run("INSERT INTO founding_refund_requests (payment_id, user_id, status, within_guarantee, member_reason, "
               "requested_at, due_at, updated_at) VALUES (%s, %s, 'requested', %s, %s, %s, %s, %s)",
               (p["id"], user_id, 1 if inside else 0, (reason or "").strip()[:1000] or None, db._iso(now),
                db._iso(rules.start_due(now)), db._iso(now)))
        r = tx.one("SELECT * FROM founding_refund_requests WHERE payment_id = %s AND status = 'requested'", (p["id"],))
        _audit(tx, user_id, "refund_requested", request_id=r["id"], ref=p["payment_reference"], now=now,
               within_guarantee=inside, kind=kind)
        return {**r, "kind": kind, "payment_reference": p["payment_reference"], "paid_cents": p["amount_cents"]}


def prorated_quote(request_id: int, stopped_at: dt.datetime) -> Dict[str, Any]:
    """The discontinuation refund for a request's payment if service stopped at `stopped_at`."""
    r = get_request(request_id)
    if not r:
        raise db.NotFound("No such refund request.")
    end = db.parse_ts(r["expires_at"])
    start = db.parse_ts(r["paid_at"]) if r["kind"] == "upgrade" else db.parse_ts(r["activated_at"])
    if start is None or end is None:
        raise BadState("This membership has no start and end to prorate over.")
    amount = rules.prorated_cents(r["paid_cents"], start, end, stopped_at, r["refunded_cents"])
    return {"request_id": int(request_id), "kind": r["kind"], "paid_cents": r["paid_cents"],
            "refunded_cents": r["refunded_cents"], "start": db._iso(start), "end": db._iso(end),
            "stopped_at": db._iso(db._utc(stopped_at)), "unused_months": str(rules.months_between(
                min(max(db._utc(stopped_at), start), end), end).quantize(rules.Decimal("0.0001"))),
            "total_months": str(rules.months_between(start, end).quantize(rules.Decimal("0.0001"))),
            "amount_cents": amount}


def approve(request_id: int, admin: str, *, basis: str, amount_cents: Optional[int] = None, reason: str = "",
            stopped_at: Optional[dt.datetime] = None, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """An admin approves a request. The amount defaults to everything still refundable
    (or, for discontinuation, the prorated amount at `stopped_at`) and can never exceed it."""
    now = db._utc(now)
    if basis not in rules.BASES:
        raise RefundError("Choose why this refund is approved.")
    prorated = None
    if basis == "discontinuation":
        if stopped_at is None:
            raise RefundError("A discontinuation refund needs the date service stopped.")
        prorated = prorated_quote(request_id, stopped_at)["amount_cents"]
    with db._transaction() as tx:
        r = tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (int(request_id),))
        if not r:
            raise db.NotFound("No such refund request.")
        if r["status"] == "approved" and r.get("amount_cents"):
            return r  # approving twice (e.g. retrying the provider call) changes nothing
        if r["status"] != "requested":
            raise BadState(f"This request is already {r['status']}.")
        p = tx.one("SELECT * FROM founding_payments WHERE id = %s", (r["payment_id"],))
        if basis == "guarantee" and not int(r["within_guarantee"]):
            raise RefundError("This request came after the 14 days; choose the exception that applies.")
        left = rules.refundable(p["amount_cents"], p["refunded_cents"])
        amount = int(amount_cents) if amount_cents is not None else rules.default_amount(
            basis, p["amount_cents"], p["refunded_cents"], prorated=prorated)
        if amount <= 0 or amount > left:
            raise RefundError(f"The amount must be between $0.01 and ${left / 100:,.2f}.")
        tx.run("UPDATE founding_refund_requests SET status = 'approved', basis = %s, amount_cents = %s, "
               "decision_reason = %s, decided_by = %s, decided_at = %s, updated_at = %s WHERE id = %s",
               (basis, amount, (reason or "").strip()[:1000] or None, admin, db._iso(now), db._iso(now), r["id"]))
        _audit(tx, admin, "refund_approved", request_id=r["id"], ref=p["payment_reference"], now=now,
               basis=basis, amount_cents=amount, reason=reason, stopped_at=stopped_at)
        return tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (r["id"],))


def deny(request_id: int, admin: str, *, reason: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = db._utc(now)
    if not (reason or "").strip():
        raise RefundError("Give the member a reason.")
    with db._transaction() as tx:
        r = tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (int(request_id),))
        if not r:
            raise db.NotFound("No such refund request.")
        if r["status"] != "requested":
            raise BadState(f"This request is already {r['status']}.")
        tx.run("UPDATE founding_refund_requests SET status = 'denied', decision_reason = %s, decided_by = %s, "
               "decided_at = %s, updated_at = %s WHERE id = %s",
               (reason.strip()[:1000], admin, db._iso(now), db._iso(now), r["id"]))
        _audit(tx, admin, "refund_denied", request_id=r["id"], now=now, reason=reason)
        return tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (r["id"],))


def mark_initiated(request_id: int, provider_refund_id: str, actor: str, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = db._utc(now)
    with db._transaction() as tx:
        r = tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (int(request_id),))
        if not r:
            raise db.NotFound("No such refund request.")
        if r["status"] in ("initiated", "completed") and r.get("provider_refund_id") == provider_refund_id:
            return r
        if r["status"] != "approved":
            raise BadState(f"This request is {r['status']}, not approved.")
        tx.run("UPDATE founding_refund_requests SET status = 'initiated', provider_refund_id = %s, initiated_at = %s, "
               "failure_reason = NULL, updated_at = %s WHERE id = %s",
               (provider_refund_id, db._iso(now), db._iso(now), r["id"]))
        _audit(tx, actor, "refund_initiated", request_id=r["id"], now=now, provider_refund_id=provider_refund_id)
        return tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (r["id"],))


def mark_failed(request_id: int, reason: str, actor: str, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """The provider refused or reversed the refund; an admin can approve a new request."""
    now = db._utc(now)
    with db._transaction() as tx:
        r = tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (int(request_id),))
        if not r:
            raise db.NotFound("No such refund request.")
        if r["status"] not in ("approved", "initiated"):
            return r
        tx.run("UPDATE founding_refund_requests SET status = 'failed', failure_reason = %s, updated_at = %s WHERE id = %s",
               (reason[:300], db._iso(now), r["id"]))
        _audit(tx, actor, "refund_failed", request_id=r["id"], now=now, reason=reason)
        return tx.one("SELECT * FROM founding_refund_requests WHERE id = %s", (r["id"],))


def note_provider_error(request_id: int, reason: str, actor: str, *, now: Optional[dt.datetime] = None) -> None:
    """A provider call failed before any refund existed: stays approved, so it can be retried."""
    now = db._utc(now)
    with db._transaction() as tx:
        tx.run("UPDATE founding_refund_requests SET failure_reason = %s, updated_at = %s WHERE id = %s",
               (reason[:300], db._iso(now), int(request_id)))
        _audit(tx, actor, "refund_provider_error", request_id=int(request_id), now=now, reason=reason)


def apply_provider_refund(payment_reference: str, *, refunded_total_cents: int, provider_refund_id: Optional[str],
                          now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """A confirmed refund from the payment provider (Stripe charge.refunded, which reports
    the payment's total refunded so far). Idempotent: the same total twice changes nothing.

    Records the total on the payment, completes the matching request (or records a refund
    made directly in Stripe), and when the payment is now fully refunded ends the
    membership or, for the upgrade payment, returns the member to Founder Pro. A repeated
    event re-checks that last step, so a crash between the two can't leave access behind.
    Returns {"outcome": ..., "full": bool, "access": ...} or {"outcome": "unknown_payment"}."""
    now = db._utc(now)
    ref = str(payment_reference or "").strip()
    out: Dict[str, Any]
    with db._transaction() as tx:
        p = tx.one("SELECT * FROM founding_payments WHERE payment_reference = %s", (ref,))
        if not p:
            return {"outcome": "unknown_payment"}
        total = min(int(refunded_total_cents), int(p["amount_cents"]))
        if total <= int(p["refunded_cents"]):
            out = {"outcome": "already_done", "full": int(p["refunded_cents"]) >= int(p["amount_cents"]),
                   "user_id": p["user_id"], "kind": p["kind"], "request_id": None}
        else:
            tx.run("UPDATE founding_payments SET refunded_cents = %s, updated_at = %s WHERE id = %s",
                   (total, db._iso(now), p["id"]))
            r = tx.one("SELECT * FROM founding_refund_requests WHERE payment_id = %s AND status IN ('approved', "
                       "'initiated') ORDER BY id DESC LIMIT 1", (p["id"],))
            if r:
                tx.run("UPDATE founding_refund_requests SET status = 'completed', completed_at = %s, updated_at = %s, "
                       "provider_refund_id = COALESCE(provider_refund_id, %s) WHERE id = %s",
                       (db._iso(now), db._iso(now), provider_refund_id, r["id"]))
            _audit(tx, "payment_provider", "refund_completed", request_id=r["id"] if r else None, ref=ref, now=now,
                   refunded_total_cents=total, provider_refund_id=provider_refund_id, matched_request=bool(r))
            out = {"outcome": "completed", "request_id": int(r["id"]) if r else None,
                   "full": total >= int(p["amount_cents"]), "user_id": p["user_id"], "kind": p["kind"],
                   "refunded_cents": total}
    if out["full"]:
        out["access"] = _end_access(ref, now)
    return out


def _end_access(ref: str, now: dt.datetime) -> str:
    try:
        return db.refund(ref, now=now)["outcome"]
    except db.NotFound:
        return "already_done"  # an upgrade already reversed no longer carries this reference


def payment_by_reference(payment_reference: str) -> Optional[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.one("SELECT * FROM founding_payments WHERE payment_reference = %s", (str(payment_reference),))


def accepted_policies(payment_reference: str) -> List[Dict[str, Any]]:
    """The policy versions accepted at the checkout that made this payment."""
    with db._transaction(write=False) as tx:
        res = tx.one("SELECT checkout_session_id FROM founding_reservations WHERE payment_reference = %s",
                     (str(payment_reference),))
    if not res or not res.get("checkout_session_id"):
        return []
    from db import legal

    return legal.acceptances(checkout_session_id=res["checkout_session_id"])


def audit_for(request_id: int) -> List[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT * FROM founding_audit_log WHERE refund_request_id = %s ORDER BY id", (int(request_id),))
