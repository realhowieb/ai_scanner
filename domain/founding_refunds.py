"""Founding 100 refund rules (legal/founding_refunds.v1.txt). Pure: no database or Stripe.

* 14-day guarantee: a full refund of a payment requested within 14 days (14 x 24 hours)
  of the time that payment was confirmed, no reason needed. The Pro-to-Elite upgrade
  payment has its own 14 days.
* Later requests are always accepted and go to a person (never refused automatically).
* Approved refunds are started within seven business days of the request (Mon-Fri; the
  due date shown to admins, not a promise about when the bank shows the money).
* Permanent discontinuation: amount paid x unused time / total time for that payment.
  The original payment covers activation -> end; the upgrade covers upgrade -> the
  original end. Time is measured in calendar months from the actual timestamps, a
  part-month counting as its share of that month (so 24 of 60 months used on $999
  refunds exactly $599.40, as the policy's example says). Exact decimal maths, rounded
  half-up to the cent, never more than what is left of that payment.
"""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

GUARANTEE_DAYS = 14
START_WITHIN_BUSINESS_DAYS = 7

# Why a refund is approved. "guarantee" only inside the 14 days; the others are the
# policy's exceptions (always a person's decision).
BASES = ("guarantee", "legal_requirement", "billing_error", "service_failure", "discontinuation", "exception")
REQUEST_STATUSES = ("requested", "approved", "denied", "initiated", "completed", "failed")
OPEN_STATUSES = ("requested", "approved", "initiated")
PAYMENT_KINDS = ("purchase", "upgrade")


def _utc(t: dt.datetime) -> dt.datetime:
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)


def guarantee_deadline(paid_at: dt.datetime) -> dt.datetime:
    return _utc(paid_at) + dt.timedelta(days=GUARANTEE_DAYS)


def within_guarantee(paid_at: dt.datetime, requested_at: dt.datetime) -> bool:
    """The request time decides, even if it is handled later."""
    return _utc(requested_at) <= guarantee_deadline(paid_at)


def add_business_days(t: dt.datetime, days: int) -> dt.datetime:
    """`days` weekdays after `t` (public holidays are not modelled)."""
    out, left = _utc(t), days
    while left > 0:
        out += dt.timedelta(days=1)
        if out.weekday() < 5:
            left -= 1
    return out


def start_due(requested_at: dt.datetime) -> dt.datetime:
    return add_business_days(requested_at, START_WITHIN_BUSINESS_DAYS)


def refundable(amount_cents: int, refunded_cents: int) -> int:
    return max(0, int(amount_cents) - int(refunded_cents))


def add_months(t: dt.datetime, months: int) -> dt.datetime:
    """Same day and time `months` later; a day the month lacks becomes its last day."""
    import calendar

    y, m = divmod(t.month - 1 + months, 12)
    year, month = t.year + y, m + 1
    return t.replace(year=year, month=month, day=min(t.day, calendar.monthrange(year, month)[1]))


def months_between(a: dt.datetime, b: dt.datetime) -> Decimal:
    """Calendar months from a to b (b >= a): whole months plus the elapsed share of the
    next month, measured in seconds within that month."""
    a, b = _utc(a), _utc(b)
    if b <= a:
        return Decimal(0)
    whole = (b.year - a.year) * 12 + (b.month - a.month)
    while whole > 0 and add_months(a, whole) > b:
        whole -= 1
    while add_months(a, whole + 1) <= b:
        whole += 1
    lo, hi = add_months(a, whole), add_months(a, whole + 1)
    part = Decimal(int((b - lo).total_seconds())) / Decimal(int((hi - lo).total_seconds()))
    return Decimal(whole) + part


def prorated_cents(paid_cents: int, start: dt.datetime, end: dt.datetime, stopped_at: dt.datetime,
                   already_refunded_cents: int = 0) -> int:
    """Refund for unused time when service stopped at `stopped_at`:
    paid x months(stopped_at -> end) / months(start -> end), ROUND_HALF_UP to the cent,
    capped at what is left of the payment."""
    start, end, stopped_at = _utc(start), _utc(end), _utc(stopped_at)
    if end <= start:
        raise ValueError("membership end must be after its start")
    total = months_between(start, end)
    unused = months_between(min(max(stopped_at, start), end), end)
    amount = (Decimal(int(paid_cents)) * unused / total).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return min(int(amount), refundable(paid_cents, already_refunded_cents))


def default_amount(basis: str, amount_cents: int, refunded_cents: int, *, prorated: Optional[int] = None) -> int:
    """What an approval refunds unless the admin enters an amount: the prorated amount
    for discontinuation, otherwise everything still refundable."""
    if basis not in BASES:
        raise ValueError(f"unknown refund basis {basis!r}")
    if basis == "discontinuation":
        if prorated is None:
            raise ValueError("a discontinuation refund needs the prorated amount")
        return min(prorated, refundable(amount_cents, refunded_cents))
    return refundable(amount_cents, refunded_cents)
