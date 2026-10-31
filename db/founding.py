"""Founding 100 memberships: seats, checkout holds and memberships (domain/founding.py
holds the prices, seat counts and the 5-year term).

Tables:

* founding_inventory: one row per tier (founder_pro 75, founder_elite 25) with `held`
  (open checkout holds) and `sold` (memberships that occupy a seat). A CHECK keeps
  held + sold <= capacity, so an oversold seat can't be written even by a bug.
* founding_reservations: a short hold (domain.founding.RESERVATION_MINUTES) on one seat for one account, linked to its
  checkout session; one open hold per account. Closed holds keep their row (converted,
  released, expired, or needs_review when a verified payment couldn't be fulfilled).
* founding_memberships: one row per purchase. membership_number 1..100 is UNIQUE and
  given on activation, in payment order, across both tiers; one pending or active
  membership per account; payment_reference (Stripe payment) UNIQUE. A refunded or
  revoked membership gives its seat and number back (kept in former_number); an
  expired one keeps both, since the founder badge is permanent.

Every change runs in one transaction that takes a Postgres advisory lock and then
locks the tier rows it touches (SELECT ... FOR UPDATE), so two buyers can't take the
last seat and two payments can't get the same number. Tests pass a SQLite connection
through set_connection_factory; there a process lock stands in for both locks.

Nothing here talks to Stripe: callers pass verified payment facts in (Milestone 2).
Times are UTC ISO strings written from Python so both backends run the same SQL.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from db.engine import get_neon_conn, schema_once
from domain import founding as rules

LOCK_KEY = 0x48534646  # pg_advisory_xact_lock key for founding changes ("HSFF")

_conn_factory: Optional[Callable[[], Any]] = None
_sqlite_lock = threading.RLock()


class FoundingError(RuntimeError):
    """A founding change that can't happen; `code` is stable for the API layer."""

    code = "founding_error"


class SoldOut(FoundingError):
    code = "sold_out"


class AlreadyMember(FoundingError):
    code = "already_member"


class NotEligible(FoundingError):
    code = "not_eligible"


class NotFound(FoundingError):
    code = "not_found"


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Tests: route every call to `factory()` (a shared SQLite connection, or a Postgres connect)."""
    global _conn_factory
    _conn_factory = factory


def _utc(t: Optional[dt.datetime] = None) -> dt.datetime:
    t = t or dt.datetime.now(dt.timezone.utc)
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return _utc(t).isoformat(timespec="seconds")


def parse_ts(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return _utc(value)
    try:
        return _utc(dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


# ---- connections ------------------------------------------------------------------------------------

def _raw_connect() -> Tuple[Any, bool]:
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")
    return conn, isinstance(conn, sqlite3.Connection)


def _sql(query: str, sq: bool) -> str:
    return query.replace("%s", "?") if sq else query


def _close(conn, sq: bool) -> None:
    if sq and _conn_factory is not None:
        return  # the test's shared connection
    try:
        conn.close()
    except Exception:
        pass


class _Tx:
    """A cursor helper for one transaction: run SQL with %s placeholders on either backend."""

    def __init__(self, conn, sq: bool):
        self.conn, self.sq, self.cur = conn, sq, conn.cursor()

    def run(self, query: str, params: Tuple = ()) -> Any:
        self.cur.execute(_sql(query, self.sq), params)
        return self.cur

    def rows(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        cur = self.run(query, params)
        names = [d[0] for d in cur.description]
        out = []
        for row in cur.fetchall():
            if isinstance(row, dict):
                out.append(dict(row))
            else:
                out.append(dict(zip(names, tuple(row))))
        return out

    def one(self, query: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        found = self.rows(query, params)
        return found[0] if found else None


@contextmanager
def _transaction(*, write: bool = True) -> Iterator[_Tx]:
    conn, sq = _raw_connect()
    guard = _sqlite_lock if sq else None
    if guard:
        guard.acquire()
    try:
        _ensure_schema(conn, sq)
        tx = _Tx(conn, sq)
        try:
            if write and not sq:
                tx.run("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            yield tx
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        _close(conn, sq)
        if guard:
            guard.release()


@schema_once
def _ensure_schema(conn, sq: bool) -> None:
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if sq else "BIGSERIAL PRIMARY KEY"
    tiers = ", ".join(f"'{t}'" for t in rules.TIERS)
    statuses = ", ".join(f"'{s}'" for s in rules.STATUSES)
    kinds = ", ".join(f"'{k}'" for k in rules.RESERVATION_KINDS)
    rstatuses = ", ".join(f"'{s}'" for s in rules.RESERVATION_STATUSES)
    cur = conn.cursor()
    if not sq:
        # Same lock as every change, so creating tables or seeding seats never interleaves
        # with a sale (two servers starting together otherwise deadlock on the seed rows).
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_inventory (
            tier TEXT PRIMARY KEY CHECK (tier IN ({tiers})),
            capacity INTEGER NOT NULL CHECK (capacity >= 0),
            held INTEGER NOT NULL DEFAULT 0 CHECK (held >= 0),
            sold INTEGER NOT NULL DEFAULT 0 CHECK (sold >= 0),
            updated_at TEXT NOT NULL,
            CHECK (held + sold <= capacity)
        )""")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_memberships (
            id {pk},
            user_id TEXT NOT NULL,
            membership_number INTEGER UNIQUE CHECK (membership_number BETWEEN 1 AND {rules.TOTAL_CAPACITY}),
            former_number INTEGER,
            tier TEXT NOT NULL CHECK (tier IN ({tiers})),
            status TEXT NOT NULL CHECK (status IN ({statuses})),
            purchase_price_cents INTEGER NOT NULL CHECK (purchase_price_cents >= 0),
            currency TEXT NOT NULL DEFAULT '{rules.CURRENCY}',
            payment_provider TEXT NOT NULL DEFAULT 'stripe',
            payment_reference TEXT UNIQUE,
            upgrade_price_cents INTEGER,
            upgrade_payment_reference TEXT UNIQUE,
            purchased_at TEXT,
            activated_at TEXT,
            expires_at TEXT,
            upgraded_at TEXT,
            cancelled_at TEXT,
            refunded_at TEXT,
            status_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (status <> 'active' OR (membership_number IS NOT NULL AND activated_at IS NOT NULL
                                          AND expires_at IS NOT NULL))
        )""")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_founding_memberships_one_live "
                "ON founding_memberships (user_id) WHERE status IN ('pending', 'active')")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_founding_memberships_user ON founding_memberships (user_id)")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_reservations (
            id {pk},
            user_id TEXT NOT NULL,
            tier TEXT NOT NULL CHECK (tier IN ({tiers})),
            kind TEXT NOT NULL CHECK (kind IN ({kinds})),
            status TEXT NOT NULL CHECK (status IN ({rstatuses})),
            checkout_session_id TEXT UNIQUE,
            payment_reference TEXT UNIQUE,
            membership_id INTEGER,
            note TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            closed_at TEXT
        )""")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_founding_reservations_one_open "
                "ON founding_reservations (user_id) WHERE status = 'held'")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_founding_reservations_tier_open "
                "ON founding_reservations (tier, status, expires_at)")
    # Payments, refunds and the audit trail (db/founding_refunds.py). One row per
    # verified payment: the purchase and an upgrade are separate payments.
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_payments (
            id {pk},
            membership_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('purchase', 'upgrade')),
            tier TEXT NOT NULL,
            payment_reference TEXT NOT NULL UNIQUE,
            amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
            currency TEXT NOT NULL,
            paid_at TEXT NOT NULL,
            refunded_cents INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (refunded_cents >= 0 AND refunded_cents <= amount_cents)
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_founding_payments_user ON founding_payments (user_id)")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_refund_requests (
            id {pk},
            payment_id INTEGER NOT NULL REFERENCES founding_payments (id),
            user_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('requested', 'approved', 'denied', 'initiated', 'completed', 'failed')),
            within_guarantee INTEGER NOT NULL,
            member_reason TEXT,
            basis TEXT,
            amount_cents INTEGER CHECK (amount_cents IS NULL OR amount_cents > 0),
            decision_reason TEXT,
            decided_by TEXT,
            provider_refund_id TEXT UNIQUE,
            failure_reason TEXT,
            requested_at TEXT NOT NULL,
            due_at TEXT NOT NULL,
            decided_at TEXT,
            initiated_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        )""")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_founding_refund_one_open ON founding_refund_requests (payment_id) "
                "WHERE status IN ('requested', 'approved', 'initiated')")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_founding_refund_user ON founding_refund_requests (user_id)")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS founding_audit_log (
            id {pk},
            at TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            refund_request_id INTEGER,
            payment_reference TEXT,
            detail TEXT
        )""")
    # Founder benefits (db/founding_benefits.py): badge on shared links and the feature poll.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS founding_member_prefs (
            user_id TEXT PRIMARY KEY,
            show_badge INTEGER NOT NULL DEFAULT 1,
            poll_choice TEXT,
            poll_at TEXT,
            updated_at TEXT NOT NULL
        )""")
    # Founders Wall names (opt-in; a row means the member chose to be listed) and the
    # monthly Founder Briefing, one row per month an admin sent it.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS founding_wall (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS founding_briefings (
            month TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            sent_by TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            sent INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0
        )""")
    now = _iso(_utc())
    for tier, capacity in rules.CAPACITY.items():
        cur.execute(_sql("INSERT INTO founding_inventory (tier, capacity, held, sold, updated_at) "
                         "VALUES (%s, %s, 0, 0, %s) ON CONFLICT (tier) DO NOTHING", sq), (tier, capacity, now))
    conn.commit()
    cur.close()


# ---- shared steps (inside a transaction) -------------------------------------------------------------

def _lock_tiers(tx: _Tx, tiers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Lock the inventory rows (always in the same order, so two changes can't deadlock)."""
    out = {}
    for tier in sorted(set(tiers)):
        lock = "" if tx.sq else " FOR UPDATE"
        row = tx.one(f"SELECT tier, capacity, held, sold FROM founding_inventory WHERE tier = %s{lock}", (tier,))
        if row is None:
            raise NotFound(f"unknown founding tier {tier!r}")
        out[tier] = row
    return out


def _sweep_expired(tx: _Tx, tier: str, now: dt.datetime) -> int:
    """Close holds on `tier` whose hold time ran out and give their seats back."""
    stale = tx.rows("SELECT id FROM founding_reservations WHERE tier = %s AND status = 'held' AND expires_at <= %s",
                    (tier, _iso(now)))
    if not stale:
        return 0
    for r in stale:
        tx.run("UPDATE founding_reservations SET status = 'expired', closed_at = %s WHERE id = %s AND status = 'held'",
               (_iso(now), r["id"]))
    tx.run("UPDATE founding_inventory SET held = held - %s, updated_at = %s WHERE tier = %s",
           (len(stale), _iso(now), tier))
    return len(stale)


def _live_membership(tx: _Tx, user_id: str) -> Optional[Dict[str, Any]]:
    return tx.one("SELECT * FROM founding_memberships WHERE user_id = %s AND status IN ('pending', 'active')", (user_id,))


def _free_seat(tx: _Tx, tier: str) -> bool:
    inv = tx.one("SELECT capacity, held, sold FROM founding_inventory WHERE tier = %s", (tier,))
    return bool(inv) and int(inv["held"]) + int(inv["sold"]) < int(inv["capacity"])


def _next_number(tx: _Tx) -> int:
    """The lowest founder number not held by any membership (refunds give theirs back)."""
    taken = {int(r["membership_number"]) for r in
             tx.rows("SELECT membership_number FROM founding_memberships WHERE membership_number IS NOT NULL")}
    for n in range(1, rules.TOTAL_CAPACITY + 1):
        if n not in taken:
            return n
    raise SoldOut("Every founder number is taken.")


def _record_payment(tx: _Tx, m: Dict[str, Any], kind: str, tier: str, ref: str, amount: int,
                    paid_at: dt.datetime, now: dt.datetime) -> None:
    tx.run("INSERT INTO founding_payments (membership_id, user_id, kind, tier, payment_reference, amount_cents, "
           "currency, paid_at, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
           (m["id"], m["user_id"], kind, tier, ref, amount, rules.CURRENCY, _iso(paid_at), _iso(now), _iso(now)))


def _close_hold(tx: _Tx, res: Dict[str, Any], status: str, now: dt.datetime, *,
                payment_reference: Optional[str] = None, membership_id: Optional[int] = None,
                note: Optional[str] = None) -> None:
    """Close a reservation; a hold that was still open gives its held seat back."""
    if res["status"] == "held":
        tx.run("UPDATE founding_inventory SET held = held - 1, updated_at = %s WHERE tier = %s", (_iso(now), res["tier"]))
    tx.run("UPDATE founding_reservations SET status = %s, closed_at = %s, "
           "payment_reference = COALESCE(%s, payment_reference), membership_id = COALESCE(%s, membership_id), "
           "note = COALESCE(%s, note) WHERE id = %s",
           (status, _iso(now), payment_reference, membership_id, note, res["id"]))


# ---- reads ------------------------------------------------------------------------------------------

def availability(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Seats per tier: capacity, sold, held (open, unexpired holds) and remaining. Verified
    counts from the database; expired holds don't count even before they're swept."""
    now = _utc(now)
    with _transaction(write=False) as tx:
        inv = {r["tier"]: r for r in tx.rows("SELECT tier, capacity, sold FROM founding_inventory")}
        open_holds = {r["tier"]: int(r["n"]) for r in tx.rows(
            "SELECT tier, COUNT(*) AS n FROM founding_reservations WHERE status = 'held' AND expires_at > %s GROUP BY tier",
            (_iso(now),))}
    tiers = {}
    for tier in rules.TIERS:
        row = inv.get(tier) or {"capacity": rules.CAPACITY[tier], "sold": 0}
        cap, sold, held = int(row["capacity"]), int(row["sold"]), open_holds.get(tier, 0)
        tiers[tier] = {"capacity": cap, "sold": sold, "held": held, "remaining": max(0, cap - sold - held)}
    return {"tiers": tiers, "capacity": sum(t["capacity"] for t in tiers.values()),
            "sold": sum(t["sold"] for t in tiers.values()),
            "remaining": sum(t["remaining"] for t in tiers.values())}


def membership(user_id: str) -> Optional[Dict[str, Any]]:
    """The account's current membership: the live one, else its most recent (an expired
    member keeps the badge). None when it never bought one."""
    user_id = user_id.strip().lower()
    with _transaction(write=False) as tx:
        live = _live_membership(tx, user_id)
        if live:
            return live
        return tx.one("SELECT * FROM founding_memberships WHERE user_id = %s ORDER BY id DESC LIMIT 1", (user_id,))


def active_tier(user_id: str, now: Optional[dt.datetime] = None) -> Tuple[Optional[str], Optional[dt.datetime]]:
    """(founder tier, access end) while the membership is active and inside its 5 years, else (None, None)."""
    now = _utc(now)
    with _transaction(write=False) as tx:
        row = tx.one("SELECT tier, expires_at FROM founding_memberships WHERE user_id = %s AND status = 'active'",
                     (user_id.strip().lower(),))
    if not row:
        return None, None
    ends = parse_ts(row["expires_at"])
    if ends is None or ends <= now:
        return None, None
    return str(row["tier"]), ends


def any_members() -> bool:
    """Cheap check so accounts skip the per-account lookup until the first sale."""
    with _transaction(write=False) as tx:
        return tx.one("SELECT 1 AS x FROM founding_memberships WHERE status = 'active' LIMIT 1") is not None


def open_reservation(user_id: str, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """The account's unexpired seat hold, if any."""
    with _transaction(write=False) as tx:
        return tx.one("SELECT * FROM founding_reservations WHERE user_id = %s AND status = 'held' AND expires_at > %s",
                      (user_id.strip().lower(), _iso(_utc(now))))


def reservation(reservation_id: int) -> Optional[Dict[str, Any]]:
    with _transaction(write=False) as tx:
        return tx.one("SELECT * FROM founding_reservations WHERE id = %s", (int(reservation_id),))


# ---- changes ----------------------------------------------------------------------------------------

def reserve(user_id: str, tier: str, *, kind: str = "purchase", now: Optional[dt.datetime] = None,
            minutes: int = rules.RESERVATION_MINUTES) -> Dict[str, Any]:
    """Hold one seat of `tier` for `user_id` for `minutes`. Asking again for the same
    tier and kind returns the open hold (no second seat); a different one replaces it.
    Raises AlreadyMember, NotEligible (upgrade without Founder Pro) or SoldOut."""
    user_id, now = user_id.strip().lower(), _utc(now)
    if tier not in rules.TIERS or kind not in rules.RESERVATION_KINDS:
        raise NotEligible(f"unknown founding option {kind!r} {tier!r}")
    if kind == "upgrade" and tier != rules.FOUNDER_ELITE:
        raise NotEligible("Only Founder Pro members can upgrade, to Founder Elite.")
    with _transaction() as tx:
        _lock_tiers(tx, [tier])
        _sweep_expired(tx, tier, now)
        live = _live_membership(tx, user_id)
        if kind == "purchase" and live:
            raise AlreadyMember("This account already has a founding membership.")
        if kind == "upgrade" and not (live and live["status"] == "active" and live["tier"] == rules.FOUNDER_PRO):
            raise NotEligible("Only active Founder Pro members can upgrade.")
        mine = tx.one("SELECT * FROM founding_reservations WHERE user_id = %s AND status = 'held'", (user_id,))
        if mine:
            if mine["tier"] == tier and mine["kind"] == kind and (parse_ts(mine["expires_at"]) or now) > now:
                return mine
            _lock_tiers(tx, [mine["tier"]])
            _close_hold(tx, mine, "released", now, note="replaced")
        if not _free_seat(tx, tier):
            raise SoldOut(f"No {rules.TIER_LABEL[tier]} seats are left.")
        tx.run("UPDATE founding_inventory SET held = held + 1, updated_at = %s WHERE tier = %s", (_iso(now), tier))
        tx.run("INSERT INTO founding_reservations (user_id, tier, kind, status, created_at, expires_at) "
               "VALUES (%s, %s, %s, 'held', %s, %s)",
               (user_id, tier, kind, _iso(now), _iso(now + dt.timedelta(minutes=minutes))))
        return tx.one("SELECT * FROM founding_reservations WHERE user_id = %s AND status = 'held'", (user_id,))


def attach_checkout(reservation_id: int, checkout_session_id: str) -> None:
    """Link a hold to its checkout session (the session id is unique across holds)."""
    with _transaction() as tx:
        tx.run("UPDATE founding_reservations SET checkout_session_id = %s WHERE id = %s",
               (checkout_session_id, int(reservation_id)))


def release(reservation_id: int, *, reason: str = "released", now: Optional[dt.datetime] = None) -> bool:
    """Give a hold's seat back (checkout abandoned or expired at the provider). False when
    it was already closed."""
    now = _utc(now)
    with _transaction() as tx:
        res = tx.one("SELECT * FROM founding_reservations WHERE id = %s", (int(reservation_id),))
        if not res or res["status"] != "held":
            return False
        _lock_tiers(tx, [res["tier"]])
        _close_hold(tx, res, "released", now, note=reason[:80])
        return True


def fulfill(reservation_id: int, *, payment_reference: str, amount_cents: int, currency: str = rules.CURRENCY,
            paid_at: Optional[dt.datetime] = None, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Turn a verified payment into a membership (purchase) or an Elite upgrade.

    Idempotent on `payment_reference`: a repeated webhook returns the same result.
    Returns {"outcome": "activated" | "upgraded" | "already_done" | "needs_review", ...}.
    needs_review means the money arrived but no seat could be given (wrong amount, the
    hold expired and the tier sold out meanwhile, or the account changed): nothing is
    granted and the hold is flagged for refund handling."""
    now = _utc(now)
    paid_at = _utc(paid_at or now)
    ref = str(payment_reference or "").strip()
    if not ref:
        raise ValueError("payment_reference is required")
    with _transaction() as tx:
        done = tx.one("SELECT * FROM founding_memberships WHERE payment_reference = %s OR upgrade_payment_reference = %s",
                      (ref, ref))
        if done:
            return {"outcome": "already_done", "membership": done}
        res = tx.one("SELECT * FROM founding_reservations WHERE id = %s", (int(reservation_id),))
        if not res:
            raise NotFound("No such founding checkout.")
        if res["status"] in ("converted", "needs_review") and res.get("payment_reference") == ref:
            return {"outcome": "already_done" if res["status"] == "converted" else "needs_review", "reservation": res}
        tier, kind, user_id = res["tier"], res["kind"], res["user_id"]
        tiers = [tier, rules.FOUNDER_PRO] if kind == "upgrade" else [tier]
        _lock_tiers(tx, tiers)
        _sweep_expired(tx, tier, now)
        res = tx.one("SELECT * FROM founding_reservations WHERE id = %s", (int(reservation_id),))

        def review(note: str) -> Dict[str, Any]:
            _close_hold(tx, res, "needs_review", now, payment_reference=ref, note=note)
            return {"outcome": "needs_review", "reason": note,
                    "reservation": tx.one("SELECT * FROM founding_reservations WHERE id = %s", (res["id"],))}

        if int(amount_cents) != rules.expected_amount(kind, tier) or str(currency).lower() != rules.CURRENCY:
            return review("amount_mismatch")
        if res["status"] not in ("held", "expired", "released"):
            return review(f"reservation_{res['status']}")
        # A hold that is still open carries its seat; a late payment needs a free one.
        if res["status"] != "held" and not _free_seat(tx, tier):
            return review("sold_out_after_hold_expired")
        live = _live_membership(tx, user_id)
        if kind == "purchase":
            if live:
                return review("already_member")
            number = _next_number(tx)
            if res["status"] == "held":
                tx.run("UPDATE founding_inventory SET held = held - 1 WHERE tier = %s", (tier,))
            tx.run("UPDATE founding_inventory SET sold = sold + 1, updated_at = %s WHERE tier = %s", (_iso(now), tier))
            tx.run("INSERT INTO founding_memberships (user_id, membership_number, tier, status, purchase_price_cents, "
                   "currency, payment_provider, payment_reference, purchased_at, activated_at, expires_at, created_at, "
                   "updated_at) VALUES (%s, %s, %s, 'active', %s, %s, 'stripe', %s, %s, %s, %s, %s, %s)",
                   (user_id, number, tier, int(amount_cents), rules.CURRENCY, ref, _iso(paid_at), _iso(paid_at),
                    _iso(rules.term_end(paid_at)), _iso(now), _iso(now)))
            m = tx.one("SELECT * FROM founding_memberships WHERE payment_reference = %s", (ref,))
            _record_payment(tx, m, "purchase", tier, ref, int(amount_cents), paid_at, now)
            tx.run("UPDATE founding_reservations SET status = 'converted', closed_at = %s, payment_reference = %s, "
                   "membership_id = %s WHERE id = %s", (_iso(now), ref, m["id"], res["id"]))
            return {"outcome": "activated", "membership": m}
        # Upgrade: same membership, number and dates; one Pro seat becomes one Elite seat.
        if not (live and live["status"] == "active" and live["tier"] == rules.FOUNDER_PRO):
            return review("not_founder_pro")
        if res["status"] == "held":
            tx.run("UPDATE founding_inventory SET held = held - 1 WHERE tier = %s", (tier,))
        tx.run("UPDATE founding_inventory SET sold = sold + 1, updated_at = %s WHERE tier = %s", (_iso(now), tier))
        tx.run("UPDATE founding_inventory SET sold = sold - 1, updated_at = %s WHERE tier = %s",
               (_iso(now), rules.FOUNDER_PRO))
        tx.run("UPDATE founding_memberships SET tier = %s, upgrade_price_cents = %s, upgrade_payment_reference = %s, "
               "upgraded_at = %s, updated_at = %s WHERE id = %s",
               (tier, int(amount_cents), ref, _iso(paid_at), _iso(now), live["id"]))
        _record_payment(tx, live, "upgrade", tier, ref, int(amount_cents), paid_at, now)
        tx.run("UPDATE founding_reservations SET status = 'converted', closed_at = %s, payment_reference = %s, "
               "membership_id = %s WHERE id = %s", (_iso(now), ref, live["id"], res["id"]))
        return {"outcome": "upgraded", "membership": tx.one("SELECT * FROM founding_memberships WHERE id = %s", (live["id"],))}


def _end(payment_reference: str, status: str, reason: str, now: Optional[dt.datetime]) -> Dict[str, Any]:
    now = _utc(now)
    ref = str(payment_reference or "").strip()
    with _transaction() as tx:
        m = tx.one("SELECT * FROM founding_memberships WHERE payment_reference = %s OR upgrade_payment_reference = %s",
                   (ref, ref))
        if not m:
            raise NotFound("No founding membership was paid with that reference.")
        if m["status"] in ("refunded", "revoked"):
            return {"outcome": "already_done", "membership": m}
        _lock_tiers(tx, list(rules.TIERS))
        if m.get("upgrade_payment_reference") == ref and m["status"] == "active":
            # Only the upgrade is reversed: back to Founder Pro if a Pro seat is free.
            if not _free_seat(tx, rules.FOUNDER_PRO):
                return {"outcome": "needs_review", "reason": "no_pro_seat_for_upgrade_reversal", "membership": m}
            tx.run("UPDATE founding_inventory SET sold = sold - 1, updated_at = %s WHERE tier = %s",
                   (_iso(now), rules.FOUNDER_ELITE))
            tx.run("UPDATE founding_inventory SET sold = sold + 1, updated_at = %s WHERE tier = %s",
                   (_iso(now), rules.FOUNDER_PRO))
            tx.run("UPDATE founding_memberships SET tier = %s, upgrade_payment_reference = NULL, upgraded_at = NULL, "
                   "status_reason = %s, updated_at = %s WHERE id = %s",
                   (rules.FOUNDER_PRO, f"upgrade_{reason}"[:80], _iso(now), m["id"]))
            return {"outcome": "upgrade_reversed",
                    "membership": tx.one("SELECT * FROM founding_memberships WHERE id = %s", (m["id"],))}
        if m["status"] in ("active", "pending"):
            tx.run("UPDATE founding_inventory SET sold = sold - 1, updated_at = %s WHERE tier = %s",
                   (_iso(now), m["tier"]))
        stamp = "refunded_at" if status == "refunded" else "cancelled_at"
        tx.run(f"UPDATE founding_memberships SET status = %s, {stamp} = %s, former_number = membership_number, "
               "membership_number = NULL, status_reason = %s, updated_at = %s WHERE id = %s",
               (status, _iso(now), reason[:80], _iso(now), m["id"]))
        return {"outcome": status, "membership": tx.one("SELECT * FROM founding_memberships WHERE id = %s", (m["id"],))}


def refund(payment_reference: str, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """A refunded purchase ends the membership and gives its seat and number back. A
    refunded upgrade payment only moves the member back to Founder Pro."""
    return _end(payment_reference, "refunded", "refund", now)


def revoke(payment_reference: str, *, reason: str = "dispute", now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """A lost dispute (or an admin revoke, audited by the caller): same as a refund."""
    return _end(payment_reference, "revoked", reason, now)


def expire_due(now: Optional[dt.datetime] = None) -> int:
    """Mark active memberships past their 5 years as expired. They keep their number and
    seat (the badge is permanent); access already stops at expires_at without this."""
    now = _utc(now)
    with _transaction() as tx:
        cur = tx.run("UPDATE founding_memberships SET status = 'expired', updated_at = %s "
                     "WHERE status = 'active' AND expires_at <= %s", (_iso(now), _iso(now)))
        return int(cur.rowcount or 0)


def sweep_holds(now: Optional[dt.datetime] = None) -> int:
    """Close every expired hold (also done on each new hold); for the reconciliation loop."""
    now = _utc(now)
    with _transaction() as tx:
        _lock_tiers(tx, list(rules.TIERS))
        return sum(_sweep_expired(tx, t, now) for t in rules.TIERS)


def needs_review() -> List[Dict[str, Any]]:
    """Paid checkouts that got no seat, oldest first, for refund handling."""
    with _transaction(write=False) as tx:
        return tx.rows("SELECT * FROM founding_reservations WHERE status = 'needs_review' ORDER BY id")
