"""Founding 100 extras kept per member: whether shared stock links show the founder
badge, the member's choice in the founder feature poll, the opt-in Founders Wall name and the
monthly Founder Briefing send log. Uses db/founding.py's
connection and tables (founding_member_prefs)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from db import founding as db
from domain import founding as rules


def _row(tx, user: str) -> Optional[Dict[str, Any]]:
    return tx.one("SELECT * FROM founding_member_prefs WHERE user_id = %s", (user,))


def _upsert(tx, user: str, now: dt.datetime, **fields: Any) -> None:
    if _row(tx, user):
        sets = ", ".join(f"{k} = %s" for k in fields)
        tx.run(f"UPDATE founding_member_prefs SET {sets}, updated_at = %s WHERE user_id = %s",
               (*fields.values(), db._iso(now), user))
    else:
        cols = ", ".join(["user_id", *fields, "updated_at"])
        marks = ", ".join(["%s"] * (len(fields) + 2))
        tx.run(f"INSERT INTO founding_member_prefs ({cols}) VALUES ({marks})", (user, *fields.values(), db._iso(now)))


def prefs(user_id: str) -> Dict[str, Any]:
    user = user_id.strip().lower()
    with db._transaction(write=False) as tx:
        r = _row(tx, user)
    return {"show_badge": bool(int(r["show_badge"])) if r else True,
            "poll_choice": (r or {}).get("poll_choice"), "poll_at": (r or {}).get("poll_at")}


def set_show_badge(user_id: str, show: bool, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    with db._transaction() as tx:
        _upsert(tx, user_id.strip().lower(), db._utc(now), show_badge=1 if show else 0)
    return prefs(user_id)


def vote(user_id: str, option: str, *, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """One choice per founder; voting again changes it. Unknown options are refused."""
    if option not in rules.poll_options():
        raise db.FoundingError("That option isn't in the poll.")
    now = db._utc(now)
    with db._transaction() as tx:
        _upsert(tx, user_id.strip().lower(), now, poll_choice=option, poll_at=db._iso(now))
    return prefs(user_id)


def tally() -> Dict[str, int]:
    """Votes per current option (choices for options since removed are not counted)."""
    options = rules.poll_options()
    out = {k: 0 for k in options}
    with db._transaction(write=False) as tx:
        for r in tx.rows("SELECT poll_choice, COUNT(*) AS n FROM founding_member_prefs WHERE poll_choice IS NOT NULL "
                         "GROUP BY poll_choice"):
            if r["poll_choice"] in out:
                out[r["poll_choice"]] = int(r["n"])
    return out


def public_badge(user_id: str) -> Optional[str]:
    """The founder badge a shared link may show for this account, or None (no membership,
    refunded or revoked, or the member turned it off)."""
    m = db.membership(user_id)
    if not m or m["status"] not in ("active", "expired"):
        return None
    if not prefs(user_id)["show_badge"]:
        return None
    return rules.badge_title(m["tier"], m.get("membership_number") or m.get("former_number"))


# ---- Founders Wall (founding_wall; opt-in, off by default) --------------------------------------

def wall_name(user_id: str) -> Optional[str]:
    with db._transaction(write=False) as tx:
        r = tx.one("SELECT display_name FROM founding_wall WHERE user_id = %s", (user_id.strip().lower(),))
    return str(r["display_name"]) if r else None


def set_wall_name(user_id: str, name: Optional[str], *, now: Optional[dt.datetime] = None) -> Optional[str]:
    """List the member on the wall under `name` (already cleaned), or take them off (None)."""
    user = user_id.strip().lower()
    with db._transaction() as tx:
        tx.run("DELETE FROM founding_wall WHERE user_id = %s", (user,))
        if name:
            tx.run("INSERT INTO founding_wall (user_id, display_name, updated_at) VALUES (%s, %s, %s)",
                   (user, name, db._iso(db._utc(now))))
    return wall_name(user)


def wall() -> List[Dict[str, Any]]:
    """Every founder seat, #001 to #100: taken seats with their tier and, only when the member
    opted in, their display name; the rest open. Never an email or account id."""
    with db._transaction(write=False) as tx:
        rows = tx.rows("SELECT m.membership_number, m.tier, m.status, m.activated_at, w.display_name "
                       "FROM founding_memberships m LEFT JOIN founding_wall w ON w.user_id = m.user_id "
                       "WHERE m.membership_number IS NOT NULL AND m.status IN ('active', 'expired')")
    taken = {int(r["membership_number"]): r for r in rows}
    out: List[Dict[str, Any]] = []
    for n in range(1, rules.TOTAL_CAPACITY + 1):
        r = taken.get(n)
        out.append({"number": rules.format_number(n), "taken": bool(r),
                    "tier": r["tier"] if r else None,
                    "label": rules.TIER_LABEL.get(r["tier"]) if r else None,
                    "name": (r.get("display_name") or None) if r else None,
                    "since": str(r["activated_at"])[:7] if r and r.get("activated_at") else None})
    return out


# ---- members for the monthly Founder Briefing, and its send log (founding_briefings) -----------

def active_members() -> List[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT user_id, tier, membership_number, activated_at, expires_at FROM founding_memberships "
                       "WHERE status = 'active' ORDER BY membership_number")


def briefing(month: str) -> Optional[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.one("SELECT * FROM founding_briefings WHERE month = %s", (month,))


def claim_briefing(month: str, subject: str, admin: str, *, now: Optional[dt.datetime] = None) -> bool:
    """True for the first send of `month` only, so a double click never mails founders twice."""
    with db._transaction() as tx:
        if tx.one("SELECT 1 AS x FROM founding_briefings WHERE month = %s", (month,)):
            return False
        tx.run("INSERT INTO founding_briefings (month, subject, sent_by, started_at) VALUES (%s, %s, %s, %s)",
               (month, subject, admin, db._iso(db._utc(now))))
    return True


def finish_briefing(month: str, sent: int, failed: int, *, now: Optional[dt.datetime] = None) -> None:
    with db._transaction() as tx:
        tx.run("UPDATE founding_briefings SET sent = %s, failed = %s, finished_at = %s WHERE month = %s",
               (int(sent), int(failed), db._iso(db._utc(now)), month))


def briefings(limit: int = 12) -> List[Dict[str, Any]]:
    with db._transaction(write=False) as tx:
        return tx.rows("SELECT * FROM founding_briefings ORDER BY month DESC LIMIT %s", (int(limit),))
