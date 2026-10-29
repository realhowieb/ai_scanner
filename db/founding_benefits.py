"""Founding 100 extras kept per member: whether shared stock links show the founder
badge, and the member's choice in the founder feature poll. Uses db/founding.py's
connection and tables (founding_member_prefs)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

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
