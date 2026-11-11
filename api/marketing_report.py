"""Admin Marketing report: where visitors, list sign-ups, accounts and sales come from.

One row per source (X, Reddit, invite link, Google, ...), from the anonymous funnel events
the landing pages record (ui.acquisition) and the email list (db.marketing_list):

    Visits        landing_visit events (one per browser session; not people)
    Clicks        "Create account" / "Try it free" clicks
    List          email-list sign-ups created in the period (confirmed ones in brackets)
    Accounts      sign-ups completed
    Paid          paid plan conversions, credited to the buyer's sign-up source
    Founders      Founding 100 seats bought in the period, credited the same way

Plus the same counts per utm campaign, so a tagged post (for example from Buffer) can be
compared with another. Pure functions over the loader's rows; admins only, read-only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

from api.admin_analytics import PERIODS, SOURCE_LABELS, AnalyticsError, AnalyticsUnavailable

COLUMNS = ("visits", "clicks", "list_signups", "list_confirmed", "accounts", "paid", "founders")
EVENT_COLUMN = {"landing_visit": "visits", "primary_cta_click": "clicks", "signup_completed": "accounts"}
MAX_CAMPAIGNS = 15


def _label(source: Any) -> str:
    return SOURCE_LABELS.get(str(source or "").strip() or "direct", "Other")


def report(data: Mapping[str, Any], subscribers: Sequence[Mapping[str, Any]], *, since: Optional[str] = None,
           days: int = 30, sync_configured: bool = False) -> Dict[str, Any]:
    from ui.acquisition import user_hash

    sources = data.get("sources") or {}
    by_source: Dict[str, Counter] = defaultdict(Counter)
    by_campaign: Dict[tuple, Counter] = defaultdict(Counter)
    revenue = 0

    def campaign(e: Mapping[str, Any]) -> Optional[tuple]:
        c = str(e.get("utm_campaign") or "").strip()
        return (str(e.get("utm_source") or "").strip() or "(none)", c) if c else None

    for e in data.get("events") or []:
        name = e.get("event")
        if name in EVENT_COLUMN:
            col = EVENT_COLUMN[name]
            by_source[_label(e.get("source"))][col] += 1
            if (key := campaign(e)) is not None:
                by_campaign[key][col] += 1
        elif name == "successful_paid_conversion":
            by_source[_label(sources.get(str(e.get("user_hash") or ""), ""))]["paid"] += 1
    for f in data.get("founders") or []:
        if f.get("status") not in {"active", "pending"}:
            continue
        by_source[_label(sources.get(user_hash(f.get("user")) or "", ""))]["founders"] += 1
        revenue += int(f.get("cents") or 0)
    counts = Counter(str(s.get("status") or "pending") for s in subscribers)
    new_list = 0
    for s in subscribers:
        if since and str(s.get("created_at") or "") < since:
            continue
        new_list += 1
        label = _label(s.get("source"))
        by_source[label]["list_signups"] += 1
        by_source[label]["list_confirmed"] += s.get("status") == "confirmed"
        if (key := campaign(s)) is not None:
            by_campaign[key]["list_signups"] += 1

    rows = [{"source": s, **{c: int(n[c]) for c in COLUMNS}}
            for s, n in sorted(by_source.items(), key=lambda kv: (-kv[1]["accounts"], -kv[1]["visits"], kv[0]))]
    totals = {c: sum(r[c] for r in rows) for c in COLUMNS}
    camps = [{"utm_source": k[0], "campaign": k[1], "visits": int(n["visits"]), "clicks": int(n["clicks"]),
              "list_signups": int(n["list_signups"]), "accounts": int(n["accounts"])}
             for k, n in sorted(by_campaign.items(), key=lambda kv: (-kv[1]["accounts"], -kv[1]["visits"], kv[0]))]
    return {
        "days": days, "by_source": rows, "totals": totals, "campaigns": camps[:MAX_CAMPAIGNS],
        "founder_revenue_cents": revenue,
        "list": {"confirmed": counts.get("confirmed", 0), "pending": counts.get("pending", 0),
                 "unsubscribed": counts.get("unsubscribed", 0), "new_in_period": new_list,
                 "sync_configured": bool(sync_configured)},
        "unavailable": sorted((data.get("errors") or {}).keys()),
    }


def marketing(days: int = 30) -> Dict[str, Any]:
    if int(days) not in PERIODS:
        raise AnalyticsError(f"Pick a period of {', '.join(map(str, PERIODS))} days.")
    from db.admin_analytics import load_marketing
    from db.marketing_list import all_rows
    from services.marketing_list import sync_configured

    data = load_marketing(days=int(days))
    if not data.get("available"):
        raise AnalyticsUnavailable("The marketing report is unavailable: the database is not reachable.")
    return report(data, all_rows(), since=data.get("since"), days=int(days), sync_configured=sync_configured())


CSV_FIELDS = ("email", "status", "source", "utm_source", "utm_medium", "utm_campaign", "surface", "created_at",
              "confirmed_at")


def subscribers_csv(rows: Sequence[Mapping[str, Any]], *, status: str = "confirmed") -> str:
    """Confirmed addresses (by default) as CSV with an `email` column first, the shape
    Resend, Kit and Beehiiv contact imports accept."""
    import csv
    import io

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(CSV_FIELDS)
    for r in rows:
        if status != "all" and r.get("status") != status:
            continue
        cells = [str(r.get(f) or "") for f in CSV_FIELDS]
        # A leading = + - @ makes spreadsheets run the cell as a formula.
        w.writerow([("'" + c) if c[:1] in "=+-@" else c for c in cells])
    return out.getvalue()

