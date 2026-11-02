"""Weekly social posts, written by HSF and sent to Buffer as drafts for an admin to approve.

Four posts a week, each to every Buffer channel (Instagram, Threads, LinkedIn, ...):

    earnings   Mon 8:30am ET   the week's biggest reporters (api/earnings_week)
    product    Wed noon ET     one of PRODUCT_POSTS, in turn by week
    founding   Fri noon ET     Founding 100 seats left (skipped when closed or sold out)
    recap      Sat 10am ET     the top 3 from the public weekly recap (api/public_recap)

api/ops_jobs.py sends each one from the evening before (recap: from Saturday 1am ET, once
the week's recap is published) until 2 hours before its post time. Each channel gets its own text: a
link tagged with utm_source for that network (so visits show up in Admin > Marketing),
or "link in bio" plus hashtags on Instagram, where caption links don't work. Every post
carries an image drawn by the web app at /social/<kind>/<week>.png from the same stored
post (GET /v1/public/social/<kind>/<week>).

Drafts only: nothing publishes until someone approves it in Buffer. Text is checked for
performance claims (returns, wins, "beat", percentages) before anything is sent; the
forward test found no proven edge, so posts say what HSF ranked or shows, never how it did.

Env:
    BUFFER_API_KEY              personal key from publish.buffer.com/settings/api (none: nothing sent)
    HSF_SOCIAL_DRAFTS_ENABLED   0 stops the weekly sends (default on)
    HSF_SOCIAL_CHANNELS         optional comma list of Buffer services to use (default: every channel)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

log = logging.getLogger("hsf.social_posts")

UTC = dt.timezone.utc
BUFFER_API = "https://api.buffer.com"
KINDS = ("earnings", "product", "founding", "recap")
# kind -> (weekday, hour, minute) of the post, ET
POST_AT = {"earnings": (0, 8, 30), "product": (2, 12, 0), "founding": (4, 12, 0), "recap": (5, 10, 0)}
MAX_ATTEMPTS = 3
PREVIEW_TTL_S = 600
LEAD = dt.timedelta(hours=2)  # automatic drafts stop this long before the post time
NAMES_PER_DAY = 3
UTM_SOURCE = {"threads": "threads", "linkedin": "linkedin", "instagram": "instagram", "twitter": "x", "x": "x",
              "bluesky": "bluesky", "facebook": "facebook", "mastodon": "mastodon"}
TOPIC = {"earnings": "Earnings"}
HASHTAGS = {"earnings": "#earnings #stocks #stockmarket #investing"}
DEFAULT_HASHTAGS = "#stocks #stockmarket #investing #trading"

PRODUCT_POSTS = (
    {"headline": "The whole U.S. market, ranked.",
     "body": ("HSF scans thousands of U.S. stocks through the day and ranks the setups that stand out, "
              "each with an HSF Score from 0 to 100.\n\n"
              "The score shows how strongly the technical evidence lines up right now. It's not a prediction, "
              "it's a shortlist to start your research."),
     "points": ["Thousands of U.S. stocks, scanned all day", "An HSF Score from 0 to 100", "The reasons behind each score"],
     "cta": "Free to start, no card", "page": "/"},
    {"headline": "Every earnings report, one calendar.",
     "body": ("Every U.S. company reporting this week, biggest names first, with the time of each report "
              "(before the open or after the close) on one calendar."),
     "points": ["Biggest reporters first", "Before open or after close", "The whole week at a glance"],
     "cta": "This week's calendar", "page": "/weekly"},
    {"headline": "Know when a setup changes.",
     "body": ("Set an alert on a stock and HSF tells you when its setup changes, so you don't have to "
              "watch charts all day."),
     "points": ["Alerts on the stocks you follow", "Sent when the setup changes", "No chart-watching"],
     "cta": "Take a look", "page": "/"},
    {"headline": "Why a stock ranks, not just that it does.",
     "body": ("Every stock HSF ranks comes with the reasons behind its score and the risks to watch, "
              "in plain English."),
     "points": ["The reasons behind each score", "The risks to watch", "In plain English"],
     "cta": "See how HSF works", "page": "/how-hsf-works"},
)

# Performance claims never go out (see the module docstring).
_CLAIMS = re.compile(r"\b(win|wins|winning|winners?|beat|beats|beating|guarantee\w*|profit\w*|returns?|"
                     r"outperform\w*|gains?|gained|crush\w*|moon\w*|10x|100x)\b|\d\s*%", re.I)


def enabled() -> bool:
    return os.environ.get("HSF_SOCIAL_DRAFTS_ENABLED", "1").strip() != "0"


def buffer_key() -> str:
    return os.environ.get("BUFFER_API_KEY", "").strip()


def configured() -> bool:
    return bool(buffer_key())


def _et():
    from analytics import market_calendar as mc

    return mc.ET


def week_key(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def post_time(kind: str, week: str) -> dt.datetime:
    """When a week's post is meant to go out (UTC)."""
    from api.public_recap import week_monday

    wd, h, m = POST_AT[kind]
    day = week_monday(week) + dt.timedelta(days=wd)
    return dt.datetime(day.year, day.month, day.day, h, m, tzinfo=_et()).astimezone(UTC)


def next_week_for(kind: str, now: dt.datetime) -> str:
    """The week of this kind's next post (this week's until its post time has passed)."""
    today = now.astimezone(_et()).date()
    monday = today - dt.timedelta(days=today.weekday())
    week = week_key(monday)
    return week if now < post_time(kind, week) else week_key(monday + dt.timedelta(days=7))


def due(kind: str, now: dt.datetime) -> Optional[str]:
    """The week to send now, or None: from 6pm ET the day before the post (recap: 1am ET on
    Saturday, after the recap publishes) until LEAD before the post time. A draft that would
    land later than that is skipped: there's no time to approve it, and the post was likely
    already made by hand (Send now still works)."""
    week = next_week_for(kind, now)
    at = post_time(kind, week).astimezone(_et())
    if kind == "recap":
        start = at.replace(hour=1, minute=0)
    else:
        start = (at - dt.timedelta(days=1)).replace(hour=18, minute=0)
    return week if start.astimezone(UTC) <= now < (at - LEAD).astimezone(UTC) else None


def check_claims(text: str) -> None:
    hit = _CLAIMS.search(re.sub(r"\$[A-Za-z.]+", "", text or ""))  # tickers like $BEAT aren't claims
    if hit:
        raise ValueError(f"performance claim: {hit.group(0)!r}")


def _range(start: dt.date, end: dt.date) -> str:
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}–{end.day}"
    return f"{start.strftime('%b')} {start.day}–{end.strftime('%b')} {end.day}"


# ---- composing --------------------------------------------------------------------------------
def compose_earnings(week: str, base: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    from api import earnings_week as ew
    from api.public_recap import week_monday

    monday = week_monday(week)
    base = base if base is not None else ew.build_base(monday)
    cut = ew.for_plan(base, per_day=NAMES_PER_DAY, cap=0)
    days = []
    for d in cut["days"]:
        if not d["names"]:
            continue
        day = dt.date.fromisoformat(d["date"])
        days.append({"day": day.strftime("%a"), "date": d["date"], "total": d["total"],
                     "names": [{"ticker": n["ticker"], "time": n["time"]} for n in d["names"]]})
    if not days:
        return None
    total = sum(d["total"] for d in cut["days"])
    lines = [f"{d['day']}: " + " ".join(f"${n['ticker']}" for n in d["names"]) for d in days]
    body = "Big earnings week. The most-traded names reporting:\n\n" + "\n".join(lines)
    if total >= 50:
        body += f"\n\n{(total // 50) * 50:,}+ companies report this week."
    start, end = dt.date.fromisoformat(cut["start"]), dt.date.fromisoformat(cut["end"])
    return {"kind": "earnings", "week": week, "body": body, "cta": "The week at a glance", "page": "/weekly",
            "image": {"kicker": f"Earnings · {_range(start, end)}", "headline": "Big earnings week.",
                      "sub": "The most-traded names reporting" + (f". {(total // 50) * 50:,}+ companies in all." if total >= 50 else "."),
                      "days": days}}


def compose_product(week: str) -> Dict[str, Any]:
    n = int(week.split("-W")[1])
    p = PRODUCT_POSTS[n % len(PRODUCT_POSTS)]
    return {"kind": "product", "week": week, "body": p["body"], "cta": p["cta"], "page": p["page"],
            "image": {"kicker": "HSFinest.AI", "headline": p["headline"], "points": p["points"]}}


def compose_founding(week: str, status: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if status is None:
        from api.founding import public_status

        status = public_status()
    tiers = [t for t in status.get("tiers") or [] if t.get("on_sale") and int(t.get("remaining") or 0) > 0]
    if not status.get("open") or not tiers:
        return None
    five = all(t.get("five_year") for t in tiers)
    seats = [{"label": t["label"], "price": f"${int(t['price_cents']) // 100:,}", "left": int(t["remaining"]),
              "of": int(t["capacity"])} for t in tiers]
    total = sum(int(t["capacity"]) for t in status.get("tiers") or [])
    lead = f"HSF's Founding {total} is open: {total} numbered memberships, one payment"
    body = lead + (", 5 years." if five else ".") + "\n\n" + "\n".join(
        f"{s['label']} {s['price']}: {s['left']} of {s['of']} left" for s in seats)
    return {"kind": "founding", "week": week, "body": body, "cta": "Seats and terms", "page": "/pricing",
            "image": {"kicker": "Founding membership", "headline": f"The Founding {total}.",
                      "sub": "One payment" + (", 5 years." if five else ".") + " Numbered seats.", "seats": seats}}


def compose_recap(week: str, recap: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if recap is None:
        from api.public_recap import recap as load

        recap = load(week)
    ranked = (recap or {}).get("ranked") or []
    if not ranked:
        return None
    top = ranked[:3]
    start, end = dt.date.fromisoformat(recap["start"]), dt.date.fromisoformat(recap["end"])
    body = (f"What HSF's scanner ranked most this week ({_range(start, end)}): "
            + " ".join(f"${r['ticker']}" for r in top)
            + "\n\nEvery week's rankings, recorded as they happened, plus next week's biggest earnings.")
    return {"kind": "recap", "week": week, "body": body, "cta": "The full recap", "page": f"/weekly/{week}",
            "image": {"kicker": f"Weekly recap · {_range(start, end)}", "headline": "Ranked most this week.",
                      "sub": "From HSF's daily scans. A record of rankings, not recommendations.",
                      "names": [{"ticker": r["ticker"], "name": r.get("name"), "days": int(r.get("days") or 0),
                                 "setup": r.get("setup")} for r in top]}}


def compose(kind: str, week: str) -> Optional[Dict[str, Any]]:
    if kind == "earnings":
        return compose_earnings(week)
    if kind == "product":
        return compose_product(week)
    if kind == "founding":
        return compose_founding(week)
    if kind == "recap":
        return compose_recap(week)
    raise ValueError(kind)


def post_for(kind: str, week: str, *, store: bool = False) -> Optional[Dict[str, Any]]:
    """The stored post for that week, else a fresh one (stored when `store`)."""
    from db import social_posts as db

    found = db.stored_post(kind, week)
    if found:
        return found
    if not store:  # previews and images: one composition per 10 minutes
        from api.today import _cached

        return _cached(("social_post", kind, week), lambda: compose(kind, week), ttl_s=PREVIEW_TTL_S)
    post = compose(kind, week)
    if post and store:
        db.store_post(post)
        return db.stored_post(kind, week) or post
    return post


def _base_url() -> str:
    from services.lifecycle_emails import _base_url as base

    return base()


def image_url(post: Dict[str, Any]) -> str:
    return f"{_base_url()}/social/{post['kind']}/{post['week']}.png"


def tagged(page: str, source: str, kind: str, week: str) -> str:
    q = urlencode({"utm_source": source, "utm_medium": "social",
                   "utm_campaign": f"{kind}-w{week.split('-W')[1]}"})
    return f"{_base_url()}{page}?{q}"


def channel_text(post: Dict[str, Any], service: str) -> str:
    service = (service or "").lower()
    if service == "instagram":
        tags = HASHTAGS.get(post["kind"], DEFAULT_HASHTAGS)
        return f"{post['body']}\n\n{post['cta']}: link in bio.\n\n{tags}"
    link = tagged(post["page"], UTM_SOURCE.get(service, service or "social"), post["kind"], post["week"])
    return f"{post['body']}\n\n{post['cta']}:\n{link}"


def preview(kind: str, week: str) -> Optional[Dict[str, Any]]:
    post = post_for(kind, week)
    if not post:
        return None
    texts = {s: channel_text(post, s) for s in ("threads", "linkedin", "instagram")}
    try:
        for t in texts.values():
            check_claims(t)
        problem = None
    except ValueError as e:
        problem = str(e)
    return {"kind": kind, "week": week, "post_at": post_time(kind, week).isoformat(), "image_url": image_url(post),
            "texts": texts, "problem": problem}


# ---- Buffer -----------------------------------------------------------------------------------
class BufferError(RuntimeError):
    pass


def _gql(query: str, variables: Optional[Dict[str, Any]] = None, *, session=None) -> Dict[str, Any]:
    if session is None:
        import httpx as session  # in api/requirements.txt
    res = session.post(BUFFER_API, json={"query": query, "variables": variables or {}}, timeout=20,
                       headers={"Authorization": f"Bearer {buffer_key()}", "Content-Type": "application/json"})
    if res.status_code >= 300:
        raise BufferError(f"HTTP {res.status_code}")
    data = res.json()
    if data.get("errors"):
        raise BufferError(str(data["errors"][0].get("message") or "error")[:200])
    return data.get("data") or {}


def channels(*, session=None) -> List[Dict[str, Any]]:
    """Every Buffer channel on the account: {id, name, service}."""
    orgs = (_gql("query { account { organizations { id } } }", session=session).get("account") or {}).get("organizations") or []
    wanted = {s.strip().lower() for s in os.environ.get("HSF_SOCIAL_CHANNELS", "").split(",") if s.strip()}
    out = []
    for org in orgs:
        q = "query { channels(input: { organizationId: %s }) { id name service } }" % json.dumps(str(org["id"]))
        found = _gql(q, session=session).get("channels") or []
        for c in found:
            service = str(c.get("service") or "").lower()
            if not wanted or service in wanted:
                out.append({"id": str(c["id"]), "name": c.get("name") or "", "service": service})
    return out


CREATE = """mutation($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id } }
    ... on MutationError { message }
  }
}"""


def draft_input(post: Dict[str, Any], channel: Dict[str, Any], due_at: dt.datetime) -> Dict[str, Any]:
    service = channel["service"]
    data: Dict[str, Any] = {"text": channel_text(post, service), "channelId": channel["id"],
                            "schedulingType": "automatic", "mode": "customScheduled",
                            "dueAt": due_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                            "saveToDraft": True, "assets": [{"image": {"url": image_url(post)}}]}
    if service == "instagram":
        data["metadata"] = {"instagram": {"type": "post", "shouldShareToFeed": True}}
    elif service == "threads":
        data["metadata"] = {"threads": {"type": "post", "topic": TOPIC.get(post["kind"], "Stocks")}}
    return data


def create_draft(post: Dict[str, Any], channel: Dict[str, Any], due_at: dt.datetime, *, session=None) -> str:
    out = (_gql(CREATE, {"input": draft_input(post, channel, due_at)}, session=session).get("createPost") or {})
    if out.get("post") and out["post"].get("id"):
        return str(out["post"]["id"])
    raise BufferError(str(out.get("message") or out.get("__typename") or "no post id")[:200])


def send(kind: str, week: str, *, now: Optional[dt.datetime] = None, session=None) -> Dict[str, Any]:
    """Send one week's post to every channel that doesn't have its draft yet. Raises when a
    channel failed and can still be retried, so api/ops_jobs tries again on its next pass."""
    from db import social_posts as db

    now = now or dt.datetime.now(UTC)
    if not configured():
        return {"sent": 0, "skipped": "no_buffer_key"}
    post = post_for(kind, week, store=True)
    if not post:
        return {"sent": 0, "skipped": "nothing_to_post"}
    try:
        for s in ("threads", "linkedin", "instagram"):
            check_claims(channel_text(post, s))
    except ValueError as e:
        log.warning(json.dumps({"event": "social_post_refused", "kind": kind, "week": week, "why": str(e)}))
        return {"sent": 0, "skipped": "claim_check", "why": str(e)}
    chans = channels(session=session)
    if not chans:
        return {"sent": 0, "skipped": "no_channels"}
    done = db.drafts_for(kind, week)
    due_at = max(post_time(kind, week), now + dt.timedelta(minutes=30))
    counts: Dict[str, Any] = {"sent": 0, "already": 0, "failed": 0, "gave_up": 0}
    for ch in chans:
        row = done.get(ch["id"]) or {}
        if row.get("status") == "drafted":
            counts["already"] += 1
            continue
        if int(row.get("attempts") or 0) >= MAX_ATTEMPTS:
            counts["gave_up"] += 1
            continue
        try:
            db.record(kind, week, ch, buffer_id=create_draft(post, ch, due_at, session=session))
            counts["sent"] += 1
        except Exception as e:
            db.record(kind, week, ch, error=f"{type(e).__name__}: {e}")
            counts["failed"] += 1
            log.warning(json.dumps({"event": "social_draft_failed", "kind": kind, "service": ch["service"],
                                    "error": type(e).__name__}))
    if counts["failed"]:
        raise BufferError(f"{counts['failed']} channel(s) failed")
    return counts


def status() -> Dict[str, Any]:
    """For Admin > Marketing: setup state, the next post of each kind, and recent drafts."""
    from db import social_posts as db

    now = dt.datetime.now(UTC)
    out: Dict[str, Any] = {"configured": configured(), "enabled": enabled(), "channels": [], "channel_error": None,
                           "posts": [], "recent": []}
    if out["configured"]:
        try:
            out["channels"] = [{"name": c["name"], "service": c["service"]} for c in channels()]
        except Exception as e:
            out["channel_error"] = f"{type(e).__name__}: {e}"[:200]
    for kind in KINDS:
        week = next_week_for(kind, now)
        try:
            p = preview(kind, week)
        except Exception:
            p = None
        drafts = db.drafts_for(kind, week)
        out["posts"].append({"kind": kind, "week": week, "post_at": post_time(kind, week).isoformat(),
                             "preview": p, "drafted": sum(1 for r in drafts.values() if r.get("status") == "drafted"),
                             "failed": sum(1 for r in drafts.values() if r.get("status") == "failed")})
    out["recent"] = [{**r, "updated_at": str(r.get("updated_at") or "")} for r in db.recent(20)]
    return out
