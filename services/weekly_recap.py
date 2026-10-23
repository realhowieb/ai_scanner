"""Saturday weekly recap email (api/ops_jobs.py runs it once per week, from 14:00 UTC).

For every verified, active account that hasn't switched the "weekly" email off:

    Last week's top picks   the first Scanner run of each trading day, its top 5 by HSF
                            Score, and how they did the next trading day vs SPY (frozen
                            hsf_scanner rows in signal_outcomes, return_1d and
                            benchmark_return_1d; names without a result yet are left out)
    Your watchlist          the default watchlist's biggest movers this week (close before
                            Monday to the latest close)
    Pro and up also get
      Alerts that fired     this week's alert_events, and each stock's move since that day
      Earnings next week    watchlist names reporting Monday to Friday
    Starter gets the first two and one line on what Pro adds, with the pricing link.

The shared parts (top picks, prices) load once per run; each account is claimed in
the lifecycle email log (db.referrals.claim_email, kind "weekly_recap", ref the ISO
week) before sending, so a restart never sends it twice.

Env: HSF_WEEKLY_RECAP_ENABLED=0 stops it (checked in api/ops_jobs.py), and
HSF_WEEKLY_RECAP_MAX_USERS (default 500) caps one run. Best effort: never raises.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("hsf.weekly_recap")

KIND = "weekly_recap"
UTC = dt.timezone.utc
TOP_PER_DAY = 5


def _max_users() -> int:
    try:
        return max(1, int(os.environ.get("HSF_WEEKLY_RECAP_MAX_USERS", "500")))
    except ValueError:
        return 500


def _et():
    from analytics import market_calendar as mc

    return mc.ET


def week_bounds(now: dt.datetime) -> Tuple[dt.date, dt.date]:
    """(Monday, Friday) of the ET week that just ended (or is ending) at `now`."""
    day = now.astimezone(_et()).date()
    monday = day - dt.timedelta(days=day.weekday())
    return monday, monday + dt.timedelta(days=4)


def iso_week(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%"


# ---- shared data ------------------------------------------------------------------------------
def _rows(conn, sql: str, args: tuple) -> List[tuple]:
    cur = conn.cursor()
    try:
        cur.execute(sql, args)
        rows = cur.fetchall() or []
    finally:
        cur.close()
    return [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in rows]


def summarize_picks(rows: Sequence[tuple], tz=None) -> Dict[str, Any]:
    """rows: (ticker, fired_at, setup_score, return_1d, benchmark_return_1d). Keeps the first
    run of each ET day, its top TOP_PER_DAY by score, and the ones with a next-day result."""
    tz = tz or _et()
    first: Dict[dt.date, dt.datetime] = {}
    for _t, fired, *_ in rows:
        d = fired.astimezone(tz).date()
        if d not in first or fired < first[d]:
            first[d] = fired
    by_day: Dict[dt.date, List[tuple]] = {}
    for r in rows:
        if first.get(r[1].astimezone(tz).date()) == r[1] and r[2] is not None:
            by_day.setdefault(r[1].astimezone(tz).date(), []).append(r)
    picks = []
    for d in sorted(by_day):
        for t, _f, score, r1, b1 in sorted(by_day[d], key=lambda x: -float(x[2]))[:TOP_PER_DAY]:
            if r1 is not None:
                picks.append({"day": d, "ticker": str(t).upper(), "score": float(score), "ret": float(r1),
                              "spy": None if b1 is None else float(b1)})
    if not picks:
        return {"n": 0, "days": len(by_day)}
    avg = sum(p["ret"] for p in picks) / len(picks)
    spy_vals = [p["spy"] for p in picks if p["spy"] is not None]
    beat = sum(1 for p in picks if p["spy"] is not None and p["ret"] > p["spy"])
    return {"n": len(picks), "days": len(by_day), "avg": avg,
            "spy": (sum(spy_vals) / len(spy_vals)) if spy_vals else None,
            "beat": beat, "with_spy": len(spy_vals),
            "best": sorted(picks, key=lambda p: -p["ret"])[:3]}


def load_picks(monday: dt.date, friday: dt.date) -> Dict[str, Any]:
    from db.engine import get_neon_conn

    tz = _et()
    start = dt.datetime.combine(monday, dt.time(0, 0), tz)
    end = dt.datetime.combine(friday + dt.timedelta(days=1), dt.time(0, 0), tz)
    conn = get_neon_conn()
    if conn is None:
        return {"n": 0, "days": 0}
    try:
        rows = _rows(conn, "SELECT ticker, fired_at, setup_score, return_1d, benchmark_return_1d FROM signal_outcomes "
                           "WHERE source = 'opportunity' AND signal_type = 'hsf_scanner' AND fired_at >= %s "
                           "AND fired_at < %s", (start, end))
    finally:
        conn.close()
    return summarize_picks([(t, f if f.tzinfo else f.replace(tzinfo=UTC), s, r1, b1) for t, f, s, r1, b1 in rows], tz)


def load_closes(tickers: Sequence[str]) -> Dict[str, List[Tuple[str, float]]]:
    """{ticker: [(ISO day, close)] oldest first} for about a month."""
    syms = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not syms:
        return {}
    from data.prices import fetch_price_data_batch

    frames, _ = fetch_price_data_batch(syms, period="1mo", timeout_s=15.0, batch_retries=1)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for t in syms:
        df = frames.get(t)
        if df is None or getattr(df, "empty", True):
            continue
        col = next((c for c in df.columns if str(c).lower() == "close"), None)
        if col is None:
            continue
        series = []
        for idx, v in zip(df.index, df[col].tolist()):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f > 0 and f == f:
                series.append((idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10], f))
        if series:
            out[t] = series
    return out


def change_since(series: List[Tuple[str, float]], day: dt.date, *, before: bool) -> Optional[float]:
    """Latest close vs the close before `day` (before=True) or on/after it."""
    if not series:
        return None
    key = day.isoformat()
    base = None
    if before:
        for d, c in series:
            if d < key:
                base = c
    else:
        base = next((c for d, c in series if d >= key), None)
    last = series[-1][1]
    return None if not base else last / base - 1.0


# ---- one account ------------------------------------------------------------------------------
def compose(*, pro: bool, picks: Dict[str, Any], movers: List[Tuple[str, float]],
            alerts: List[Tuple[str, Optional[float]]], earnings: List[Tuple[str, str]],
            monday: dt.date, base_url: str) -> tuple:
    """(subject, text, html inner). Pure."""
    from services.lifecycle_emails import _button

    week = f"{monday:%b} {monday.day}"
    subject = f"Your week in review (week of {week})"
    T: List[str] = []
    H: List[str] = [f"<h2 style='font-size:20px;margin:16px 0 8px'>Your week in review</h2>"
                    f"<p style='color:#555;margin:0 0 12px'>Week of {html.escape(week)}</p>"]

    H.append("<h3 style='margin:16px 0 6px'>🏆 Last week's top picks</h3>")
    T.append("Last week's top picks")
    if picks.get("n"):
        spy = f", SPY {_pct(picks['spy'])}" if picks.get("spy") is not None else ""
        line = (f"The Scanner's top {TOP_PER_DAY} at each day's first scan: {picks['n']} picks averaged "
                f"{_pct(picks['avg'])} the next trading day{spy}.")
        if picks.get("with_spy"):
            line += f" {picks['beat']} of {picks['with_spy']} beat SPY."
        best = ", ".join(f"{p['ticker']} {_pct(p['ret'])}" for p in picks["best"])
        T += [line, f"Best: {best}", ""]
        H.append(f"<p>{html.escape(line)}</p><p>Best: <strong>{html.escape(best)}</strong></p>")
    else:
        T += ["No next-day results for last week's picks yet.", ""]
        H.append("<p>No next-day results for last week's picks yet.</p>")

    H.append("<h3 style='margin:16px 0 6px'>📋 Your watchlist this week</h3>")
    T.append("Your watchlist this week")
    if movers:
        T += [f"  {t}: {_pct(c)}" for t, c in movers] + [""]
        H.append("<ul style='padding-left:20px'>" + "".join(
            f"<li><strong>{html.escape(t)}</strong> {_pct(c)}</li>" for t, c in movers) + "</ul>")
    else:
        T += [f"Add stocks to a watchlist and this shows how they moved: {base_url}/watchlists", ""]
        H.append(f"<p>Add stocks to a <a href='{base_url}/watchlists'>watchlist</a> and this shows how they moved.</p>")

    if pro:
        H.append("<h3 style='margin:16px 0 6px'>🔔 Alerts that fired</h3>")
        T.append("Alerts that fired")
        if alerts:
            items = [f"{t} ({_pct(c)} since)" if c is not None else t for t, c in alerts]
            T += ["  " + ", ".join(items), ""]
            H.append(f"<p>{html.escape(', '.join(items))}</p>")
        else:
            T += ["None this week.", ""]
            H.append("<p>None this week.</p>")
        H.append("<h3 style='margin:16px 0 6px'>📅 Earnings next week (your watchlist)</h3>")
        T.append("Earnings next week (your watchlist)")
        if earnings:
            items = [f"{t} {d}" for t, d in earnings]
            T += ["  " + ", ".join(items), ""]
            H.append(f"<p>{html.escape(', '.join(items))}</p>")
        else:
            T += ["None of your names report next week.", ""]
            H.append("<p>None of your names report next week.</p>")
    else:
        T += [f"Pro adds your alerts recap and next week's earnings for your names: {base_url}/pricing", ""]
        H.append("<p style='color:#555'>Pro adds your alerts recap and next week's earnings for your names.</p>"
                 + _button(f"{base_url}/pricing", "See Pro"))
    H.append(_button(f"{base_url}/today", "Open Today"))
    T.append(f"Open Today: {base_url}/today")
    return subject, "\n".join(T), "".join(H)


def _account_parts(email: str, *, pro: bool, closes, monday: dt.date, friday: dt.date,
                   watch: List[str], alert_rows: List[tuple], earnings_rows: List[Dict[str, Any]]):
    movers = []
    for t in watch:
        c = change_since(closes.get(t, []), monday, before=True)
        if c is not None:
            movers.append((t, c))
    movers = sorted(movers, key=lambda x: -abs(x[1]))[:5]
    alerts: List[Tuple[str, Optional[float]]] = []
    if pro:
        seen = set()
        for t, fired in alert_rows:
            t = str(t or "").upper()
            if not t or t in seen:
                continue
            seen.add(t)
            alerts.append((t, change_since(closes.get(t, []), fired.astimezone(_et()).date(), before=False)))
        alerts = alerts[:6]
    names = set(watch)
    earnings = [(r["symbol"], f"{r['earnings_date']:%a}") for r in earnings_rows if r.get("symbol") in names][:8] if pro else []
    return movers, alerts, earnings


def run(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now(UTC)
    counts: Dict[str, Any] = {"sent": 0, "skipped": {}, "failed": 0}

    def skip(why: str) -> None:
        counts["skipped"][why] = counts["skipped"].get(why, 0) + 1

    monday, friday = week_bounds(now)
    week = iso_week(monday)
    try:
        from auth.tiering import get_user_tier, has_min_tier
        from db import referrals as store
        from db.email_prefs import unsubscribe_url, wants_email
        from db.email_verification import is_email_verified
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from scheduler.morning_digest import _email_tier_key, default_watchlist_tickers, record_email_job
        from services.lifecycle_emails import _base_url
        from ui.email_utils import send_digest_email

        users = load_users() or {}
        picks = load_picks(monday, friday)
    except Exception as e:
        log.warning(json.dumps({"event": "weekly_recap_failed", "error": type(e).__name__}))
        return {"error": type(e).__name__}
    base = _base_url()

    # Who gets it, and their lists (before the price batch, so it's one batch).
    todo = []
    for username in list(users)[:_max_users()]:
        email = str(username or "").strip().lower()
        record = users.get(username) or {}
        if "@" not in email or record.get("is_active") is False:
            skip("not_eligible")
            continue
        try:
            if not is_email_verified(email):
                skip("unverified")
                continue
        except Exception:
            pass
        if not wants_email(email, "weekly"):
            skip("unsubscribed")
            continue
        try:
            tier = _email_tier_key(email, record, users, get_user_tier)
            pro = has_min_tier(tier, "pro") or bool(store.grant_until(email))
        except Exception:
            pro = False
        try:
            watch = default_watchlist_tickers(email, list_watchlists, get_watchlist_tickers)
        except Exception:
            watch = []
        todo.append((email, pro, watch))

    alerts_by_user: Dict[str, List[tuple]] = {}
    earnings_rows: List[Dict[str, Any]] = []
    try:
        from db.engine import get_neon_conn

        conn = get_neon_conn()
        if conn is not None:
            try:
                tz = _et()
                start = dt.datetime.combine(monday, dt.time(0, 0), tz)
                for u, t, f in _rows(conn, "SELECT lower(user_id), ticker, fired_at FROM alert_events "
                                           "WHERE fired_at >= %s ORDER BY fired_at DESC LIMIT 5000", (start,)):
                    alerts_by_user.setdefault(u, []).append((t, f if f.tzinfo else f.replace(tzinfo=UTC)))
            finally:
                conn.close()
        from db.earnings import fetch_earnings_this_week

        next_monday = monday + dt.timedelta(days=7)
        earnings_rows = fetch_earnings_this_week(start=next_monday, days_ahead=4)
    except Exception as e:
        log.warning(json.dumps({"event": "weekly_recap_inputs_partial", "error": type(e).__name__}))

    tickers = {t for _e, _p, w in todo for t in w}
    tickers |= {str(t).upper() for e, p, _w in todo if p for t, _f in alerts_by_user.get(e, []) if t}
    try:
        closes = load_closes(sorted(tickers))
    except Exception as e:
        log.warning(json.dumps({"event": "weekly_recap_prices_failed", "error": type(e).__name__}))
        closes = {}

    for email, pro, watch in todo:
        try:
            if not store.claim_email(email, KIND, week):
                skip("already_sent")
                continue
            movers, alerts, earnings = _account_parts(
                email, pro=pro, closes=closes, monday=monday, friday=friday, watch=watch,
                alert_rows=alerts_by_user.get(email, []), earnings_rows=earnings_rows)
            subject, text, inner = compose(pro=pro, picks=picks, movers=movers, alerts=alerts,
                                           earnings=earnings, monday=monday, base_url=base)
            if send_digest_email(email, subject, inner, text, unsubscribe_url=unsubscribe_url(email, "weekly")):
                counts["sent"] += 1
            else:
                store.release_email(email, KIND, week)
                counts["failed"] += 1
        except Exception as e:
            counts["failed"] += 1
            log.warning(json.dumps({"event": "weekly_recap_send_failed", "error": type(e).__name__}))
    try:
        record_email_job("weekly_recap", {"sent": counts["sent"], "skipped": counts["skipped"],
                                          "email_failed": counts["failed"]})
    except Exception:
        pass
    return counts
