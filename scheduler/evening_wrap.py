"""Evening wrap email (Pro+): how today went, sent from the postmarket slot.

Per verified Pro+ user with a watchlist: today's watchlist closing moves, the
alerts that fired for them today, and tomorrow's earnings on their names.
Gated by the same MORNING_DIGEST_ENABLED flag (one switch for daily emails)
and throttled once per day. Deterministic, best-effort, never raises.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

try:
    from ui.monitoring import capture as _capture
except Exception:  # pragma: no cover
    def _capture(exc: BaseException) -> None:
        pass

# Customer addresses never reach logs unmasked (GitHub Actions logs are public).
try:
    from ui.log_privacy import log_id, redact
except Exception:  # pragma: no cover - never print an address if the helper is missing
    def log_id(value) -> str:  # type: ignore[no-redef]
        return "***"

    def redact(value) -> str:  # type: ignore[no-redef]
        return "(details hidden)"

_WRAP_KEY = "evening_wrap"


def _todays_events(user_id: str) -> List[str]:
    """Messages of alerts that fired for this user today (New York date)."""
    try:
        from db.alerts import list_recent_events
        from scheduler.morning_digest import et_now

        now_et = et_now()
        today = now_et.date()
        events = list_recent_events(user_id, limit=25) or []
        seen: List[str] = []
        for ev in events:
            fired = ev.get("fired_at")
            if fired is None or not hasattr(fired, "date"):
                continue
            if getattr(fired, "tzinfo", None) is not None:
                fired = fired.astimezone(now_et.tzinfo)
            if fired.date() != today:
                continue
            msg = str(ev.get("message") or "")
            if msg and msg not in seen:
                seen.append(msg)
        # One short line per alert, overlaps removed (same rules as the alert email).
        from scheduler.alert_email import summarize_fired

        return summarize_fired(seen, max_items=8)
    except Exception:
        return []


def _tomorrows_earnings(tickers: List[str]) -> List[str]:
    try:
        from db.earnings import load_earnings_map
        from scheduler.morning_digest import et_now

        tomorrow = et_now().date() + timedelta(days=1)
        emap = load_earnings_map(tickers) or {}
        return sorted(
            {str(s).upper() for s, d in emap.items() if d == tomorrow and str(s).upper() in tickers}
        )
    except Exception:
        return []


def _earnings_calendar_rows(snap_df) -> tuple:
    """(today, next trading day, week calendar, scored rows) for tonight's and the next
    morning's reporters, shared by every recipient. Each reporter is scored once."""
    from scheduler import earnings_email as ee
    from scheduler.morning_digest import _symbol_column, et_now

    today = et_now().date()
    nxt = ee.next_trading_day(today)
    week = ee.reporters(today, today + timedelta(days=7))
    window = [r for r in week
              if (r["date"] == today and r["time"] == "amc") or (r["date"] == nxt and r["time"] != "amc")]
    ranked: List[str] = []
    try:
        col = _symbol_column(snap_df) if snap_df is not None else None
        if col:
            ranked = [str(t) for t in snap_df[col].tolist()]
    except Exception:
        ranked = []
    return today, nxt, week, ee.build_rows(window, ranked=ranked)


def _earnings_section(earn, tickers: List[str]) -> tuple:
    """(html, text) "Earnings tonight & tomorrow morning" for one recipient; ('', '') on error."""
    if not earn:
        return "", ""
    try:
        from scheduler import earnings_email as ee

        today, nxt, week, rows = earn
        day = "Tomorrow" if nxt == today + timedelta(days=1) else nxt.strftime("%A")
        word = "tomorrow" if day == "Tomorrow" else day
        return ee.compose_section(
            rows, title=f"Earnings tonight & {word} morning",
            order=[(today, "amc", "🌙 Tonight after the close"),
                   (nxt, "bmo", f"☀️ {day} before the open ({nxt:%a})"),
                   (nxt, None, f"🕒 {day}, time not announced")],
            price_label="Close", pct_label="Today %",
            link=ee.scanner_link("tomorrow" if day == "Tomorrow" else "week"),
            link_label=f"See {word} in the Earnings scanner" if day == "Tomorrow" else "See the week in the Earnings scanner",
            watched=tickers, watchlist_line=ee.watchlist_line(tickers, week, today),
            footnote="Scores from today's scan of every reporter. Educational only, not financial advice.",
            sub_suffix="scored after the close",
        )
    except Exception:
        return "", ""


def _pct_html(chg) -> str:
    """Colored signed percent for email HTML."""
    try:
        c = float(chg)
    except (TypeError, ValueError):
        return "—"
    color = "#16a34a" if c >= 0 else "#dc2626"
    return f"<span style='color:{color}'>{c:+.2f}%</span>"


def _market_close_context() -> List[tuple]:
    """[(label, last, chg_pct)] for the index proxies — the day's backdrop."""
    try:
        from market_data import build_day_trader_metrics

        rows = build_day_trader_metrics(["SPY", "QQQ"], with_rvol=False) or []
        by = {str(r.get("ticker")).upper(): r for r in rows}
        out = []
        for sym, label in (("SPY", "S&P 500 (SPY)"), ("QQQ", "NASDAQ 100 (QQQ)")):
            r = by.get(sym)
            if r and r.get("last") is not None:
                out.append((label, r.get("last"), r.get("chg_pct")))
        return out
    except Exception:
        return []


def _watchlist_summary(watch_rows: List[Dict[str, Any]]) -> str | None:
    """One-line recap: best/worst performer + up/down breadth."""
    pairs = [
        (str(r.get("ticker")).upper(), float(r["chg_pct"]))
        for r in watch_rows
        if r.get("chg_pct") is not None
    ]
    if len(pairs) < 2:   # one mover is both "best" and "worst"
        return None
    ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
    best, worst = ranked[0], ranked[-1]
    up = sum(1 for _t, c in pairs if c >= 0)
    down = len(pairs) - up
    return (
        f"Best {best[0]} {best[1]:+.1f}% · Worst {worst[0]} {worst[1]:+.1f}% · "
        f"{up} up / {down} down"
    )


def _day_movers(df, limit: int = 5) -> tuple:
    """(gainers, losers) as [(ticker, chg_pct)] from the snapshot's PctChange."""
    try:
        import pandas as pd

        from scheduler.morning_digest import _symbol_column

        sym_col = _symbol_column(df) if df is not None else None
        if df is None or not sym_col or "PctChange" not in df.columns:
            return [], []
        work = df.assign(_c=pd.to_numeric(df["PctChange"], errors="coerce")).dropna(subset=["_c"])
        gain = work.sort_values("_c", ascending=False).head(limit)
        lose = work.sort_values("_c", ascending=True).head(limit)
        gainers = [(str(r[sym_col]).upper(), float(r["_c"])) for _, r in gain.iterrows() if float(r["_c"]) > 0]
        losers = [(str(r[sym_col]).upper(), float(r["_c"])) for _, r in lose.iterrows() if float(r["_c"]) < 0]
        return gainers, losers
    except Exception:
        return [], []


def _tomorrow_setups(df, limit: int = 6) -> tuple:
    """(golden_crosses, top_scores) to watch tomorrow, from today's snapshot."""
    try:
        import pandas as pd

        from scheduler.morning_digest import _symbol_column

        sym_col = _symbol_column(df) if df is not None else None
        if df is None or not sym_col:
            return [], []
        golden: List[str] = []
        if "EMACross" in df.columns:
            gc = df[df["EMACross"].astype(str).str.lower() == "golden"]
            golden = [str(r[sym_col]).upper() for _, r in gc.head(limit).iterrows()]
        top: List[tuple] = []
        if "BreakoutScore" in df.columns:
            w = df.assign(_s=pd.to_numeric(df["BreakoutScore"], errors="coerce")).dropna(subset=["_s"])
            w = w.sort_values("_s", ascending=False).head(limit)
            top = [(str(r[sym_col]).upper(), round(float(r["_s"]), 1)) for _, r in w.iterrows()]
        return golden, top
    except Exception:
        return [], []


def _compose_wrap(
    watch_rows: List[Dict[str, Any]],
    fired_today: List[str],
    earnings_tomorrow: List[str],
    *,
    market_close: List[tuple] | None = None,
    day_gainers: List[tuple] | None = None,
    day_losers: List[tuple] | None = None,
    golden_crosses: List[str] | None = None,
    top_setups: List[tuple] | None = None,
    earnings_section: tuple | None = None,
) -> tuple:
    """(html_inner, text_inner): email_kit cards; the shell adds header + At a glance."""
    import html as _html

    from scheduler.morning_digest import WATCHLIST_ROWS, _app_url, _movers_table, _movers_text, et_now
    from ui import email_kit as k

    date_s = et_now().strftime("%A, %b %d")
    html: List[str] = []
    text = [f"Evening wrap · {date_s}", ""]

    # 🏦 Market close — the day's backdrop
    if market_close:
        parts_text = [f"{label} {last:,.2f} {chg:+.2f}%" for label, last, chg in market_close]
        rows = [[f"<strong>{_html.escape(label)}</strong>", k.num(last), _pct_html(chg)]
                for label, last, chg in market_close]
        html.append(k.card("🏦 Market close", k.table(["Index", "Close", "Day %"], rows)))
        text += ["Market close:", "  " + " · ".join(parts_text), ""]

    # 📋 Your watchlist today (+ one-line summary)
    summary = _watchlist_summary(watch_rows)
    html.append(k.card("📋 Your watchlist today",
                       _movers_table(watch_rows, show_gap=False, limit=WATCHLIST_ROWS)
                       + (k.note(f"📌 {_html.escape(summary)}") if summary else "")))
    text += ["Your watchlist today:", _movers_text(watch_rows)]
    if summary:
        text += [f"📌 {summary}"]
    text += [""]

    # 📊 Today's market movers (beyond the watchlist)
    if day_gainers or day_losers:
        body = ""
        if day_gainers:
            body += k.group("Gainers") + k.chips((t, _pct_html(c)) for t, c in day_gainers)
            text += ["Top gainers: " + ", ".join(f"{t} {c:+.1f}%" for t, c in day_gainers)]
        if day_losers:
            body += k.group("Losers") + k.chips((t, _pct_html(c)) for t, c in day_losers)
            text += ["Top losers: " + ", ".join(f"{t} {c:+.1f}%" for t, c in day_losers)]
        html.append(k.card("📊 Today's market movers", body, sub="Biggest moves in today's market scan"))
        text += [""]

    # 🎯 Tomorrow's setups — forward-looking watchlist for the open
    if golden_crosses or top_setups:
        body = ""
        if top_setups:
            body += k.group("🚀 Top breakout scores") + k.chips((t, k.chip_score(s)) for t, s in top_setups)
        if golden_crosses:
            names = ", ".join(golden_crosses)
            body += k.group("📈 Fresh EMA 9/21 golden crosses") + k.chips((g, None) for g in golden_crosses)
            text += [f"Fresh golden crosses: {names}"]
        if top_setups:
            text += ["Top breakout scores: " + ", ".join(f"{t} ({s:g})" for t, s in top_setups)]
        html.append(k.card("🎯 Tomorrow's setups", body
                           + k.note("Educational only, not financial advice. Confirm setups yourself at the open.")
                           + k.button(_app_url("/today"), "Open Today in HSF →")))
        text += [""]

    if fired_today:
        items = "".join(f"<li style='margin:3px 0'>{_html.escape(m)}</li>" for m in fired_today)
        html.append(k.card("🔔 Alerts that fired today",
                           f"<ul style='margin:0;padding-left:18px;font-size:14px'>{items}</ul>"))
        text += ["Alerts that fired today:"] + [f"  - {m}" for m in fired_today] + [""]

    if earnings_section and earnings_section[0]:
        html.append(earnings_section[0])
        text.append(earnings_section[1])
    elif earnings_tomorrow:
        names = ", ".join(earnings_tomorrow)
        html.append(k.card("📅 Earnings tomorrow (your watchlist)", f"<p style='margin:0'>{_html.escape(names)} ⚠️</p>"))
        text += ["Earnings tomorrow (your watchlist):", f"  {names}", ""]

    return "".join(html), "\n".join(text)


def wrap_glance(watch_rows: List[Dict[str, Any]], *, market_close: List[tuple] | None = None,
                fired_count: int = 0, earnings_count: int = 0,
                top_setups: List[tuple] | None = None) -> str:
    """'At a glance' for the evening wrap: the market, your watchlist, alerts, earnings tonight."""
    from ui import email_kit as k

    parts: List[str] = []
    if market_close:
        bits = []
        for label, _last, chg in market_close:
            short = label.split(" (")[0]
            try:
                bits.append(f"{k.esc(short)} {float(chg):+.1f}%")
            except (TypeError, ValueError):
                continue
        if bits:
            parts.append("Market: " + ", ".join(bits) + ".")
    movers = [r for r in watch_rows if r.get("chg_pct") is not None]
    if movers:
        big = max(movers, key=lambda r: abs(r["chg_pct"]))
        up = sum(1 for r in movers if r["chg_pct"] >= 0)
        name = k.esc(k.split_flag(big["ticker"])[0])
        parts.append(f"Your watchlist: {up} up, {len(movers) - up} down; biggest move "
                     f"<b style='color:#fff'>{name}</b> {float(big['chg_pct']):+.1f}%.")
    if fired_count:
        parts.append(f"{fired_count} alert{'s' if fired_count != 1 else ''} fired today.")
    if earnings_count:
        parts.append(f"<b style='color:#fff'>{earnings_count} compan{'y' if earnings_count == 1 else 'ies'}</b> "
                     "report tonight or before tomorrow's open.")
    if top_setups:
        t, sc = top_setups[0]
        parts.append(f"Top setup for tomorrow: <b style='color:#fff'>{k.esc(t)}</b> (score {sc:g}).")
    return k.glance_text(parts)


def run_evening_wrap(force: bool = False) -> None:
    """Send the evening wrap to eligible Pro+ users (once/day)."""
    try:
        from config import MORNING_DIGEST_ENABLED, MORNING_DIGEST_MAX_USERS
    except Exception:
        return
    if not MORNING_DIGEST_ENABLED:
        return

    # Once per ET day, and only from 16:00 ET (see scheduler.morning_digest).
    from scheduler.morning_digest import WRAP_START_ET, already_sent, daily_send_key, et_now, mark_sent

    send_key = daily_send_key(_WRAP_KEY)
    if not force:
        et = et_now()
        if et.hour < WRAP_START_ET:
            print(f"[evening_wrap] before the evening window ({et:%H:%M} ET); skipping")
            return
        if already_sent(send_key):
            print("[evening_wrap] already sent today; skipping")
            return

    try:
        from auth.tiering import get_user_tier, has_min_tier
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from market_data import build_day_trader_metrics
        from ui.email_utils import send_digest_email
    except Exception as e:
        print(f"[evening_wrap] import failed: {e}")
        return

    try:
        users = load_users() or {}
    except Exception:
        users = {}

    # Market-wide context computed once for all recipients (not per user).
    market_close = _market_close_context()
    try:
        from scheduler.morning_digest import _latest_snapshot_df

        snap_df = _latest_snapshot_df()
    except Exception:
        snap_df = None
    day_gainers, day_losers = _day_movers(snap_df)
    golden_crosses, top_setups = _tomorrow_setups(snap_df)
    try:
        earn = _earnings_calendar_rows(snap_df)
    except Exception as e:
        print(f"[evening_wrap] earnings section unavailable: {type(e).__name__}")
        earn = None

    from scheduler.morning_digest import (
        _email_tier_key,
        default_watchlist_tickers,
        email_opted_in,
        record_email_job,
        unsubscribe_link,
    )

    sent = 0
    skipped: Dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for username in list(users.keys())[:MORNING_DIGEST_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            _skip("not_email")
            continue
        try:  # same eligibility as the morning digest (admin accounts included)
            if not has_min_tier(_email_tier_key(email, users.get(username), users, get_user_tier), "pro"):
                _skip("plan_below_pro")
                continue
            from db.email_verification import is_email_verified

            if not is_email_verified(email):
                _skip("unverified")
                continue
        except Exception:
            _skip("plan_lookup_failed")
            continue
        if not email_opted_in(email, "evening"):
            _skip("unsubscribed")
            continue

        try:
            tickers = default_watchlist_tickers(email, list_watchlists, get_watchlist_tickers)
            if not tickers:
                _skip("empty_watchlist")
                continue

            watch_rows = build_day_trader_metrics(tickers, with_rvol=False)
            fired = _todays_events(email)
            html_inner, text_inner = _compose_wrap(
                watch_rows, fired, _tomorrows_earnings(tickers),
                market_close=market_close,
                day_gainers=day_gainers, day_losers=day_losers,
                golden_crosses=golden_crosses, top_setups=top_setups,
                earnings_section=_earnings_section(earn, tickers),
            )
            if send_digest_email(
                to_address=email,
                subject="Your evening market wrap",
                html_inner=html_inner,
                text_inner=text_inner,
                unsubscribe_url=unsubscribe_link(email, "evening"),
                title_line=f"Evening wrap · {et_now():%A, %b %-d} · after the close",
                glance_html=wrap_glance(watch_rows, market_close=market_close, fired_count=len(fired),
                                        earnings_count=len(earn[3]) if earn else 0, top_setups=top_setups),
            ):
                sent += 1
            else:
                _skip("send_failed")
        except Exception as e:
            print(f"[evening_wrap] {log_id(email)}: {redact(e)}")
            _capture(e)
            _skip("error")
            continue

    if sent > 0:
        mark_sent(send_key)
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())) or "none"
    print(f"[evening_wrap] sent {sent} wrap(s); skipped: {reasons}")
    record_email_job("evening", {"sent": sent, "skipped": skipped})
