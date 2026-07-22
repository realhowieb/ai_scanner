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

_WRAP_KEY = "evening_wrap"


def _todays_events(user_id: str) -> List[str]:
    """Messages of alerts that fired for this user today (UTC)."""
    try:
        from db.alerts import list_recent_events

        today = datetime.now(timezone.utc).date()
        events = list_recent_events(user_id, limit=25) or []
        seen: List[str] = []
        for ev in events:
            fired = ev.get("fired_at")
            if fired is None or not hasattr(fired, "date") or fired.date() != today:
                continue
            msg = str(ev.get("message") or "")
            if msg and msg not in seen:
                seen.append(msg)
        return seen[:8]
    except Exception:
        return []


def _tomorrows_earnings(tickers: List[str]) -> List[str]:
    try:
        from db.earnings import load_earnings_map

        tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
        emap = load_earnings_map(tickers) or {}
        return sorted(
            {str(s).upper() for s, d in emap.items() if d == tomorrow and str(s).upper() in tickers}
        )
    except Exception:
        return []


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
    if not pairs:
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
) -> tuple:
    from scheduler.morning_digest import _movers_table, _movers_text

    date_s = datetime.now(timezone.utc).strftime("%A, %b %d")
    html = [f"<p style='color:#666;margin:0 0 12px'>Evening wrap · {date_s}</p>"]
    text = [f"Evening wrap · {date_s}", ""]

    # 🏦 Market close — the day's backdrop
    if market_close:
        parts_html = [f"{label} {last:,.2f} {_pct_html(chg)}" for label, last, chg in market_close]
        parts_text = [f"{label} {last:,.2f} {chg:+.2f}%" for label, last, chg in market_close]
        html.append(
            "<h3 style='margin:16px 0 6px'>🏦 Market close</h3>"
            f"<p>{' &nbsp;·&nbsp; '.join(parts_html)}</p>"
        )
        text += ["Market close:", "  " + " · ".join(parts_text), ""]

    # 📋 Your watchlist today (+ one-line summary)
    html.append("<h3 style='margin:16px 0 6px'>📋 Your watchlist today</h3>")
    html.append(_movers_table(watch_rows, show_gap=False))
    summary = _watchlist_summary(watch_rows)
    if summary:
        html.append(f"<p style='color:#666;margin:4px 0 0'>📌 {summary}</p>")
    text += ["Your watchlist today:", _movers_text(watch_rows)]
    if summary:
        text += [f"📌 {summary}"]
    text += [""]

    # 📊 Today's market movers (beyond the watchlist)
    if day_gainers or day_losers:
        html.append("<h3 style='margin:16px 0 6px'>📊 Today's market movers</h3>")
        if day_gainers:
            g = ", ".join(f"{t} {_pct_html(c)}" for t, c in day_gainers)
            html.append(f"<p><b>Gainers:</b> {g}</p>")
            text += ["Top gainers: " + ", ".join(f"{t} {c:+.1f}%" for t, c in day_gainers)]
        if day_losers:
            lo = ", ".join(f"{t} {_pct_html(c)}" for t, c in day_losers)
            html.append(f"<p><b>Losers:</b> {lo}</p>")
            text += ["Top losers: " + ", ".join(f"{t} {c:+.1f}%" for t, c in day_losers)]
        text += [""]

    # 🎯 Tomorrow's setups — forward-looking watchlist for the open
    if golden_crosses or top_setups:
        html.append("<h3 style='margin:16px 0 6px'>🎯 Tomorrow's setups</h3>")
        if golden_crosses:
            names = ", ".join(golden_crosses)
            html.append(f"<p><b>📈 Fresh EMA 9/21 golden crosses:</b> {names}</p>")
            text += [f"Fresh golden crosses: {names}"]
        if top_setups:
            ts = ", ".join(f"{t} ({s:g})" for t, s in top_setups)
            html.append(f"<p><b>🚀 Top breakout scores:</b> {ts}</p>")
            text += ["Top breakout scores: " + ", ".join(f"{t} ({s:g})" for t, s in top_setups)]
        html.append(
            "<p style='color:#888;font-size:12px'>Educational only — not financial advice; "
            "confirm setups yourself at the open.</p>"
        )
        text += [""]

    if fired_today:
        items = "".join(f"<li>{m}</li>" for m in fired_today)
        html.append(
            f"<h3 style='margin:16px 0 6px'>🔔 Alerts that fired today</h3><ul>{items}</ul>"
        )
        text += ["Alerts that fired today:"] + [f"  - {m}" for m in fired_today] + [""]

    if earnings_tomorrow:
        names = ", ".join(earnings_tomorrow)
        html.append(
            "<h3 style='margin:16px 0 6px'>📅 Earnings tomorrow (your watchlist)</h3>"
            f"<p>{names} ⚠️</p>"
        )
        text += ["Earnings tomorrow (your watchlist):", f"  {names}", ""]

    return "".join(html), "\n".join(text)


def run_evening_wrap(force: bool = False) -> None:
    """Send the evening wrap to eligible Pro+ users (once/day)."""
    try:
        from config import MORNING_DIGEST_ENABLED, MORNING_DIGEST_MAX_USERS
    except Exception:
        return
    if not MORNING_DIGEST_ENABLED:
        return

    if not force:
        try:
            from db.earnings import should_refresh_earnings_today

            if not should_refresh_earnings_today(_WRAP_KEY):
                print("[evening_wrap] already sent today; skipping")
                return
        except Exception:
            pass

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

    sent = 0
    for username in list(users.keys())[:MORNING_DIGEST_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            continue
        try:
            if not has_min_tier(get_user_tier(email, users), "pro"):
                continue
            from db.email_verification import is_email_verified

            if not is_email_verified(email):
                continue
        except Exception:
            continue

        try:
            wls = list_watchlists(email) or []
            tickers: List[str] = []
            for wl in wls:
                tickers.extend(get_watchlist_tickers(wl.get("id"), email) or [])
            tickers = sorted({str(t).strip().upper() for t in tickers if t})
            if not tickers:
                continue

            watch_rows = build_day_trader_metrics(tickers, with_rvol=False)
            html_inner, text_inner = _compose_wrap(
                watch_rows, _todays_events(email), _tomorrows_earnings(tickers),
                market_close=market_close,
                day_gainers=day_gainers, day_losers=day_losers,
                golden_crosses=golden_crosses, top_setups=top_setups,
            )
            send_digest_email(
                to_address=email,
                subject="Your evening market wrap",
                html_inner=html_inner,
                text_inner=text_inner,
            )
            sent += 1
        except Exception as e:
            print(f"[evening_wrap] {email}: {e}")
            _capture(e)
            continue

    if sent > 0:
        try:
            from db.earnings import mark_earnings_refreshed_today

            mark_earnings_refreshed_today(_WRAP_KEY)
        except Exception:
            pass
    print(f"[evening_wrap] sent {sent} wrap(s)")
