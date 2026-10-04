"""Pre-open morning digest email (Pro+, structured, no AI dependency).

For each verified Pro+ user with a watchlist, assemble and send a compact
pre-open email: their watchlist's overnight move, the day's top market gappers,
any of their names reporting earnings today, and one PreBreakout pick from the
latest snapshot. Deterministic (no Claude call) so it can't fail on an AI quota.

Runs headless from the reliable cron (scheduler.cron_runner), throttled to once
per day. Best-effort throughout — it never raises into the caller.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Report silent per-user failures to Sentry when configured (no-op otherwise).
try:
    from ui.monitoring import capture as _capture
except Exception:  # pragma: no cover - fallback when monitoring is unavailable
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

_DIGEST_REFRESH_KEY = "morning_digest"

# "Once a day" means once per New York trading day. Keys are dated in ET, so a
# scheduled run that GitHub delays past UTC midnight (e.g. the 21:10 UTC slot
# firing at 00:27 UTC = 8:27 PM ET) can neither re-send nor claim the next day.
# Each email also has its own ET time window (bypassed by forced manual runs).
DIGEST_WINDOW_ET = (6, 12)   # morning digest: 06:00 <= ET < 12:00
WRAP_START_ET = 16           # evening wrap: from 16:00 ET (same ET day)


def et_now(now=None):
    from zoneinfo import ZoneInfo

    now = now or datetime.now(timezone.utc)
    return now.astimezone(ZoneInfo("America/New_York"))


def daily_send_key(base: str, now=None) -> str:
    """e.g. 'morning_digest:2026-09-29' — one record per ET day."""
    return f"{base}:{et_now(now).date().isoformat()}"


def already_sent(key: str) -> bool:
    """True when this ET-dated key was recorded (any time that ET day). On a
    database error returns False, matching the old throttle (send rather than drop)."""
    try:
        from db.earnings import _get_conn, ensure_earnings_refresh_log_table

        conn = _get_conn(None)
        ensure_earnings_refresh_log_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM earnings_refresh_log WHERE refresh_key = %s;", (key,))
            return cur.fetchone() is not None
    except Exception:
        return False


def mark_sent(key: str) -> None:
    try:
        from db.earnings import mark_earnings_refreshed_today

        mark_earnings_refreshed_today(key)
    except Exception:
        pass


def _latest_snapshot_df():
    """Load the most recent meaningful scan as a DataFrame, or None.

    Prefers a real daily snapshot, then the most recent run that actually
    produced results. The cron's session scans can log 0-row premarket/postmarket
    runs, and picking the newest run blindly (or a snapshot that loads empty)
    would blank the whole digest — so we load candidates in priority order and
    return the first that yields a non-empty DataFrame.
    """
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_runs(limit=25, all_users=True) or []
        if not runs:
            return None
        # Snapshots first (newest→oldest), then any run reporting results, then
        # the newest run as a last resort.
        candidates = [r for r in runs if r.get("is_snapshot")]
        candidates += [
            r for r in runs
            if not r.get("is_snapshot") and (r.get("row_count") or 0) > 0
        ]
        candidates.append(runs[0])

        seen: set = set()
        for snap in candidates:
            rid = snap.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            raw = load_run_results(rid)
            df = normalize_results_to_df(raw) if raw else None
            if df is not None and len(df) > 0:
                return df
        return None
    except Exception:
        return None


def _symbol_column(df) -> Optional[str]:
    for col in ("Symbol", "Ticker", "symbol", "ticker"):
        if col in df.columns:
            return col
    return None


def _prebreakout_picks(df, limit: int = 3) -> List[Dict[str, Any]]:
    """Top PreBreakout candidates [{symbol, prob}] from the snapshot (best first)."""
    try:
        from ml_prebreakout import score_prebreakout

        scored = score_prebreakout(df)
        if scored is None or len(scored) == 0 or "PreBreakoutProb%" not in scored.columns:
            return []
        sym_col = _symbol_column(scored)
        if not sym_col:
            return []
        # Calibrated % is the primary rank; the raw model probability breaks ties
        # within the calibrated floor (matches the Scanner ranking) so the picks
        # are ordered by real discrimination, not arbitrary row order.
        sort_cols = ["PreBreakoutProb%"]
        if "PreBreakoutProbRaw" in scored.columns:
            sort_cols.append("PreBreakoutProbRaw")
        top = scored.sort_values(sort_cols, ascending=False).head(limit)
        price_col = next((c for c in ("Last", "last", "Close", "Price") if c in scored.columns), None)
        out: List[Dict[str, Any]] = []
        for _, r in top.iterrows():
            prob = float(r.get("PreBreakoutProb%") or 0.0)
            if prob <= 0:
                continue
            pick = {"symbol": str(r.get(sym_col)).upper(), "prob": round(prob, 1)}
            if price_col is not None:
                try:
                    pick["last"] = round(float(r.get(price_col)), 2)
                except (TypeError, ValueError):
                    pass
            out.append(pick)
        return out
    except Exception:
        return []


def _todays_setups(df, limit: int = 6) -> tuple:
    """(golden_crosses, top_breakout_scores) for the open, from the latest snapshot.

    Reuses the evening wrap's setup extractor so the two emails stay consistent.
    """
    try:
        from scheduler.evening_wrap import _tomorrow_setups

        return _tomorrow_setups(df, limit=limit)
    except Exception:
        return [], []


def _market_gappers(df, limit: int = 5) -> List[Dict[str, Any]]:
    """Top gappers over the snapshot universe.

    Prefer live premarket snapshots (freshest overnight move), but the snapshot
    universe is often low-liquidity names Alpaca has no premarket quote for, so
    that path can come back empty. Fall back to the snapshot's own stored GapPct
    (computed at scan time) so the digest's gappers section is never blank.
    """
    try:
        from market_data import build_day_trader_metrics

        sym_col = _symbol_column(df) if df is not None else None
        if df is not None and sym_col:
            symbols = [str(s).upper() for s in df[sym_col].tolist() if str(s).strip()][:60]
            rows = build_day_trader_metrics(symbols, with_rvol=False)
            rows = [r for r in rows if r.get("gap_pct") is not None]
            if rows:
                rows.sort(key=lambda r: abs(r["gap_pct"]), reverse=True)
                return rows[:limit]
    except Exception:
        pass
    return _gappers_from_snapshot(df, limit)


def _gappers_from_snapshot(df, limit: int = 5) -> List[Dict[str, Any]]:
    """Gappers from the snapshot's stored GapPct/Last/PctChange columns."""
    try:
        sym_col = _symbol_column(df) if df is not None else None
        if df is None or not sym_col or "GapPct" not in df.columns:
            return []

        def _f(row, col):
            try:
                v = float(row.get(col))
                return v if v == v else None  # drop NaN
            except (TypeError, ValueError):
                return None

        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            gap = _f(r, "GapPct")
            if gap is None:
                continue
            last = _f(r, "Last")
            rows.append({
                "ticker": str(r.get(sym_col) or "").upper(),
                "last": None if last is None else round(last, 2),
                "chg_pct": _f(r, "PctChange"),
                "gap_pct": gap,
            })
        rows.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
        return rows[:limit]
    except Exception:
        return []


def _earnings_days_map(symbols: List[str], flag_days: int = 5) -> Dict[str, int]:
    """{symbol: days-until-earnings} for names reporting within flag_days.

    DB-only (earnings_calendar); best-effort empty map on any failure.
    """
    try:
        from db.earnings import load_earnings_map

        syms = sorted({str(s).upper() for s in symbols if str(s).strip()})
        if not syms:
            return {}
        emap = load_earnings_map(syms)
        today = et_now().date()
        out: Dict[str, int] = {}
        for sym, edate in emap.items():
            if edate is None:
                continue
            days = (edate - today).days
            if 0 <= days <= flag_days:
                out[str(sym).upper()] = days
        return out
    except Exception:
        return {}


def _flag_earnings_rows(rows: List[Dict[str, Any]], edays: Dict[str, int]) -> None:
    """Append '⚠️E{n}d' to each row's ticker when earnings are imminent (in place)."""
    for r in rows:
        days = edays.get(str(r.get("ticker") or "").upper())
        if days is not None:
            r["ticker"] = f"{r['ticker']} ⚠️E{days}d"


def _earnings_today() -> set:
    """Set of symbols reporting earnings today (UTC)."""
    try:
        from db.earnings import fetch_earnings_this_week

        today = et_now().date()
        rows = fetch_earnings_this_week(days_ahead=1) or []
        return {
            str(r.get("symbol")).upper()
            for r in rows
            if r.get("symbol") and r.get("earnings_date") == today
        }
    except Exception:
        return set()


def _premarket_movers(now=None, limit: int = 5) -> List[Dict[str, Any]]:
    """P2-74: biggest pre-market moves from this morning's premarket scan
    ({ticker, pct, last, score}), the same list as Today's Before the open card.
    Empty before that scan exists, after the open, or on any error."""
    try:
        import datetime as _dt

        from db.runs import list_runs
        from ui.before_open import current_premarket_run, premarket_movers
        from ui.market_scans import _run_df_uncached

        now = now or _dt.datetime.now(_dt.timezone.utc)
        run = current_premarket_run(list_runs(limit=60, include_snapshots=False, username="scheduler") or [], now)
        if run is None:
            return []
        return premarket_movers(_run_df_uncached(int(run["id"])), n=limit)
    except Exception:
        return []


def _premarket_section(rows: List[Dict[str, Any]]) -> tuple[str, str]:
    """(html, text) for the pre-market movers section, or ('', '') when empty."""
    if not rows:
        return "", ""
    items, lines = [], []
    for r in rows:
        color = "#16a34a" if r["pct"] >= 0 else "#dc2626"
        price = f" · ${r['last']:,.2f}" if r.get("last") is not None else ""
        score = f" · HSF Score {r['score']}" if r.get("score") is not None else ""
        items.append(f"<li><strong>{r['ticker']}</strong> "
                     f"<span style='color:{color}'>{r['pct']:+.2f}%</span>{price}{score}</li>")
        lines.append(f"  {r['ticker']} {r['pct']:+.2f}%{price}{score}")
    html = ("<h3 style='margin:16px 0 6px'>🌅 Pre-market movers</h3>"
            f"<ul style='margin:4px 0 0;padding-left:18px'>{''.join(items)}</ul>"
            "<p style='margin:4px 0 0;font-size:12px;color:#888'>From the 8:35 AM ET "
            "pre-market scan, vs the previous close. Pre-market prices keep moving.</p>")
    return html, "\n".join(["Pre-market movers (8:35 AM ET scan, vs previous close):", *lines, ""])


def _movers_table(rows: List[Dict[str, Any]], *, show_gap: bool = True) -> str:
    if not rows:
        return "<p style='color:#888'>No data.</p>"
    head = "<tr><th align='left'>Ticker</th><th align='right'>Last</th><th align='right'>Chg %</th><th></th>"
    if show_gap:
        head += "<th align='right'>Gap %</th>"
    head += "</tr>"
    body = ""
    for r in rows:
        chg = r.get("chg_pct")
        gap = r.get("gap_pct")
        color = "#16a34a" if (chg is not None and chg >= 0) else "#dc2626"
        # Email-safe inline bar-let: width scales with |move| (capped at 5%).
        bar = ""
        if chg is not None:
            width = int(min(abs(chg) / 5.0, 1.0) * 48) + 2
            bar = (
                f"<div style='height:8px;width:{width}px;border-radius:3px;"
                f"background:{color};display:inline-block'></div>"
            )
        body += (
            f"<tr><td>{r.get('ticker')}</td>"
            f"<td align='right'>{r.get('last')}</td>"
            f"<td align='right' style='color:{color}'>"
            f"{('%+.2f%%' % chg) if chg is not None else '—'}</td>"
            f"<td align='left'>{bar}</td>"
        )
        if show_gap:
            body += f"<td align='right'>{('%+.2f%%' % gap) if gap is not None else '—'}</td>"
        body += "</tr>"
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:13px' "
        "cellpadding='4'>" + head + body + "</table>"
    )


def _movers_text(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(
        f"  {r.get('ticker')}: {r.get('last')} "
        f"({('%+.2f%%' % r['chg_pct']) if r.get('chg_pct') is not None else '—'})"
        for r in rows
    )


def _track_record_line() -> tuple[str, str]:
    """Return (html, text) one-liner for the signal track record, or ('','')."""
    try:
        from db.track_record import load_latest_track_record

        tr = load_latest_track_record(horizon_days=5)
        if not tr or not tr.get("sample_size") or tr.get("avg_return") is None:
            return "", ""
        # Only show once the sample is statistically meaningful (see UI gate).
        if int(tr.get("sample_size") or 0) < 150 or int(tr.get("runs_used") or 0) < 8:
            return "", ""
        avg = tr["avg_return"]
        win = tr.get("win_rate") or 0.0
        h = tr.get("horizon_days", 5)
        bench = tr.get("benchmark") or "SPY"
        top_n = tr.get("top_n") or 5
        html = (
            f"<p style='color:#166534;background:#f0fdf4;padding:8px 10px;border-radius:6px'>"
            f"📈 <strong>Track record:</strong> top-{top_n} candidates beat the {bench} by "
            f"<strong>{avg:+.1%}</strong> over {h} trading days · {win:.0%} beat the benchmark.</p>"
        )
        text = (
            f"Track record: top-{top_n} candidates beat {bench} by {avg:+.1%} "
            f"over {h} days ({win:.0%} beat the benchmark)."
        )
        return html, text
    except Exception:
        return "", ""


def _watchlist_notes(
    watch_rows: List[Dict[str, Any]], week_earnings: Dict[str, int]
) -> List[str]:
    """Short narrative notes: biggest watchlist mover + who reports this week."""
    notes: List[str] = []
    movers = [r for r in watch_rows if r.get("chg_pct") is not None]
    if movers:
        ranked = sorted(movers, key=lambda r: r["chg_pct"], reverse=True)
        best, worst = ranked[0], ranked[-1]
        up = sum(1 for r in movers if r["chg_pct"] >= 0)
        down = len(movers) - up
        # split off any '⚠️E{n}d' earnings flag so the recap stays clean
        bn = str(best["ticker"]).split(" ")[0]
        wn = str(worst["ticker"]).split(" ")[0]
        notes.append(
            f"Best {bn} {best['chg_pct']:+.1f}% · Worst {wn} {worst['chg_pct']:+.1f}% · "
            f"{up} up / {down} down"
        )
    for sym, days in sorted(week_earnings.items(), key=lambda kv: kv[1]):
        when = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days}d")
        notes.append(f"{sym} reports {when}")
    return notes[:5]


def _compose(
    username: str,
    watch_rows: List[Dict[str, Any]],
    gappers: List[Dict[str, Any]],
    earnings_hits: List[str],
    picks: Optional[List[Dict[str, Any]]],
    notes: Optional[List[str]] = None,
    golden: Optional[List[str]] = None,
    top_setups: Optional[List[tuple]] = None,
    premarket: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, str]:
    """Return (html_inner, text_inner) for one user's digest."""
    date_s = et_now().strftime("%A, %b %d")
    html = [f"<p style='color:#666;margin:0 0 12px'>Morning snapshot · {date_s}</p>"]
    text = [f"Morning snapshot · {date_s}", ""]

    tr_html, tr_text = _track_record_line()
    if tr_html:
        html.append(tr_html)
        text += [tr_text, ""]

    html.append("<h3 style='margin:16px 0 6px'>📋 Your watchlist</h3>")
    html.append(_movers_table(watch_rows, show_gap=True))
    text += ["Your watchlist:", _movers_text(watch_rows), ""]
    if notes:
        joined = " · ".join(notes)
        html.append(f"<p style='color:#555;margin:4px 0 0'>📌 {joined}</p>")
        text += [f"Notes: {joined}", ""]

    html.append("<h3 style='margin:16px 0 6px'>🚀 Top market gappers</h3>")
    html.append(_movers_table(gappers, show_gap=True))
    text += ["Top market gappers:", _movers_text(gappers), ""]

    pm_html, pm_text = _premarket_section(premarket or [])
    if pm_html:
        html.append(pm_html)
        text.append(pm_text)

    if earnings_hits:
        names = ", ".join(sorted(earnings_hits))
        html.append(
            "<h3 style='margin:16px 0 6px'>📅 Earnings today (your watchlist)</h3>"
            f"<p>{names}</p>"
        )
        text += ["Earnings today (your watchlist):", f"  {names}", ""]

    # 🎯 Today's setups — actionable candidates for the open (mirrors the evening
    # wrap's forward-looking section).
    if golden or top_setups:
        html.append("<h3 style='margin:16px 0 6px'>🎯 Today's setups</h3>")
        text += ["Today's setups:"]
        if golden:
            names = ", ".join(golden)
            html.append(f"<p><b>📈 Fresh EMA 9/21 golden crosses:</b> {names}</p>")
            text += [f"  Fresh golden crosses: {names}"]
        if top_setups:
            ts = ", ".join(f"{t} ({s:g})" for t, s in top_setups)
            html.append(f"<p><b>🚀 Top breakout scores:</b> {ts}</p>")
            text += [f"  Top breakout scores: {ts}"]
        html.append(
            "<p style='color:#888;font-size:12px'>Educational only — not financial "
            "advice; confirm setups yourself at the open.</p>"
        )
        text += [""]

    if picks:
        def _pick_price(p):
            return f" &middot; ${p['last']:.2f}" if p.get("last") is not None else ""
        items = "".join(
            f"<li><strong>{p['symbol']}</strong>{_pick_price(p)} — "
            f"{p['prob']:.1f}% PreBreakout likelihood</li>"
            for p in picks
        )
        label = "🧠 PreBreakout picks" if len(picks) > 1 else "🧠 PreBreakout pick"
        html.append(
            f"<h3 style='margin:16px 0 6px'>{label}</h3>"
            f"<ul style='margin:4px 0 0;padding-left:18px'>{items}</ul>"
            "<p style='margin:4px 0 0;font-size:12px;color:#888'>Calibrated model "
            "likelihood of setup follow-through — not a price forecast.</p>"
        )
        text += ["PreBreakout picks:"]
        text += [f"  {p['symbol']}"
                 + (f" ${p['last']:.2f}" if p.get('last') is not None else "")
                 + f" — {p['prob']:.1f}% likelihood" for p in picks]
        text += [""]

    return "".join(html), "\n".join(text)


def record_email_job(job: str, stats: Dict[str, Any]) -> None:
    """P1-36: store this run's counts for the Email delivery health card, and report
    failed sends to Sentry (counts only, never addresses). Never raises."""
    try:  # P1-39: which email settings this environment (GitHub Actions) used
        from ui.email_setup import describe_from_config

        stats = {**stats, "smtp": describe_from_config()}
    except Exception:
        pass
    try:
        from db.email_job_runs import record_email_run

        record_email_run(job, stats)
    except Exception:
        pass
    failed = int((stats.get("skipped") or {}).get("send_failed") or stats.get("email_failed") or 0)
    if failed:
        _capture(RuntimeError(f"{job} email: {failed} send(s) failed this run"))


def default_watchlist_tickers(email: str, list_watchlists, get_watchlist_tickers) -> List[str]:
    """Tickers of the account's DEFAULT watchlist only — the same list Today shows
    as "Your watchlist". Merging every list made a 40-ticker digest for an account
    with 8 lists (2026-09-29). Falls back to the first list if none is marked."""
    wls = list_watchlists(email) or []
    if not wls:
        return []
    default = next((w for w in wls if w.get("is_default")), wls[0])
    tickers = get_watchlist_tickers(default.get("id"), email) or []
    return sorted({str(t).strip().upper() for t in tickers if t})


def email_opted_in(email: str, kind: str) -> bool:
    """P1-41: the account hasn't switched this email type off (defaults to on)."""
    try:
        from db.email_prefs import wants_email

        return wants_email(email, kind)
    except Exception:
        return True


def unsubscribe_link(email: str, kind: str) -> Optional[str]:
    try:
        from db.email_prefs import unsubscribe_url

        return unsubscribe_url(email, kind)
    except Exception:
        return None


def _email_tier_key(email: str, record: Optional[Dict[str, Any]], users: Dict[str, Any], get_user_tier) -> str:
    """Plan used for email delivery. Admin accounts (stored tier 'admin' or the DB
    is_admin flag) count as 'admin', whatever plan is stored, so they get the digest."""
    if str((record or {}).get("tier") or "").strip().lower() == "admin":
        return "admin"
    key = str(getattr(get_user_tier(email, users), "key", "basic") or "basic").strip().lower()
    if key in ("pro", "premium"):
        return key
    try:
        from db.users import is_admin_from_db

        if is_admin_from_db(email):
            return "admin"
    except Exception:
        pass
    return key


def run_morning_digest(force: bool = False) -> None:
    """Assemble and email the pre-open digest to eligible Pro+ users."""
    try:
        from config import MORNING_DIGEST_ENABLED, MORNING_DIGEST_MAX_USERS
    except Exception:
        return
    if not MORNING_DIGEST_ENABLED:
        return

    # Once per ET day, and only in the morning (see DIGEST_WINDOW_ET).
    send_key = daily_send_key(_DIGEST_REFRESH_KEY)
    if not force:
        et = et_now()
        if not DIGEST_WINDOW_ET[0] <= et.hour < DIGEST_WINDOW_ET[1]:
            print(f"[morning_digest] outside the morning window ({et:%H:%M} ET); skipping")
            return
        if already_sent(send_key):
            print("[morning_digest] already sent today; skipping")
            return

    try:
        from auth.tiering import get_user_tier, has_min_tier
        from db.users import load_users
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from ui.email_utils import send_digest_email
    except Exception as e:
        print(f"[morning_digest] import failed: {e}")
        return

    df = _latest_snapshot_df()
    gappers = _market_gappers(df)
    premarket = _premarket_movers()
    picks = _prebreakout_picks(df) if df is not None else []
    golden, top_setups = _todays_setups(df) if df is not None else ([], [])
    earnings_today = _earnings_today()

    # Earnings safety rail (shared across all users): flag gappers and the picks
    # when they report within days, so nobody trades the digest blind to it.
    gapper_edays = _earnings_days_map([r.get("ticker") for r in gappers])
    _flag_earnings_rows(gappers, gapper_edays)
    if picks:
        pick_edays = _earnings_days_map([p["symbol"] for p in picks])
        for p in picks:
            d = pick_edays.get(p["symbol"])
            if d is not None:
                p["symbol"] = f"{p['symbol']} ⚠️E{d}d"

    try:
        users = load_users() or {}
    except Exception:
        users = {}

    try:
        from market_data import build_day_trader_metrics
    except Exception:
        build_day_trader_metrics = None  # type: ignore

    sent = 0
    skipped: Dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for username in list(users.keys())[:MORNING_DIGEST_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            _skip("not_email")
            continue
        # Email delivery is a Pro+ feature (admin accounts included).
        try:
            tier_key = _email_tier_key(email, users.get(username), users, get_user_tier)
            if not has_min_tier(tier_key, "pro"):
                _skip("plan_below_pro")
                continue
        except Exception:
            _skip("plan_lookup_failed")
            continue
        # Only email verified addresses to protect deliverability.
        try:
            from db.email_verification import is_email_verified

            if not is_email_verified(email):
                _skip("unverified")
                continue
        except Exception:
            pass
        if not email_opted_in(email, "digest"):
            _skip("unsubscribed")
            continue

        try:
            tickers = default_watchlist_tickers(email, list_watchlists, get_watchlist_tickers)
            if not tickers:
                _skip("empty_watchlist")
                continue

            watch_rows = (
                build_day_trader_metrics(tickers, with_rvol=False)
                if build_day_trader_metrics
                else []
            )
            week_earnings = _earnings_days_map(tickers, flag_days=7)
            _flag_earnings_rows(watch_rows, {k: v for k, v in week_earnings.items() if v <= 5})
            earnings_hits = [t for t in tickers if t in earnings_today]
            notes = _watchlist_notes(watch_rows, week_earnings)

            # Run 85D: PreBreakout candidates are a Premium feature (Pro gets the rest).
            try:
                user_picks = picks if has_min_tier(tier_key, "premium") else []
            except Exception:
                user_picks = []
            html_inner, text_inner = _compose(
                email, watch_rows, gappers, earnings_hits, user_picks, notes=notes,
                golden=golden, top_setups=top_setups, premarket=premarket,
            )
            # Count only sends the mail service accepted.
            if send_digest_email(
                to_address=email,
                subject="Your morning market digest",
                html_inner=html_inner,
                text_inner=text_inner,
                unsubscribe_url=unsubscribe_link(email, "digest"),
            ):
                sent += 1
            else:
                _skip("send_failed")
        except Exception as e:
            print(f"[morning_digest] {log_id(email)}: {redact(e)}")
            _capture(e)
            _skip("error")
            continue

    if sent > 0:
        mark_sent(send_key)
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())) or "none"
    print(f"[morning_digest] sent {sent} digest(s); skipped: {reasons}")
    record_email_job("digest", {"sent": sent, "skipped": skipped})
