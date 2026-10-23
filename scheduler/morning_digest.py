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
            from analytics.prediction_provenance import models_from_row
            pick["models"] = models_from_row(r)
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


def _premarket_gappers(limit: int = 5, now=None) -> Optional[List[Dict[str, Any]]]:
    """Before the open: today's pre-market gappers over S&P 500 + NASDAQ, the
    same screen as Day Trader's pre-market gappers (traded today, >=2% vs
    yesterday's close, clears the pre-market dollar-volume floor). None outside
    pre-market so the caller uses its regular-session path; [] when nothing
    qualifies yet. Yesterday's scan GapPct is never shown as today's gap."""
    if not _premarket_gappers_active(now):
        return None
    try:
        from market_data import build_premarket_gappers
        from ui.day_trader import _movers_universe

        rows = build_premarket_gappers(_movers_universe(), now=now, limit=limit)
    except Exception:
        return []
    return [{"ticker": r["ticker"], "last": r.get("last"), "chg_pct": r.get("pct"),
             "gap_pct": None, "session": "premarket"} for r in rows[:limit]]


def _premarket_gappers_active(now=None) -> bool:
    try:
        from market_data import premarket_session_date

        return premarket_session_date(now) is not None
    except Exception:
        return False


def _market_gappers(df, limit: int = 5) -> List[Dict[str, Any]]:
    """Top gappers over the snapshot universe.

    Prefer live premarket snapshots (freshest overnight move), but the snapshot
    universe is often low-liquidity names Alpaca has no premarket quote for, so
    that path can come back empty. Fall back to the snapshot's own stored GapPct
    (computed at scan time) so the digest's gappers section is never blank.
    """
    pre = _premarket_gappers(limit)
    if pre is not None:
        return pre
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


def _earnings_calendar_rows(df, *, preopen: bool) -> tuple:
    """(today, week calendar, today's reporters scored) shared by every recipient.
    Every reporter today is scored once (screens off, like the Earnings tab's
    "Score all"); names in the latest market scan are marked ranked."""
    from datetime import timedelta

    from scheduler import earnings_email as ee

    today = et_now().date()
    week = ee.reporters(today, today + timedelta(days=7))
    ranked: List[str] = []
    try:
        col = _symbol_column(df) if df is not None else None
        if col:
            ranked = [str(t) for t in df[col].tolist()]
    except Exception:
        ranked = []
    rows = ee.build_rows([r for r in week if r["date"] == today], premarket=preopen, ranked=ranked)
    return today, week, rows


def _earnings_section(rows, week, today, tickers, *, preopen: bool) -> tuple:
    """(html, text) "Earnings today" section for one recipient; ('', '') on any error."""
    try:
        from scheduler import earnings_email as ee

        return ee.compose_section(
            rows, title="Earnings today",
            order=[(today, "bmo", "☀️ Reported before the open"),
                   (today, "amc", "🌙 Reports after the close"),
                   (today, None, "🕒 Time not announced")],
            price_label="Last", pct_label="Pre-mkt %" if preopen else "Chg %",
            link=ee.scanner_link("today"), link_label="Open the Earnings scanner",
            watched=tickers, watchlist_line=ee.watchlist_line(tickers, week, today),
            footnote="Scores from this morning's scan of every reporter. Results can move a stock "
                     "sharply both ways. Educational only, not financial advice.",
            sub_suffix="scored this morning", zero_is_missing=preopen,
        )
    except Exception:
        return "", ""


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
    from ui import email_kit as k

    lines = []
    for r in rows:
        price = f" · ${r['last']:,.2f}" if r.get("last") is not None else ""
        score = f" · HSF Score {r['score']}" if r.get("score") is not None else ""
        lines.append(f"  {r['ticker']} {r['pct']:+.2f}%{price}{score}")
    table = k.table(["Ticker", "Last", "Pre-mkt %", "HSF Score"],
                    [[k.ticker(r["ticker"]), k.num(r.get("last")), k.pct(r.get("pct")), k.score_badge(r.get("score"))]
                     for r in rows])
    html = k.card("🌅 Pre-market movers", table,
                  sub="From the 8:35 AM ET pre-market scan, vs the previous close. Pre-market prices keep moving.")
    return html, "\n".join(["Pre-market movers (8:35 AM ET scan, vs previous close):", *lines, ""])


def _movers_table(rows: List[Dict[str, Any]], *, show_gap: bool = True, limit: Optional[int] = None) -> str:
    from ui import email_kit as k

    if not rows:
        return k.empty("No data.")
    shown, more = _cap(rows, limit)
    head = ["Ticker", "Last", "Chg %", "Gap %"] if show_gap else ["Ticker", "Last", "Chg %"]
    body = []
    for r in shown:
        cells = [k.flagged_ticker(r.get("ticker"), et_now().date()), k.num(r.get("last")), k.pct(r.get("chg_pct"))]
        if show_gap:
            cells.append(k.pct(r.get("gap_pct")))
        body.append(cells)
    return k.table(head, body) + _more_line(more)


def _cap(rows: List[Dict[str, Any]], limit: Optional[int]) -> tuple:
    """(rows to show, how many are hidden) when capped: names reporting earnings soon
    first, then names that traded today, then the biggest moves."""
    if not limit or len(rows) <= limit:
        return rows, 0

    def key(r):
        flagged = " ⚠️E" in str(r.get("ticker") or "")
        traded = r.get("chg_pct") is not None
        v = r.get("chg_pct") if traded else r.get("prior_session_chg_pct")
        try:
            size = abs(float(v))
        except (TypeError, ValueError):
            size = -1.0
        return (flagged, traded, size)
    return sorted(rows, key=key, reverse=True)[:limit], len(rows) - limit


def _more_line(more: int) -> str:
    if not more:
        return ""
    from ui import email_kit as k

    return k.note(f"+ {more} more on your watchlist (earnings and biggest moves shown first)")


def _is_preopen(rows: List[Dict[str, Any]]) -> bool:
    return any(r.get("session") == "premarket" for r in rows or [])


def _preopen_last(r: Dict[str, Any]):
    """Today's pre-market trade when there is one, else yesterday's close (not
    a stale after-hours print)."""
    if r.get("traded_today") is False and r.get("previous_close") is not None:
        return r.get("previous_close")
    return r.get("last")


def _pct(v) -> str:
    return ("%+.2f%%" % v) if v is not None else "—"


def _preopen_table(rows: List[Dict[str, Any]], *, show_prior: bool, limit: Optional[int] = None) -> str:
    """Before the open there is no gap or session change yet: show the move vs
    yesterday's close from today's latest pre-market trade, plus yesterday's
    own move for watchlist rows."""
    from ui import email_kit as k

    if not rows:
        return k.empty("No data.")
    shown, more = _cap(rows, limit)
    head = ["Ticker", "Last", "Pre-mkt %", "Yesterday %"] if show_prior else ["Ticker", "Last", "Pre-mkt %"]
    body = []
    for r in shown:
        cells = [k.flagged_ticker(r.get("ticker"), et_now().date()), k.num(_preopen_last(r)), k.pct(r.get("chg_pct"))]
        if show_prior:
            cells.append(k.pct(r.get("prior_session_chg_pct")))
        body.append(cells)
    return k.table(head, body) + _more_line(more)


def _preopen_text(rows: List[Dict[str, Any]], *, show_prior: bool) -> str:
    if not rows:
        return "  (none)"
    out = []
    for r in rows:
        line = f"  {r.get('ticker')}: {_preopen_last(r)} (pre-mkt {_pct(r.get('chg_pct'))}"
        if show_prior:
            line += f", yesterday {_pct(r.get('prior_session_chg_pct'))}"
        out.append(line + ")")
    return "\n".join(out)


_PREOPEN_NOTE = "<p style='margin:4px 0 0;font-size:12px;color:#888'>{}</p>"


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
            "<p style='color:#166534;background:#f0fdf4;border:1px solid #bbf7d0;padding:10px 14px;"
            "border-radius:10px;margin:12px 0 0;font-size:14px'>"
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
    if len(movers) >= 2:   # one mover is both "best" and "worst": say nothing
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
    preopen_gappers: bool = False,
    earnings_section: Optional[tuple] = None,
) -> tuple[str, str]:
    """Return (html_inner, text_inner) for one user's digest: email_kit cards
    (the shell adds the header and the At a glance band)."""
    from ui import email_kit as k

    date_s = et_now().strftime("%A, %b %d")
    html: List[str] = []
    text = [f"Morning snapshot · {date_s}", ""]

    tr_html, tr_text = _track_record_line()
    if tr_html:
        html.append(tr_html)
        text += [tr_text, ""]

    note_html = k.note("📌 " + " · ".join(notes)) if notes else ""
    if _is_preopen(watch_rows):
        html.append(k.card("📋 Your watchlist",
                           _preopen_table(watch_rows, show_prior=True, limit=WATCHLIST_ROWS) + note_html,
                           sub="Pre-market vs yesterday's close. — means no pre-market trade yet, "
                               "so Last is yesterday's close."))
        text += ["Your watchlist (before the open):", _preopen_text(watch_rows, show_prior=True), ""]
    else:
        html.append(k.card("📋 Your watchlist", _movers_table(watch_rows, show_gap=True, limit=WATCHLIST_ROWS)
                           + note_html))
        text += ["Your watchlist:", _movers_text(watch_rows), ""]
    if notes:
        text += [f"Notes: {' · '.join(notes)}", ""]

    if preopen_gappers:
        body = (_preopen_table(gappers, show_prior=False) if gappers else
                k.empty("No S&amp;P 500 or NASDAQ stock is gapping 2% or more on enough pre-market volume yet."))
        html.append(k.card("🚀 Top market gappers", body,
                           sub="S&amp;P 500 and NASDAQ stocks moving 2%+ before the open, with enough "
                               "pre-market volume to trade."))
        text += ["Top market gappers (pre-market vs yesterday's close):",
                 _preopen_text(gappers, show_prior=False), ""]
    else:
        html.append(k.card("🚀 Top market gappers", _movers_table(gappers, show_gap=True)))
        text += ["Top market gappers:", _movers_text(gappers), ""]

    pm_html, pm_text = _premarket_section(premarket or [])
    if pm_html:
        html.append(pm_html)
        text.append(pm_text)

    if earnings_section and earnings_section[0]:
        html.append(earnings_section[0])
        text.append(earnings_section[1])
    elif earnings_hits:
        names = ", ".join(sorted(earnings_hits))
        html.append(k.card("📅 Earnings today (your watchlist)", f"<p style='margin:0'>{k.esc(names)}</p>"))
        text += ["Earnings today (your watchlist):", f"  {names}", ""]

    # 🎯 Today's setups — actionable candidates for the open, with the PreBreakout picks.
    setup_parts: List[str] = []
    if golden or top_setups:
        text += ["Today's setups:"]
        if golden:
            text += [f"  Fresh golden crosses: {', '.join(golden)}"]
        if top_setups:
            text += [f"  Top breakout scores: {', '.join(f'{t} ({s:g})' for t, s in top_setups)}"]
        text += [""]
    if top_setups:
        setup_parts.append(k.group("🚀 Top breakout scores")
                           + k.chips((t, k.chip_score(s)) for t, s in top_setups))
    if golden:
        setup_parts.append(k.group("📈 Fresh EMA 9/21 golden crosses") + k.chips((g, None) for g in golden))
    if picks:
        label = "🧠 PreBreakout picks" if len(picks) > 1 else "🧠 PreBreakout pick"
        rows = [[k.flagged_ticker(p["symbol"], et_now().date()), k.num(p.get("last"), money=True),
                 f"{p['prob']:.1f}% likelihood"] for p in picks]
        setup_parts.append(k.group(label) + k.table(["", "", ""], rows)
                           + k.note("Calibrated model likelihood of setup follow-through, not a price forecast."))
        text += ["PreBreakout picks:"]
        text += [f"  {p['symbol']}"
                 + (f" ${p['last']:.2f}" if p.get('last') is not None else "")
                 + f" — {p['prob']:.1f}% likelihood" for p in picks]
        text += [""]
    if setup_parts:
        html.append(k.card("🎯 Today's setups", "".join(setup_parts)
                           + k.note("Educational only, not financial advice. Confirm setups yourself at the open.")
                           + k.button(_app_url("/today"), "Open Today in HSF →"),
                           sub="Highest HSF breakout scores from the latest scan"))

    return "".join(html), "\n".join(text)


WATCHLIST_ROWS = 6


def _app_url(path: str) -> str:
    try:
        from services.lifecycle_emails import _base_url

        return _base_url() + path
    except Exception:
        return "https://hsf-web-v1.onrender.com" + path


def _fmt_move(v) -> str:
    return f"{v:+.1f}%"


def digest_glance(watch_rows: List[Dict[str, Any]], gappers: List[Dict[str, Any]], *,
                  preopen: bool = False, earnings_count: int = 0,
                  week_earnings: Optional[Dict[str, int]] = None,
                  top_setups: Optional[List[tuple]] = None) -> str:
    """The 'At a glance' line under the header: watchlist mood, the biggest gapper,
    how many report today (and which watchlist names report soon), the top setup."""
    from ui import email_kit as k

    parts: List[str] = []
    movers = [r for r in watch_rows if r.get("chg_pct") is not None]
    if watch_rows:
        if not movers:
            parts.append("No pre-market trades on your watchlist yet." if preopen
                         else "Your watchlist hasn't traded yet.")
        else:
            big = max(movers, key=lambda r: abs(r["chg_pct"]))
            name = k.split_flag(big["ticker"])[0]
            if len(movers) <= 2 and abs(big["chg_pct"]) < 1:
                when = " before the open" if preopen else ""
                parts.append(f"Your watchlist is quiet{when} ({k.esc(name)} {_fmt_move(big['chg_pct'])}).")
            else:
                up = sum(1 for r in movers if r["chg_pct"] >= 0)
                parts.append(f"Your watchlist: {up} up, {len(movers) - up} down; biggest move "
                             f"<b style='color:#fff'>{k.esc(name)}</b> {_fmt_move(big['chg_pct'])}.")
    moved = [g for g in gappers if g.get("chg_pct") is not None]
    if moved:
        g = max(moved, key=lambda r: abs(r["chg_pct"]))
        name = k.esc(k.split_flag(g["ticker"])[0])
        lead = "is the only big gapper" if len(moved) == 1 else "leads the gappers"
        parts.append(f"<b style='color:#fff'>{name}</b> {lead} ({_fmt_move(g['chg_pct'])}).")
    earn = []
    if earnings_count:
        earn.append(f"<b style='color:#fff'>{earnings_count} compan{'y' if earnings_count == 1 else 'ies'}</b> "
                    "report today")
    soon = [s for s, _ in sorted((week_earnings or {}).items(), key=lambda kv: kv[1])][:3]
    if soon:
        names = " and ".join(", ".join(f"<b style='color:#fff'>{k.esc(s)}</b>" for s in soon).rsplit(", ", 1))
        verb = "reports" if len(soon) == 1 else "report"
        earn.append(f"{names} on your watchlist {verb} this week")
    if earn:
        parts.append("; ".join(earn) + ".")
    if top_setups:
        t, sc = top_setups[0]
        parts.append(f"Top setup: <b style='color:#fff'>{k.esc(t)}</b> (score {sc:g}).")
    return k.glance_text(parts)


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
    preopen_gappers = _premarket_gappers_active()
    premarket = _premarket_movers()
    picks = _prebreakout_picks(df) if df is not None else []
    golden, top_setups = _todays_setups(df) if df is not None else ([], [])
    earnings_today = _earnings_today()
    try:
        earn_today, earn_week, earn_rows = _earnings_calendar_rows(df, preopen=preopen_gappers)
    except Exception as e:
        print(f"[morning_digest] earnings section unavailable: {type(e).__name__}")
        earn_today, earn_week, earn_rows = et_now().date(), [], []

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
            section = _earnings_section(earn_rows, earn_week, earn_today, tickers, preopen=preopen_gappers)
            # the section's watchlist line names this week's reporters; keep notes to the recap then
            notes = _watchlist_notes(watch_rows, {} if section[0] and earn_week else week_earnings)

            # Run 85D: PreBreakout candidates are a Premium feature (Pro gets the rest).
            try:
                user_picks = picks if has_min_tier(tier_key, "premium") else []
            except Exception:
                user_picks = []
            html_inner, text_inner = _compose(
                email, watch_rows, gappers, earnings_hits, user_picks, notes=notes,
                golden=golden, top_setups=top_setups, premarket=premarket,
                preopen_gappers=preopen_gappers, earnings_section=section,
            )
            # Count only sends the mail service accepted.
            glance = digest_glance(
                watch_rows, gappers, preopen=preopen_gappers or _is_preopen(watch_rows),
                earnings_count=len(earn_rows), week_earnings=week_earnings, top_setups=top_setups)
            if send_digest_email(
                to_address=email,
                subject="Your morning market digest",
                html_inner=html_inner,
                text_inner=text_inner,
                unsubscribe_url=unsubscribe_link(email, "digest"),
                title_line=f"Morning snapshot · {et_now():%A, %b %-d}"
                           + (" · before the open" if preopen_gappers else ""),
                glance_html=glance,
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
