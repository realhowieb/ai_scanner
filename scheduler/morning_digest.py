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

_DIGEST_REFRESH_KEY = "morning_digest"


def _latest_snapshot_df():
    """Load the most recent saved snapshot as a DataFrame, or None."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_runs(limit=10) or []
        snap = next((r for r in runs if r.get("is_snapshot")), None) or (runs[0] if runs else None)
        if not snap:
            return None
        raw = load_run_results(snap["id"])
        return normalize_results_to_df(raw) if raw else None
    except Exception:
        return None


def _symbol_column(df) -> Optional[str]:
    for col in ("Symbol", "Ticker", "symbol", "ticker"):
        if col in df.columns:
            return col
    return None


def _prebreakout_pick(df) -> Optional[Dict[str, Any]]:
    """Return the top PreBreakout candidate {symbol, prob} from the snapshot."""
    try:
        from ml_prebreakout import score_prebreakout

        scored = score_prebreakout(df)
        if scored is None or len(scored) == 0 or "PreBreakoutProb%" not in scored.columns:
            return None
        sym_col = _symbol_column(scored)
        if not sym_col:
            return None
        top = scored.sort_values("PreBreakoutProb%", ascending=False).iloc[0]
        prob = float(top.get("PreBreakoutProb%") or 0.0)
        if prob <= 0:
            return None
        return {"symbol": str(top.get(sym_col)).upper(), "prob": round(prob, 1)}
    except Exception:
        return None


def _market_gappers(df, limit: int = 5) -> List[Dict[str, Any]]:
    """Top gappers computed from live snapshots over the snapshot universe."""
    try:
        from market_data import build_day_trader_metrics

        sym_col = _symbol_column(df) if df is not None else None
        if df is None or not sym_col:
            return []
        symbols = [str(s).upper() for s in df[sym_col].tolist() if str(s).strip()][:60]
        rows = build_day_trader_metrics(symbols, with_rvol=False)
        rows = [r for r in rows if r.get("gap_pct") is not None]
        rows.sort(key=lambda r: abs(r["gap_pct"]), reverse=True)
        return rows[:limit]
    except Exception:
        return []


def _earnings_today() -> set:
    """Set of symbols reporting earnings today (UTC)."""
    try:
        from db.earnings import fetch_earnings_this_week

        today = datetime.now(timezone.utc).date()
        rows = fetch_earnings_this_week(days_ahead=1) or []
        return {
            str(r.get("symbol")).upper()
            for r in rows
            if r.get("symbol") and r.get("earnings_date") == today
        }
    except Exception:
        return set()


def _movers_table(rows: List[Dict[str, Any]], *, show_gap: bool = True) -> str:
    if not rows:
        return "<p style='color:#888'>No data.</p>"
    head = "<tr><th align='left'>Ticker</th><th align='right'>Last</th><th align='right'>Chg %</th>"
    if show_gap:
        head += "<th align='right'>Gap %</th>"
    head += "</tr>"
    body = ""
    for r in rows:
        chg = r.get("chg_pct")
        gap = r.get("gap_pct")
        color = "#16a34a" if (chg is not None and chg >= 0) else "#dc2626"
        body += (
            f"<tr><td>{r.get('ticker')}</td>"
            f"<td align='right'>{r.get('last')}</td>"
            f"<td align='right' style='color:{color}'>"
            f"{('%+.2f%%' % chg) if chg is not None else '—'}</td>"
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


def _compose(
    username: str,
    watch_rows: List[Dict[str, Any]],
    gappers: List[Dict[str, Any]],
    earnings_hits: List[str],
    pick: Optional[Dict[str, Any]],
) -> tuple[str, str]:
    """Return (html_inner, text_inner) for one user's digest."""
    date_s = datetime.now(timezone.utc).strftime("%A, %b %d")
    html = [f"<p style='color:#666;margin:0 0 12px'>Morning snapshot · {date_s}</p>"]
    text = [f"Morning snapshot · {date_s}", ""]

    tr_html, tr_text = _track_record_line()
    if tr_html:
        html.append(tr_html)
        text += [tr_text, ""]

    html.append("<h3 style='margin:16px 0 6px'>📋 Your watchlist</h3>")
    html.append(_movers_table(watch_rows, show_gap=True))
    text += ["Your watchlist:", _movers_text(watch_rows), ""]

    html.append("<h3 style='margin:16px 0 6px'>🚀 Top market gappers</h3>")
    html.append(_movers_table(gappers, show_gap=True))
    text += ["Top market gappers:", _movers_text(gappers), ""]

    if earnings_hits:
        names = ", ".join(sorted(earnings_hits))
        html.append(
            "<h3 style='margin:16px 0 6px'>📅 Earnings today (your watchlist)</h3>"
            f"<p>{names}</p>"
        )
        text += ["Earnings today (your watchlist):", f"  {names}", ""]

    if pick:
        html.append(
            "<h3 style='margin:16px 0 6px'>🧠 PreBreakout pick</h3>"
            f"<p><strong>{pick['symbol']}</strong> — {pick['prob']}% model confidence</p>"
        )
        text += ["PreBreakout pick:", f"  {pick['symbol']} — {pick['prob']}%", ""]

    return "".join(html), "\n".join(text)


def run_morning_digest(force: bool = False) -> None:
    """Assemble and email the pre-open digest to eligible Pro+ users."""
    try:
        from config import MORNING_DIGEST_ENABLED, MORNING_DIGEST_MAX_USERS
    except Exception:
        return
    if not MORNING_DIGEST_ENABLED:
        return

    # Once-per-day throttle (reuses the earnings refresh-log table with our key).
    if not force:
        try:
            from db.earnings import should_refresh_earnings_today

            if not should_refresh_earnings_today(_DIGEST_REFRESH_KEY):
                print("[morning_digest] already sent today; skipping")
                return
        except Exception:
            pass

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
    pick = _prebreakout_pick(df) if df is not None else None
    earnings_today = _earnings_today()

    try:
        users = load_users() or {}
    except Exception:
        users = {}

    try:
        from market_data import build_day_trader_metrics
    except Exception:
        build_day_trader_metrics = None  # type: ignore

    sent = 0
    for username in list(users.keys())[:MORNING_DIGEST_MAX_USERS]:
        email = (username or "").strip().lower()
        if not email or "@" not in email:
            continue
        # Email delivery is a Pro+ feature.
        try:
            if not has_min_tier(get_user_tier(email, users), "pro"):
                continue
        except Exception:
            continue
        # Only email verified addresses to protect deliverability.
        try:
            from db.email_verification import is_email_verified

            if not is_email_verified(email):
                continue
        except Exception:
            pass

        try:
            wls = list_watchlists(email) or []
            tickers: List[str] = []
            for wl in wls:
                tickers.extend(get_watchlist_tickers(wl.get("id"), email) or [])
            tickers = sorted({str(t).strip().upper() for t in tickers if t})
            if not tickers:
                continue

            watch_rows = (
                build_day_trader_metrics(tickers, with_rvol=False)
                if build_day_trader_metrics
                else []
            )
            earnings_hits = [t for t in tickers if t in earnings_today]

            html_inner, text_inner = _compose(email, watch_rows, gappers, earnings_hits, pick)
            send_digest_email(
                to_address=email,
                subject="Your morning market digest",
                html_inner=html_inner,
                text_inner=text_inner,
            )
            sent += 1
        except Exception as e:
            print(f"[morning_digest] {email}: {e}")
            continue

    if sent > 0:
        try:
            from db.earnings import mark_earnings_refreshed_today

            mark_earnings_refreshed_today(_DIGEST_REFRESH_KEY)
        except Exception:
            pass
    print(f"[morning_digest] sent {sent} digest(s)")
