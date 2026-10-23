"""Market-wide earnings sections for the morning digest and the evening wrap.

Both emails go to Pro and up only, so the whole calendar is fine to show here
(the 7-day calendar is the Pro feature; see #132). The reporter list is built
once per run and shared by every recipient: each name in the window is scored
with the same "score all" scan the Earnings tab uses, so rows carry an HSF Score
even when the name never ranked in the market scan. Best-effort throughout:
scoring or quotes failing leaves dashes, and nothing here ever raises.
"""
from __future__ import annotations

import datetime as dt
import html as _html
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

SCORE_MAX_SYMBOLS = 400   # a heavy earnings day is ~150 names; keep the cron bounded
GROUP_LIMIT = {"bmo": 6, "amc": 6, None: 3}
MIN_PRICE = 5.0           # sub-$5 names stay in the "+N more" count, not the email's tables
GROUP_LABEL = {"bmo": "☀️ Before the open", "amc": "🌙 After the close", None: "🕒 Time not announced"}


def _key(ticker: Any) -> str:
    return "".join(ch for ch in str(ticker or "").upper() if ch.isalnum())


def _time(raw: Any) -> Optional[str]:
    t = str(raw or "").strip().lower()
    return t if t in ("bmo", "amc") else None


def reporters(start: dt.date, end: dt.date) -> List[Dict[str, Any]]:
    """[{ticker, date, time}] reporting between start and end (inclusive), from the calendar."""
    try:
        from db.earnings import fetch_earnings_this_week

        rows = fetch_earnings_this_week(days_ahead=(end - start).days, start=start) or []
    except Exception:
        return []
    out, seen = [], set()
    for r in rows:
        d = r.get("earnings_date")
        if isinstance(d, dt.datetime):
            d = d.date()
        k = _key(r.get("symbol"))
        if not k or k in seen or not isinstance(d, dt.date) or not (start <= d <= end):
            continue
        seen.add(k)
        out.append({"ticker": str(r.get("symbol")).strip().upper(), "date": d, "time": _time(r.get("earnings_time"))})
    return out


def score_tickers(tickers: List[str], *, premarket: bool = False) -> Dict[str, Dict[str, Any]]:
    """{key: {score, last, chg_pct}} from one scan over `tickers` with the screens off
    (the Earnings tab's "Score all"). Empty on any failure."""
    tickers = list(dict.fromkeys(t for t in tickers if t))[:SCORE_MAX_SYMBOLS]
    if not tickers:
        return {}
    try:
        from api.scans import _scan_row
        from scan.engine import run_breakout_scan
        from scan.execution import run_manual_scan_execution
        from ui.market_scans import top_setups
        from ui.scan_providers import apply_alpaca_extended_prices, sanitize_universe_symbols

        tickers = sanitize_universe_symbols(tickers)
        df = run_manual_scan_execution(
            runner=run_breakout_scan, tickers=tickers, premarket=premarket, afterhours=False,
            unusual_volume=False, min_gap=0.0, min_price=0.0, max_price=1_000_000.0,
            top_n=len(tickers), profile="regular", apply_gap_filter=False, diagnostics=False,
            extended_price_transform=apply_alpaca_extended_prices, min_dollar_vol=0.0,
        )
        out: Dict[str, Dict[str, Any]] = {}
        for o in top_setups(df, n=100_000):
            row = _scan_row(o)
            k = _key(row.get("ticker"))
            if k and k not in out:
                out[k] = {"score": row.get("score"), "last": row.get("last"), "chg_pct": row.get("chg_pct")}
        return out
    except Exception as e:
        print(f"[earnings_email] scoring failed: {type(e).__name__}")
        return {}


def _fill_quotes(rows: List[Dict[str, Any]]) -> None:
    """Last / change from live quotes for rows the scan didn't price."""
    missing = [r for r in rows if r.get("last") is None]
    if not missing:
        return
    try:
        from api.scans import _fill_quotes as fill

        fill(missing)
    except Exception:
        pass


def build_rows(
    calendar: List[Dict[str, Any]],
    *,
    premarket: bool = False,
    ranked: Iterable[str] = (),
    scorer: Optional[Callable[..., Dict[str, Dict[str, Any]]]] = None,
    quotes: bool = True,
) -> List[Dict[str, Any]]:
    """Calendar rows with score / last / chg_pct and `ranked` (in the market scan),
    highest score first within each day."""
    scores = (scorer or score_tickers)([r["ticker"] for r in calendar], premarket=premarket)
    ranked_keys = {_key(t) for t in ranked}
    rows, seen = [], set()
    for r in calendar:
        if _key(r["ticker"]) in seen:
            continue
        seen.add(_key(r["ticker"]))
        s = scores.get(_key(r["ticker"])) or {}
        rows.append({**r, "score": s.get("score"), "last": s.get("last"), "chg_pct": s.get("chg_pct"),
                     "ranked": _key(r["ticker"]) in ranked_keys})
    rows.sort(key=lambda r: (r["date"], -(r["score"] if r["score"] is not None else -1), r["ticker"]))
    if quotes:
        _fill_quotes(rows)
    return rows


def scanner_link(when: str) -> str:
    try:
        from services.lifecycle_emails import _base_url

        base = _base_url()
    except Exception:
        base = "https://hsf-web-v1.onrender.com"
    return f"{base}/scanner?signal=earnings&when={when}&all=1"


def _fmt_price(v: Any) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v: Any) -> Tuple[str, str]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—", "#888"
    return ("%+.2f%%" % f).replace("-", "−"), ("#16a34a" if f >= 0 else "#dc2626")


def _table(rows: List[Dict[str, Any]], *, price_label: str, pct_label: str, watched: set) -> Tuple[str, List[str]]:
    head = ("<tr><th align='left'>Ticker</th><th align='right'>HSF Score</th>"
            f"<th align='right'>{price_label}</th><th align='right'>{pct_label}</th></tr>")
    body, text = "", []
    for r in rows:
        t = _html.escape(r["ticker"])
        star = " ⭐" if _key(r["ticker"]) in watched else ""
        name = f"<b>{t}</b>{star}" if r.get("ranked") else f"{t}{star}"
        score = r.get("score")
        score_html = (f"{score:g}" if isinstance(score, (int, float)) else "<span style='color:#888'>—</span>")
        pct, color = _fmt_pct(r.get("chg_pct"))
        body += (f"<tr><td>{name}</td><td align='right'>{score_html}</td>"
                 f"<td align='right'>{_fmt_price(r.get('last'))}</td>"
                 f"<td align='right' style='color:{color}'>{pct}</td></tr>")
        text.append(f"  {r['ticker']}{' *' if star else ''}"
                    f"  score {score if score is not None else '—'}  {_fmt_price(r.get('last'))}  {pct}")
    return ("<table style='border-collapse:collapse;width:100%;font-size:13px' cellpadding='4'>"
            + head + body + "</table>"), text


def _groups(rows: List[Dict[str, Any]], order: List[Tuple[dt.date, Optional[str], str]]):
    """Yield (label, shown_rows, total) for each (date, time, label) in order."""
    for d, t, label in order:
        group = [r for r in rows if r["date"] == d and r["time"] == t]
        shown = [r for r in group if not _penny(r)][:GROUP_LIMIT.get(t, 3)]
        if shown:
            yield label, shown, len(group)
        elif group:
            yield None, [], len(group)   # only counted in "+N more"


def _penny(r: Dict[str, Any]) -> bool:
    try:
        return r.get("last") is not None and float(r["last"]) < MIN_PRICE
    except (TypeError, ValueError):
        return False


def compose_section(
    rows: List[Dict[str, Any]],
    *,
    title: str,
    order: List[Tuple[dt.date, Optional[str], str]],
    price_label: str,
    pct_label: str,
    link: str,
    link_label: str,
    watched: Iterable[str] = (),
    watchlist_line: str = "",
    footnote: str = "",
) -> Tuple[str, str]:
    """(html, text) for one earnings section, or ('', '') when nobody reports."""
    watched_keys = {_key(t) for t in watched}
    groups = list(_groups(rows, order))
    if not any(g[1] for g in groups) and not watchlist_line:
        return "", ""
    html = [f"<h3 style='margin:16px 0 6px'>📅 {title}</h3>"]
    text = [f"{title}:"]
    shown = total = 0
    for label, group, n in groups:
        total += n
        if not group:
            continue
        table, lines = _table(group, price_label=price_label, pct_label=pct_label, watched=watched_keys)
        html.append(f"<p style='margin:10px 0 2px;font-weight:600'>{label}</p>{table}")
        text += [f" {label}:"] + lines
        shown += len(group)
    if shown:
        more = total - shown
        lead = f"+ {more} more · " if more > 0 else ""
        html.append(f"<p style='color:#555;margin:6px 0 0'>{lead}"
                    f"<a href='{_html.escape(link)}' style='color:#2563eb'>{link_label} →</a></p>")
        text.append(f" {lead}{link_label}: {link}")
    if watchlist_line:
        html.append(f"<p style='margin:12px 0 2px;font-weight:600'>📋 Your watchlist</p>"
                    f"<p style='margin:2px 0'>{watchlist_line}</p>")
        text.append(f" Your watchlist: {_strip_tags(watchlist_line)}")
    if footnote:
        html.append(f"<p style='color:#888;font-size:12px;margin:8px 0 0'>{footnote}</p>")
    text.append("")
    return "".join(html), "\n".join(text)


def _strip_tags(s: str) -> str:
    import re

    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def _when_phrase(d: dt.date, t: Optional[str], today: dt.date) -> str:
    day = "today" if d == today else ("tomorrow" if d == today + dt.timedelta(days=1) else d.strftime("%a %b %-d"))
    tail = {"bmo": " before the open", "amc": " after the close"}.get(t or "", "")
    return day + tail


def watchlist_line(tickers: Iterable[str], week: List[Dict[str, Any]], today: dt.date) -> str:
    """'⚠️ BE today after the close · AXP Fri before the open' for watched names reporting in `week`."""
    keys = {_key(t) for t in tickers}
    hits, seen = [], set()
    for r in sorted(week, key=lambda r: (r["date"], r["time"] or "zz")):
        if _key(r["ticker"]) in keys and _key(r["ticker"]) not in seen:
            seen.add(_key(r["ticker"]))
            hits.append(r)
    if not hits:
        return ""
    parts = [f"<b>{_html.escape(r['ticker'])}</b> {_when_phrase(r['date'], r['time'], today)}" for r in hits[:6]]
    return "⚠️ " + " · ".join(parts)


def next_trading_day(d: dt.date) -> dt.date:
    """The next weekday after d (market holidays aren't modeled; the calendar is empty then)."""
    n = d + dt.timedelta(days=1)
    while n.weekday() >= 5:
        n += dt.timedelta(days=1)
    return n
