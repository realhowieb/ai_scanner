"""Alert email layout: one email per user per scan (P1-49), readable sections.

Each alert that fired becomes a section with a heading ("Breakout alert ·
BreakoutScore ≥ 30") and one row per ticker. The HTML goes into the same
branded shell as the morning digest; the plain text mirrors it.

P2-50: a breakout section whose names all appear in another breakout section
(e.g. ≥ 50 inside ≥ 30) is dropped from the email. Each alert still records
its own event and throttle; this is display only.
"""
from __future__ import annotations

import html as _html
import re
from typing import Any, Dict, List, Optional

MAX_ROWS = 10

LABELS = {
    "breakout": "Breakout alert",
    "watchlist": "Watchlist alert",
    "price": "Price alert",
    "ema_cross": "EMA cross alert",
    "ewo_cross": "EWO cross alert",
}

_THRESHOLD_SUFFIX = re.compile(r"\s*\(≥ [0-9.]+\)")
# Watchlist rows: the heading already says "in the latest scan".
_IN_SCAN = re.compile(r"^in scan results(?: \((.*)\))?(.*)$")


def label_for(alert: Dict[str, Any]) -> str:
    return LABELS.get(alert.get("alert_type"), "Alert")


def heading_for(alert: Dict[str, Any]) -> str:
    """Section heading that states the alert's rule once."""
    atype = alert.get("alert_type")
    label = label_for(alert)
    if atype == "breakout":
        thr = alert.get("threshold")
        rule = f"BreakoutScore ≥ {float(thr or 0):g}"
        if alert.get("watchlist_only"):
            rule += " · your watchlist"
        return f"{label} · {rule}"
    if atype == "watchlist":
        return f"{label} · your watchlist names in the latest scan"
    tk = str(alert.get("ticker") or "").upper()
    return f"{label} · {tk}" if tk else label


def make_section(alert: Dict[str, Any], lines: List[str]) -> Dict[str, Any]:
    return {"type": alert.get("alert_type"), "label": label_for(alert),
            "heading": heading_for(alert), "lines": list(lines)}


def split_line(line: str) -> tuple:
    """'IOVA: BreakoutScore 118.9 (≥ 30)' -> ('IOVA', 'BreakoutScore 118.9')."""
    ticker, sep, rest = str(line).partition(":")
    if not sep:
        return "", str(line).strip()
    detail = _THRESHOLD_SUFFIX.sub("", rest).strip()
    m = _IN_SCAN.match(detail)
    if m:
        detail = ((m.group(1) or "in the latest scan") + (m.group(2) or "")).strip()
    return ticker.strip().upper(), detail


def _tickers(section: Dict[str, Any]) -> set:
    return {split_line(ln)[0] for ln in section["lines"] if split_line(ln)[0]}


def drop_covered_breakouts(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove breakout sections whose tickers are all listed in another breakout
    section (identical sets keep the first). Also removes exact repeats."""
    out: List[Dict[str, Any]] = []
    for i, sec in enumerate(sections):
        if any(sec["heading"] == o["heading"] and sec["lines"] == o["lines"] for o in out):
            continue
        if sec["type"] == "breakout":
            mine = _tickers(sec)
            covered = False
            for j, other in enumerate(sections):
                if j == i or other["type"] != "breakout":
                    continue
                theirs = _tickers(other)
                if mine and mine <= theirs and (mine != theirs or j < i):
                    covered = True
                    break
            if covered:
                continue
        out.append(sec)
    return out


def _shown(lines: List[str]) -> tuple:
    shown = lines[:MAX_ROWS]
    return shown, len(lines) - len(shown)


def _key(t: Any) -> str:
    return "".join(ch for ch in str(t or "").upper() if ch.isalnum())


def quotes_from_df(df) -> Dict[str, Dict[str, Any]]:
    """{key: {score, last, chg_pct, setup}} for every row of the scan the alerts
    ran on, so the email can show price, move and HSF Score. Empty on failure."""
    if df is None:
        return {}
    try:
        from api.scans import _scan_row
        from ui.market_scans import top_setups

        out: Dict[str, Dict[str, Any]] = {}
        for o in top_setups(df, n=100_000):
            row = _scan_row(o)
            k = _key(row.get("ticker"))
            if k and k not in out:
                out[k] = {"score": row.get("score"), "last": row.get("last"),
                          "chg_pct": row.get("chg_pct"), "setup": row.get("primary_setup")}
        return out
    except Exception:
        return {}


_EARN = re.compile(r"\s*⚠️ earnings in (\d+)d")


def _app_url(path: str) -> str:
    try:
        from services.lifecycle_emails import _base_url

        return _base_url() + path
    except Exception:
        return "https://hsf-web-v1.onrender.com" + path


def _row_cells(tk: str, detail: str, quote: Dict[str, Any], watched: set) -> List[str]:
    from ui import email_kit as k

    days = None
    m = _EARN.search(detail)
    if m:
        days = int(m.group(1))
        detail = _EARN.sub("", detail).strip()
    extra = (k.earnings_tag(days) if days is not None else "") + (
        k.tag("Watchlist", "watch") if _key(tk) in watched else "")
    sub = " · ".join(x for x in (str(quote.get("setup") or "").strip(), detail) if x)
    sub_html = (f"<div style='font-size:12px;color:{k.MUTED};margin-top:1px'>{k.esc(sub)}</div>"
                if sub else "")
    return [k.ticker(tk, extra=extra) + sub_html, k.num(quote.get("last")),
            k.pct(quote.get("chg_pct")), k.score_badge(quote.get("score"))]


def _section_card(sec: Dict[str, Any], quotes: Dict[str, Dict[str, Any]], watched: set) -> str:
    from ui import email_kit as k

    shown, extra = _shown(sec["lines"])
    rows = [_row_cells(tk, detail, quotes.get(_key(tk)) or {}, watched)
            for tk, detail in map(split_line, shown) if tk]
    body = k.table(["Ticker", "Last", "Chg %", "HSF Score"], rows) if rows else ""
    plain = [detail for tk, detail in map(split_line, shown) if not tk]
    body += "".join(k.empty(k.esc(p)) for p in plain)
    total = len(sec["lines"])
    if extra > 0:
        body += k.note(f"+ {extra} more")
    tickers = [tk for tk, _ in map(split_line, sec["lines"]) if tk]
    if sec["type"] in ("breakout", "watchlist") or len(tickers) != 1:
        label = f"See all {total} in the Scanner →" if extra > 0 else "Open the Scanner →"
        body += k.button(_app_url("/scanner"), label)
    else:
        body += k.button(_app_url(f"/stocks/{tickers[0]}"), f"Open {k.esc(tickers[0])} in HSF →")
    sub = f"{total} stock{'s' if total != 1 else ''}" if sec["type"] in ("breakout", "watchlist") else ""
    return k.card(k.esc(sec["heading"]), body, sub=sub)


def subject_for(sections: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for sec in sections:
        for ln in sec["lines"]:
            tk = split_line(ln)[0]
            if tk and tk not in names:
                names.append(tk)
    lead = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    what = f"{sections[0]['label']}" if len(sections) == 1 else f"{len(sections)} alerts"
    return f"📈 {what}: {lead}" if lead else f"📈 {what} triggered"


def text_body(sections: List[Dict[str, Any]]) -> str:
    parts = []
    for sec in sections:
        shown, extra = _shown(sec["lines"])
        rows = [f"  {tk:<6} {detail}" if tk else f"  {detail}" for tk, detail in map(split_line, shown)]
        if extra > 0:
            rows.append(f"  …and {extra} more")
        parts.append(sec["heading"] + "\n" + "\n".join(rows))
    return "\n\n".join(parts)


def html_body(sections: List[Dict[str, Any]], quotes: Optional[Dict[str, Dict[str, Any]]] = None,
              watched: Any = ()) -> str:
    quotes = quotes or {}
    watched_keys = {_key(t) for t in watched or ()}
    return "".join(_section_card(sec, quotes, watched_keys) for sec in sections)


def _all_tickers(sections: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for sec in sections:
        for ln in sec["lines"]:
            tk = split_line(ln)[0]
            if tk and tk not in names:
                names.append(tk)
    return names


def glance(sections: List[Dict[str, Any]], quotes: Optional[Dict[str, Dict[str, Any]]] = None,
           watched: Any = ()) -> str:
    """'At a glance' band: how many names fired, the strongest three, watchlist names."""
    from ui import email_kit as k

    sections = drop_covered_breakouts(sections)
    names = _all_tickers(sections)
    if not names:
        return ""
    quotes = quotes or {}
    parts = []
    if len(sections) == 1 and sections[0]["type"] == "breakout":
        rule = sections[0]["heading"].split(" · ", 1)[-1]
        parts.append(f"<b style='color:#ffffff'>{len(names)} stock{'s' if len(names) != 1 else ''}</b> "
                     f"cleared {k.esc(rule)}.")
    else:
        parts.append(f"<b style='color:#ffffff'>{len(names)} stock{'s' if len(names) != 1 else ''}</b> "
                     f"in {len(sections)} alert{'s' if len(sections) != 1 else ''}.")

    def score(t: str) -> float:
        try:
            return float((quotes.get(_key(t)) or {}).get("score"))
        except (TypeError, ValueError):
            return -1.0

    if len(names) > 1:
        ranked = sorted(names, key=lambda t: -score(t)) if any(score(t) >= 0 for t in names) else names
        top = []
        for t in ranked[:3]:
            s = score(t)
            top.append(f"{k.esc(t)}{' ' + format(s, 'g') if s >= 0 else ''}")
        lead = "Top HSF Scores: " if any(score(t) >= 0 for t in names) else "Strongest: "
        parts.append(lead + ", ".join(top) + ".")
    watched_keys = {_key(t) for t in watched or ()}
    mine = [t for t in names if _key(t) in watched_keys]
    if mine:
        parts.append("On your watchlist: " + ", ".join(k.esc(t) for t in mine[:5])
                     + ("…" if len(mine) > 5 else "") + ".")
    return k.glance_text(parts)


def title_line(sections: List[Dict[str, Any]]) -> str:
    try:
        from scheduler.morning_digest import et_now

        when = et_now().strftime("%a %b %-d · %-I:%M %p ET")
    except Exception:
        when = ""
    what = sections[0]["label"] if len(sections) == 1 else "Your alerts"
    return f"{_html.escape(what)}" + (f" · {when}" if when else "")


def compose(sections: List[Dict[str, Any]], *, quotes: Optional[Dict[str, Dict[str, Any]]] = None,
            watched: Any = ()) -> Optional[tuple]:
    """(subject, text, html) for one user's email, or None if nothing to send."""
    sections = drop_covered_breakouts(sections)
    if not sections:
        return None
    return subject_for(sections), text_body(sections), html_body(sections, quotes, watched)


# ---- "Alerts that fired today" (evening wrap + Market Brief) ----------------
_RULE = re.compile(r"\(≥ ([0-9.]+)\)")
_MORE = re.compile(r"^…and (\d+) more\.?$")
SUMMARY_NAMES = 5


def _parse_event(message: str) -> Optional[Dict[str, Any]]:
    """A stored alert_events message ('Breakout alert: IOVA: BreakoutScore 118.9
    (≥ 30)\\nKOD: …\\n…and 7 more.') -> {label, rule, tickers, extra}; None if it
    isn't in that shape."""
    label, sep, rest = str(message or "").partition(": ")
    if not sep or not label.endswith("alert"):
        return None
    tickers, extra, rule = [], 0, None
    for ln in rest.split("\n"):
        ln = ln.strip()
        more = _MORE.match(ln)
        if more:
            extra += int(more.group(1))
            continue
        tk, _detail = split_line(ln)
        if not tk:
            continue
        m = _RULE.search(ln)
        if m and rule is None:
            rule = m.group(1)
        if tk not in tickers:
            tickers.append(tk)
    if not tickers:
        return None
    return {"label": label, "rule": rule, "tickers": tickers, "extra": extra}


def summarize_fired(messages: List[str], *, max_items: int = 8) -> List[str]:
    """One short line per alert that fired: 'Breakout alert (≥ 30) · 32 names:
    IOVA, KOD, VICR, GRAL, MXL…'. The same alert firing twice keeps the first
    (newest); a breakout alert whose names all appear in another breakout alert
    is dropped (as in the alert email). Unrecognised messages pass through."""
    parsed: List[Dict[str, Any]] = []
    passthrough: List[str] = []
    for msg in messages or []:
        ev = _parse_event(msg)
        if ev is None:
            if msg and msg not in passthrough:
                passthrough.append(str(msg))
            continue
        if any(p["label"] == ev["label"] and p["rule"] == ev["rule"] for p in parsed):
            continue
        parsed.append(ev)

    def covered(i: int) -> bool:
        ev = parsed[i]
        if not ev["label"].startswith("Breakout") or ev["extra"]:
            return False
        mine = set(ev["tickers"])
        for j, other in enumerate(parsed):
            if j == i or not other["label"].startswith("Breakout"):
                continue
            theirs = set(other["tickers"])
            if mine <= theirs and (mine != theirs or j < i or other["extra"]):
                return True
        return False

    out: List[str] = []
    for i, ev in enumerate(parsed):
        if covered(i):
            continue
        head = ev["label"] + (f" (≥ {ev['rule']})" if ev["rule"] else "")
        total = len(ev["tickers"]) + ev["extra"]
        names = ", ".join(ev["tickers"][:SUMMARY_NAMES])
        if total > SUMMARY_NAMES:
            out.append(f"{head} · {total} names: {names}…")
        else:
            out.append(f"{head} · {names}")
    return (out + passthrough)[:max_items]
