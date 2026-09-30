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

MAX_ROWS = 25

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


def html_body(sections: List[Dict[str, Any]]) -> str:
    esc = _html.escape
    out = []
    for sec in sections:
        shown, extra = _shown(sec["lines"])
        rows = "".join(
            "<tr>"
            f"<td style='padding:4px 12px 4px 0;font-weight:700;white-space:nowrap'>{esc(tk)}</td>"
            f"<td style='padding:4px 0;color:#333'>{esc(detail)}</td>"
            "</tr>"
            for tk, detail in map(split_line, shown)
        )
        more = (f"<p style='color:#888;font-size:13px;margin:4px 0 0'>…and {extra} more</p>"
                if extra > 0 else "")
        out.append(
            f"<h3 style='font-size:15px;margin:18px 0 6px'>{esc(sec['heading'])}</h3>"
            f"<table style='border-collapse:collapse;font-size:14px'>{rows}</table>{more}"
        )
    return "".join(out)


def compose(sections: List[Dict[str, Any]]) -> Optional[tuple]:
    """(subject, text, html) for one user's email, or None if nothing to send."""
    sections = drop_covered_breakouts(sections)
    if not sections:
        return None
    return subject_for(sections), text_body(sections), html_body(sections)


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
