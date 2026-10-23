"""Email-safe building blocks for the HSF emails (inline styles, table layout).

Every piece returns an HTML fragment with inline styles only, so it renders the
same in Gmail, Apple Mail and Outlook (no <style> block, no flexbox). The look:
white cards on a light grey page, fixed column widths so numbers line up across
tables, colored score badges and small tags.
"""
from __future__ import annotations

import html as _html
from typing import Any, Iterable, List, Optional, Sequence, Tuple

FONT = "-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
INK = "#0f172a"
MUTED = "#64748b"
FAINT = "#94a3b8"
LINE = "#e2e8f0"
UP = "#16a34a"
DOWN = "#dc2626"
PAGE_BG = "#eef1f5"
HEADER_BG = "#0b1220"
GLANCE_BG = "#111a2e"
LINK = "#2563eb"

# Ticker / Last / move / extra: the same widths in every four-column table.
WIDTHS_4 = (38, 20, 20, 22)
WIDTHS_3 = (40, 28, 32)


def esc(v: Any) -> str:
    return _html.escape(str(v), quote=True)


def num(v: Any, *, money: bool = False) -> str:
    """'1,234.50' (or '$1,234.50'); '—' when missing."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return f"<span style='color:{FAINT}'>—</span>"
    s = f"{f:,.2f}"
    return f"${s}" if money else s


def pct(v: Any, *, zero_is_missing: bool = False) -> str:
    """Green / red '+1.23%' / '-1.23%'; a muted '—' when missing (or exactly zero
    when zero_is_missing, e.g. a pre-market move with no trade yet)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return f"<span style='color:{FAINT}'>—</span>"
    if zero_is_missing and f == 0:
        return f"<span style='color:{FAINT}'>—</span>"
    color = UP if f >= 0 else DOWN
    return f"<span style='color:{color}'>{f:+.2f}%</span>"


def score_badge(score: Any) -> str:
    """HSF Score as a small colored badge: green 60+, amber 40-59, grey below."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return f"<span style='color:{FAINT}'>—</span>"
    bg, fg = ("#dcfce7", "#166534") if s >= 60 else (("#fef9c3", "#854d0e") if s >= 40 else ("#f1f5f9", "#475569"))
    return (f"<span style='display:inline-block;min-width:26px;text-align:center;border-radius:6px;"
            f"padding:1px 6px;font-weight:700;font-size:13px;background:{bg};color:{fg}'>{s:g}</span>")


def tag(text: str, kind: str = "earnings") -> str:
    bg, fg = {"earnings": ("#fef3c7", "#92400e"), "watch": ("#e0e7ff", "#3730a3")}.get(kind, ("#f1f5f9", "#475569"))
    return (f" <span style='display:inline-block;font-size:11px;font-weight:600;border-radius:999px;"
            f"padding:1px 7px;margin-left:4px;background:{bg};color:{fg};white-space:nowrap'>{esc(text)}</span>")


def ticker(t: Any, *, extra: str = "") -> str:
    return f"<strong>{esc(t)}</strong>{extra}"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]], widths: Optional[Sequence[int]] = None) -> str:
    """A data table. Cells are pre-rendered HTML; the first column is left-aligned,
    the rest right-aligned with tabular figures. Empty headers hide the header row."""
    widths = widths or (WIDTHS_4 if len(headers) == 4 else WIDTHS_3 if len(headers) == 3 else ())
    cols = "".join(f"<col style='width:{w}%'>" for w in widths)
    th = ""
    if any(headers):
        th = "<tr>" + "".join(
            f"<th align='{'left' if i == 0 else 'right'}' style='font-size:11px;font-weight:600;color:{MUTED};"
            f"letter-spacing:.03em;padding:6px 4px;border-bottom:1px solid {LINE}'>{h}</th>"
            for i, h in enumerate(headers)) + "</tr>"
    body = "".join(
        "<tr>" + "".join(
            f"<td align='{'left' if i == 0 else 'right'}' style='padding:7px 4px;border-bottom:1px solid #f1f5f9;"
            f"font-size:14px;font-variant-numeric:tabular-nums'>{c}</td>"
            for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
            f"style='border-collapse:collapse;table-layout:fixed;width:100%;color:{INK}'>"
            f"<colgroup>{cols}</colgroup>{th}{body}</table>")


def group(label: str) -> str:
    return f"<p style='margin:12px 0 2px;font-size:13px;font-weight:700;color:#334155'>{label}</p>"


def note(text: str) -> str:
    return f"<p style='margin:8px 0 0;font-size:12px;color:{MUTED};line-height:1.45'>{text}</p>"


def empty(text: str) -> str:
    return f"<p style='margin:4px 0 0;font-size:14px;color:{MUTED}'>{text}</p>"


def chips(items: Iterable[Tuple[str, Optional[str]]]) -> str:
    """Pill list: [(label, value_html_or_None)]."""
    out = []
    for label, value in items:
        v = f" {value}" if value else ""
        out.append(f"<span style='display:inline-block;background:#f1f5f9;border-radius:999px;padding:3px 10px;"
                   f"margin:3px 4px 0 0;font-size:13px;color:{INK}'>{esc(label)}{v}</span>")
    return f"<div style='margin:2px 0 0'>{''.join(out)}</div>"


def chip_score(v: Any) -> str:
    try:
        return f"<b style='color:#166534'>{float(v):g}</b>"
    except (TypeError, ValueError):
        return ""


def button(url: str, label: str) -> str:
    return (f"<p style='margin:14px 0 0'><a href='{esc(url)}' style='display:inline-block;background:{LINK};"
            f"color:#ffffff;text-decoration:none;font-weight:600;font-size:14px;border-radius:8px;"
            f"padding:9px 16px'>{label}</a></p>")


def card(title: str, body: str, sub: str = "") -> str:
    sub_html = f"<p style='margin:2px 0 10px;font-size:12px;color:{MUTED}'>{sub}</p>" if sub else \
        "<div style='height:8px'></div>"
    return ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' style='margin:12px 0 0'>"
            f"<tr><td style='background:#ffffff;border:1px solid {LINE};border-radius:12px;padding:16px 20px'>"
            f"<h3 style='margin:0;font-size:16px;color:{INK}'>{title}</h3>{sub_html}{body}"
            "</td></tr></table>")


def is_kit(html_inner: str) -> bool:
    """True when the fragment is already built from cards (so the shell doesn't wrap it again)."""
    return "border-radius:12px;padding:16px 20px" in html_inner


def shell(html_inner: str, *, title_line: str = "", glance: str = "", footer_html: str = "") -> str:
    """Whole email: dark header (brand + title line), optional 'At a glance' band,
    then the body (cards, or a plain fragment wrapped in one card)."""
    sub = f"<div style='color:{FAINT};font-size:13px;margin-top:2px'>{title_line}</div>" if title_line else ""
    radius = "12px 12px 0 0" if glance else "12px"
    head = (f"<tr><td style='background:{HEADER_BG};color:#ffffff;border-radius:{radius};padding:18px 22px'>"
            f"<div style='font-size:18px;font-weight:700'>⚡ HSFinest.AI</div>{sub}</td></tr>")
    if glance:
        head += (f"<tr><td style='background:{GLANCE_BG};color:#e2e8f0;border-radius:0 0 12px 12px;"
                 f"padding:14px 22px;font-size:14px;line-height:1.55'>{glance}</td></tr>")
    body = html_inner if is_kit(html_inner) else (
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' style='margin:12px 0 0'>"
        f"<tr><td style='background:#ffffff;border:1px solid {LINE};border-radius:12px;padding:16px 20px;"
        f"color:{INK};font-size:14px;line-height:1.5'>{html_inner}</td></tr></table>")
    return (f"<div style='background:{PAGE_BG};padding:20px 10px;font-family:{FONT};color:{INK}'>"
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
            "style='max-width:600px;margin:0 auto'><tr><td>"
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0'>"
            f"{head}</table>{body}"
            f"<div style='color:{FAINT};font-size:11px;text-align:center;padding:16px 10px;line-height:1.5'>"
            f"{footer_html}</div></td></tr></table></div>")


def split_flag(t: Any) -> Tuple[str, Optional[int]]:
    """'AMT ⚠️E4d' -> ('AMT', 4): the earnings flag the digest appends to tickers."""
    s = str(t or "")
    if " ⚠️E" in s and s.endswith("d"):
        base, _, flag = s.partition(" ⚠️E")
        try:
            return base, int(flag[:-1])
        except ValueError:
            return base, None
    return s, None


def earnings_tag(days: Optional[int], today=None) -> str:
    """'Earnings today' / 'Earnings tomorrow' / 'Earnings Tue' for a days-until value."""
    if days is None:
        return ""
    if days == 0:
        return tag("Earnings today")
    if days == 1:
        return tag("Earnings tomorrow")
    import datetime as dt

    base = today or dt.date.today()
    return tag(f"Earnings {(base + dt.timedelta(days=days)).strftime('%a')}")


def flagged_ticker(t: Any, today=None) -> str:
    """Ticker cell for a row whose ticker may carry the ⚠️E{n}d flag."""
    base, days = split_flag(t)
    return ticker(base, extra=earnings_tag(days, today))


def glance_text(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    return ("<b style='color:#ffffff'>At a glance:</b> " + " ".join(parts)) if parts else ""
