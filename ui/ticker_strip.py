"""Lightweight scrolling ticker symbols for secondary pages.

This intentionally renders already-known symbols only. It does not fetch quotes
or load watchlists, so pages can paint quickly before slower DB/API work starts.
"""
from __future__ import annotations

from html import escape
from typing import Iterable, Mapping

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs
    st = None  # type: ignore[assignment]

TAPE_GROUP_COUNT = 8


def normalize_tickers(tickers: Iterable[object] | None, limit: int = 80) -> list[str]:
    """Return uppercase, deduped ticker symbols safe for display."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers or []:
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
        if len(out) >= int(limit):
            break
    return out


def ticker_change_map(rows: Iterable[Mapping[str, object]] | None) -> dict[str, float]:
    """Build {TICKER: day_change_pct} from cached quote/scan rows."""
    out: dict[str, float] = {}
    for row in rows or []:
        symbol = str(
            row.get("ticker") or row.get("symbol") or row.get("Ticker") or row.get("Symbol") or ""
        ).strip().upper()
        if not symbol:
            continue
        raw = (
            row.get("chg_pct")
            if row.get("chg_pct") is not None
            else row.get("change_pct")
            if row.get("change_pct") is not None
            else row.get("% Change")
            if row.get("% Change") is not None
            else row.get("Chg %")
        )
        try:
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = raw.strip().replace("%", "").replace("+", "")
                if raw in ("", "—", "-"):
                    continue
            out[symbol] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _item_class(change: float | None) -> str:
    if change is None:
        return "symbol-tape__item symbol-tape__item--flat"
    if change > 0:
        return "symbol-tape__item symbol-tape__item--up"
    if change < 0:
        return "symbol-tape__item symbol-tape__item--down"
    return "symbol-tape__item symbol-tape__item--flat"


def build_ticker_strip_html(
    tickers: Iterable[object] | None,
    *,
    label: str = "Watchlist",
    changes: Mapping[str, float] | None = None,
) -> str:
    symbols = normalize_tickers(tickers)
    if not symbols:
        return ""

    change_by_symbol = {str(k).upper(): v for k, v in (changes or {}).items()}
    items = ""
    for symbol in symbols:
        change = change_by_symbol.get(symbol)
        change_html = ""
        if change is not None:
            change_html = f"<span class='symbol-tape__move'>{float(change):+.2f}%</span>"
        items += (
            f"<span class='{_item_class(change)}'>"
            f"<span>{escape(symbol)}</span>{change_html}</span>"
        )
    groups = "".join(
        f"<div class='symbol-tape__group'{' aria-hidden=\"true\"' if i else ''}>{items}</div>"
        for i in range(TAPE_GROUP_COUNT)
    )
    safe_label = escape(label)
    return f"""
    <style>
    .symbol-tape-wrap {{
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        overflow: hidden;
        margin: 4px 0 12px;
        padding: 6px 0;
        border-top: 1px solid rgba(148, 163, 184, 0.24);
        border-bottom: 1px solid rgba(148, 163, 184, 0.24);
    }}
    .symbol-tape__label {{
        flex: 0 0 auto;
        color: #9ca3af;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
    }}
    .symbol-tape__track {{
        display: inline-flex;
        min-width: max-content;
        white-space: nowrap;
        animation: symbol-tape-scroll 32s linear infinite;
    }}
    .symbol-tape__group {{
        display: inline-flex;
        gap: 10px;
        padding-right: 10px;
    }}
    .symbol-tape__item {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        min-height: 26px;
        padding: 3px 10px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.55);
        color: #e5e7eb;
        font-size: 13px;
        font-weight: 700;
    }}
    .symbol-tape__item--up {{
        border-color: rgba(34, 197, 94, 0.48);
        background: rgba(22, 163, 74, 0.20);
        color: #dcfce7;
    }}
    .symbol-tape__item--down {{
        border-color: rgba(248, 113, 113, 0.48);
        background: rgba(220, 38, 38, 0.20);
        color: #fee2e2;
    }}
    .symbol-tape__move {{
        font-size: 12px;
        opacity: 0.9;
    }}
    .symbol-tape-wrap:hover .symbol-tape__track {{
        animation-play-state: paused;
    }}
    @keyframes symbol-tape-scroll {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(calc(-100% / {TAPE_GROUP_COUNT}), 0, 0); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .symbol-tape__track {{ animation: none; overflow-x: auto; max-width: 100%; }}
    }}
    </style>
    <div class="symbol-tape-wrap" aria-label="{safe_label} ticker strip">
      <span class="symbol-tape__label">{safe_label}</span>
      <div style="overflow:hidden; width:100%;">
        <div class="symbol-tape__track">{groups}</div>
      </div>
    </div>
    """


def render_ticker_strip(
    tickers: Iterable[object] | None,
    *,
    label: str = "Watchlist",
    quote_rows: Iterable[Mapping[str, object]] | None = None,
) -> None:
    """Render a non-blocking ticker symbol strip when symbols are available."""
    if st is None:
        return
    html = build_ticker_strip_html(tickers, label=label, changes=ticker_change_map(quote_rows))
    if html:
        st.markdown(html, unsafe_allow_html=True)
