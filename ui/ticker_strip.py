"""Lightweight scrolling ticker symbols for secondary pages.

This intentionally renders already-known symbols only. It does not fetch quotes
or load watchlists, so pages can paint quickly before slower DB/API work starts.
"""
from __future__ import annotations

from html import escape
from typing import Iterable

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs
    st = None  # type: ignore[assignment]


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


def build_ticker_strip_html(
    tickers: Iterable[object] | None,
    *,
    label: str = "Watchlist",
) -> str:
    symbols = normalize_tickers(tickers)
    if not symbols:
        return ""

    items = "".join(
        f"<span class='symbol-tape__item'>{escape(symbol)}</span>" for symbol in symbols
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
        gap: 10px;
        min-width: max-content;
        white-space: nowrap;
        animation: symbol-tape-scroll 38s linear infinite;
    }}
    .symbol-tape__item {{
        display: inline-flex;
        align-items: center;
        min-height: 26px;
        padding: 3px 10px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.55);
        color: #e5e7eb;
        font-size: 13px;
        font-weight: 700;
    }}
    @keyframes symbol-tape-scroll {{
        0% {{ transform: translate3d(0, 0, 0); }}
        100% {{ transform: translate3d(-50%, 0, 0); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .symbol-tape__track {{ animation: none; overflow-x: auto; max-width: 100%; }}
    }}
    </style>
    <div class="symbol-tape-wrap" aria-label="{safe_label} ticker strip">
      <span class="symbol-tape__label">{safe_label}</span>
      <div style="overflow:hidden; width:100%;">
        <div class="symbol-tape__track">{items}{items}</div>
      </div>
    </div>
    """


def render_ticker_strip(
    tickers: Iterable[object] | None,
    *,
    label: str = "Watchlist",
) -> None:
    """Render a non-blocking ticker symbol strip when symbols are available."""
    if st is None:
        return
    html = build_ticker_strip_html(tickers, label=label)
    if html:
        st.markdown(html, unsafe_allow_html=True)
