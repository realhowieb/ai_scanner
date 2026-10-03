"""P2-72 — import a watchlist from a file (Manage Watchlist → Import from a file).

Reads tickers from:
- a CSV with a Symbol / Ticker column (Robinhood, Schwab, Fidelity, Yahoo and
  most broker exports; preamble lines above the header are skipped),
- a TradingView watchlist export (.txt: NASDAQ:AAPL,NYSE:IBM,###Section),
- any plain list (commas, spaces or one per line).

Tickers are validated with the same rule as the bulk-edit box and capped at
200. The file is only read in memory; only the tickers are saved.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Dict, List

MAX_FILE_BYTES = 512 * 1024
MAX_TICKERS = 200
_HEADER_NAMES = {"symbol", "symbols", "ticker", "tickers", "ticker symbol", "instrument"}
# Same rule as ui.watchlists._TICKER_RE (a test keeps them equal); kept here so
# the parser doesn't import the Streamlit page module.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")
_NOT_TICKERS = {"SYMBOL", "SYMBOLS", "TICKER", "TICKERS", "CASH", "TOTAL", "N/A", "NA"}


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def _clean(token: str) -> str:
    """'NASDAQ:aapl ' -> 'AAPL'; '$TSLA' -> 'TSLA'. Empty for section markers."""
    t = str(token or "").strip().strip('"').strip()
    if not t or t.startswith("###"):
        return ""
    if ":" in t:
        t = t.rsplit(":", 1)[1]
    return t.lstrip("$").strip().upper()


def _csv_column(text: str) -> List[str] | None:
    """Values of the Symbol/Ticker column, or None when no such header exists."""
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return None
    for i, row in enumerate(rows[:50]):  # broker exports put a few title lines first
        names = [str(c or "").strip().lower() for c in row]
        col = next((j for j, n in enumerate(names) if n in _HEADER_NAMES), None)
        if col is not None:
            return [r[col] for r in rows[i + 1:] if len(r) > col]
    return None


def parse_watchlist_file(name: str, data: bytes) -> Dict[str, List[str]]:
    """{'tickers': [...], 'skipped': [...]} from an uploaded file's bytes."""
    if len(data or b"") > MAX_FILE_BYTES:
        raise ValueError("That file is too large (512 KB max).")
    text = _decode(data or b"")
    tokens = _csv_column(text) if Path(str(name or "")).suffix.lower() == ".csv" else None
    if tokens is None:
        tokens = [t for line in text.splitlines() for t in line.replace(";", ",").replace("\t", ",")
                  .replace(" ", ",").split(",")]
    tickers: List[str] = []
    skipped: List[str] = []
    for raw in tokens:
        t = _clean(raw)
        if not t or t in _NOT_TICKERS or " " in t:  # multi-word cells are labels ("Account Total")
            continue
        if not _TICKER_RE.match(t):
            if t not in skipped and len(skipped) < 20:
                skipped.append(t[:20])
            continue
        if t not in tickers:
            tickers.append(t)
    return {"tickers": tickers[:MAX_TICKERS], "skipped": skipped,
            "over_cap": max(0, len(tickers) - MAX_TICKERS)}


def render_watchlist_import(username: str, active_id, active_name: str) -> None:
    """File uploader + preview + 'add to this list' / 'new list'. Never raises."""
    import streamlit as st

    st.markdown("#### Import from a file")
    st.caption("A CSV with a Symbol or Ticker column (most broker exports), a TradingView "
               "watchlist export (.txt), or any list of tickers. Only the tickers are saved.")
    up = st.file_uploader("Watchlist file", type=["csv", "txt"], key="wl_import_file",
                          label_visibility="collapsed")
    if up is None:
        return
    try:
        parsed = parse_watchlist_file(up.name, up.getvalue())
    except ValueError as exc:
        st.warning(str(exc))
        return
    tickers = parsed["tickers"]
    if not tickers:
        st.warning("No tickers found in that file.")
        return
    preview = ", ".join(tickers[:15]) + (f" and {len(tickers) - 15} more" if len(tickers) > 15 else "")
    st.markdown(f"Found **{len(tickers)}** ticker{'s' if len(tickers) != 1 else ''}: {preview}")
    if parsed["over_cap"]:
        st.caption(f"Only the first {MAX_TICKERS} are imported ({parsed['over_cap']} more in the file).")
    if parsed["skipped"]:
        st.caption("Skipped (not a ticker): " + ", ".join(parsed["skipped"]))
    options = (["add", "new"] if active_id is not None else ["new"])
    labels = {"add": f"Add to {active_name}", "new": "Create a new watchlist"}
    choice = st.radio("Import into", options, format_func=labels.get, key="wl_import_target",
                      horizontal=True)
    new_name = ""
    if choice == "new":
        new_name = st.text_input("New watchlist name", value=Path(up.name).stem[:40] or "Imported",
                                 key="wl_import_name")
    if not st.button("Import", key="wl_import_btn"):
        return
    from db.watchlists import add_tickers_to_watchlist, create_watchlist

    try:
        if choice == "new":
            target = create_watchlist(username, new_name.strip())
        else:
            target = int(active_id)
        result = add_tickers_to_watchlist(username, tickers, target)
    except ValueError as exc:  # e.g. a watchlist with that name already exists
        st.warning(str(exc))
        return
    except Exception:
        st.error("Could not import right now.")
        return
    st.session_state["active_watchlist_id"] = target
    added = len(result.get("added") or [])
    already = len(result.get("already_present") or [])
    st.success(f"Imported {added} ticker{'s' if added != 1 else ''}"
               + (f" ({already} already on the list)." if already else "."))
    st.rerun()
