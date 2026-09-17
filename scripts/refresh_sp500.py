#!/usr/bin/env python3
"""Refresh sp500.txt from the SPDR S&P 500 ETF (SPY) daily holdings file.

SPY is the fund that tracks the S&P 500, so its published holdings are the
authoritative constituent list — far more reliable than scraping Wikipedia
(which returned garbled tickers). The holdings come as an .xlsx, parsed here with
the standard library only (zipfile + ElementTree) so no new runtime dependency
is added.

Safety: the file is only rewritten when the fetch yields a plausible list (>= a
minimum count and the mega-cap sentinels present), so a bad/empty download can
never corrupt sp500.txt. Prints a diff summary. Non-zero exit on failure.

Usage:  python -m scripts.refresh_sp500
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP500_PATH = ROOT / "sp500.txt"
SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_TICKER_RE = re.compile(r"[A-Z][A-Z\-]{0,6}")
# Sanity floor + sentinels: never overwrite the file unless the parse is clearly
# a real S&P 500 list.
_MIN_TICKERS = 480
_SENTINELS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "BRK-B")


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def parse_spy_tickers(xlsx_bytes: bytes) -> list[str]:
    """Parse SPY holdings .xlsx bytes -> sorted, de-duped equity tickers."""
    import xml.etree.ElementTree as ET

    z = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in sroot.findall(f"{_NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))

    sheet_name = next(n for n in z.namelist()
                      if re.match(r"xl/worksheets/sheet1\.xml$", n))
    root = ET.fromstring(z.read(sheet_name))

    def cell_value(c) -> str:
        t = c.get("t")
        v = c.find(f"{_NS}v")
        inline = c.find(f"{_NS}is")
        if t == "s" and v is not None:
            try:
                return shared[int(v.text)]
            except (ValueError, IndexError):
                return ""
        if inline is not None:
            return "".join(x.text or "" for x in inline.iter(f"{_NS}t"))
        return v.text if v is not None and v.text else ""

    rows = list(root.iter(f"{_NS}row"))
    parsed = []
    for row in rows:
        cells = {}
        for c in row.findall(f"{_NS}c"):
            ref = c.get("r") or ""
            m = re.match(r"[A-Z]+", ref)
            if m:
                cells[m.group()] = cell_value(c)
        parsed.append(cells)

    # Locate the header row + the 'Ticker' column.
    hdr_idx = None
    ticker_col = None
    for i, cells in enumerate(parsed):
        for col, val in cells.items():
            if str(val).strip().lower() == "ticker":
                hdr_idx, ticker_col = i, col
                break
        if hdr_idx is not None:
            break
    if ticker_col is None:
        raise ValueError("no 'Ticker' column found in SPY holdings file")

    syms = set()
    for cells in parsed[hdr_idx + 1:]:
        s = _norm(cells.get(ticker_col, ""))
        if _TICKER_RE.fullmatch(s):
            syms.add(s)
    return sorted(syms)


def fetch_spy_tickers() -> list[str]:
    import requests

    resp = requests.get(SPY_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return parse_spy_tickers(resp.content)


def main() -> int:
    try:
        tickers = fetch_spy_tickers()
    except Exception as e:  # network / parse failure -> leave file untouched
        print(f"[refresh_sp500] fetch/parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    missing = [s for s in _SENTINELS if s not in tickers]
    if len(tickers) < _MIN_TICKERS or missing:
        print(f"[refresh_sp500] implausible result (n={len(tickers)}, missing={missing}); "
              "leaving sp500.txt unchanged", file=sys.stderr)
        return 1

    old = {ln.strip() for ln in SP500_PATH.read_text().splitlines() if ln.strip()} \
        if SP500_PATH.exists() else set()
    new = set(tickers)
    SP500_PATH.write_text("\n".join(tickers) + "\n")
    print(f"[refresh_sp500] wrote {len(tickers)} tickers "
          f"(+{len(new - old)} added, -{len(old - new)} removed)")
    if new - old:
        print("  added:", ", ".join(sorted(new - old)))
    if old - new:
        print("  removed:", ", ".join(sorted(old - new)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
