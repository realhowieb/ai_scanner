#!/usr/bin/env python3
"""Refresh sectors.csv (ticker -> sector) from Nasdaq's public stock screener.

The screener lists every US-listed stock on Nasdaq, NYSE and NYSE American with its
sector, so one request covers the whole scan universe. The Scanner's Sector column
and filter read this file (api/sectors.py); a ticker missing from it simply shows
no sector.

Safety: sectors.csv is only rewritten when the fetch yields a plausible list
(>= a minimum count and mega-cap sentinels present), so a bad/empty download can
never wipe it. Prints a diff summary. Non-zero exit on failure.

Usage:  python -m scripts.refresh_sectors
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTORS_PATH = ROOT / "sectors.csv"
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
HEADERS = {
    # Nasdaq refuses requests without a browser-like User-Agent.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                  "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "application/json, text/plain, */*",
}
_MIN_ROWS = 3000
_SENTINELS = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")


def _norm(sym: object) -> str:
    """BRK/B, BRK.B and BRK^B -> BRK-B (the universe files' spelling)."""
    s = str(sym or "").strip().upper()
    for ch in "/.^":
        s = s.replace(ch, "-")
    return s


def parse_screener(payload: dict) -> dict[str, str]:
    """Screener JSON -> {ticker: sector}; rows without a sector are left out."""
    rows = ((payload or {}).get("data") or {}).get("rows") or []
    out: dict[str, str] = {}
    for r in rows:
        sym, sector = _norm(r.get("symbol")), str(r.get("sector") or "").strip()
        if sym and sector:
            out[sym] = sector
    return dict(sorted(out.items()))


def fetch_sectors() -> dict[str, str]:
    import requests

    resp = requests.get(SCREENER_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return parse_screener(resp.json())


def to_csv(sectors: dict[str, str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["ticker", "sector"])
    w.writerows(sectors.items())
    return buf.getvalue()


def main() -> int:
    try:
        sectors = fetch_sectors()
    except Exception as e:
        print(f"[refresh_sectors] fetch/parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    missing = [s for s in _SENTINELS if s not in sectors]
    if len(sectors) < _MIN_ROWS or missing:
        print(f"[refresh_sectors] implausible result (n={len(sectors)}, missing={missing}); "
              "leaving sectors.csv unchanged", file=sys.stderr)
        return 1
    old = set()
    if SECTORS_PATH.exists():
        old = {r["ticker"] for r in csv.DictReader(SECTORS_PATH.read_text().splitlines())}
    SECTORS_PATH.write_text(to_csv(sectors))
    print(f"[refresh_sectors] wrote {len(sectors)} tickers "
          f"(+{len(set(sectors) - old)} added, -{len(old - set(sectors))} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
