#!/usr/bin/env python3
"""Refresh nasdaq.txt from the official Nasdaq listed-securities directory.

Source: nasdaqtrader.com's ``nasdaqlisted.txt`` — the authoritative, structured
(pipe-delimited) list of Nasdaq-listed securities. We keep active common stocks
only: test issues and ETFs are flagged in dedicated columns, and units / warrants
/ rights are dropped by security name and ticker suffix.

Safety: nasdaq.txt is only rewritten when the parse yields a plausible list
(>= a minimum count and mega-cap sentinels present), so a bad/empty download can
never corrupt the file. Prints a diff summary. Non-zero exit on failure.

Usage:  python -m scripts.refresh_nasdaq
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NASDAQ_PATH = ROOT / "nasdaq.txt"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

# Non-common-stock securities to drop by security-name keyword.
_BAD_NAME = re.compile(
    r"(Warrant|Unit|Right|Depositary|Preferred|Note|Debenture|Subordinated|% )", re.I
)
# Plausibility guard.
_MIN_TICKERS = 2000
_SENTINELS = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META")


def parse_nasdaq_listed(text: str) -> list[str]:
    """Parse nasdaqlisted.txt content -> sorted, de-duped common-stock tickers."""
    out: set[str] = set()
    lines = text.splitlines()
    for line in lines[1:]:  # skip header row
        if line.startswith("File Creation Time") or "|" not in line:
            continue
        f = line.split("|")
        if len(f) < 8:
            continue
        sym, name, _mkt, test, _fin, _lot, etf, _ns = f[:8]
        if test.strip() != "N" or etf.strip() != "N":
            continue
        if _BAD_NAME.search(name):
            continue
        s = sym.strip().upper()
        # Plain common tickers 1-4 chars, or 5-char NOT ending in U/W/R
        # (unit / warrant / right suffixes).
        if re.fullmatch(r"[A-Z]{1,4}", s) or (
            re.fullmatch(r"[A-Z]{5}", s) and s[-1] not in "UWR"
        ):
            out.add(s)
    return sorted(out)


def fetch_nasdaq_tickers() -> list[str]:
    import requests

    resp = requests.get(NASDAQ_LISTED_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return parse_nasdaq_listed(resp.text)


def main() -> int:
    try:
        tickers = fetch_nasdaq_tickers()
    except Exception as e:
        print(f"[refresh_nasdaq] fetch/parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    missing = [s for s in _SENTINELS if s not in tickers]
    if len(tickers) < _MIN_TICKERS or missing:
        print(f"[refresh_nasdaq] implausible result (n={len(tickers)}, missing={missing}); "
              "leaving nasdaq.txt unchanged", file=sys.stderr)
        return 1

    old = {ln.strip() for ln in NASDAQ_PATH.read_text().splitlines() if ln.strip()} \
        if NASDAQ_PATH.exists() else set()
    new = set(tickers)
    NASDAQ_PATH.write_text("\n".join(tickers) + "\n")
    print(f"[refresh_nasdaq] wrote {len(tickers)} tickers "
          f"(+{len(new - old)} added, -{len(old - new)} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
