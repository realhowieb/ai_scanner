#!/usr/bin/env python3
"""Refresh sp500.txt from the live Wikipedia S&P 500 constituents table.

Wikipedia's "List of S&P 500 companies" is well-maintained and updated promptly
when the index changes. The page has TWO tables — the current constituents AND a
"selected changes" table that lists *old* tickers (e.g. FISV). We must read only
the constituents table (the one with both 'Symbol' and 'Security' columns), or
we'd pull in stale symbols.

Class-share dots are converted to the dash form the scanner uses (BRK.B->BRK-B).
Always run with --dry-run first and review the diff before writing.

Requires pandas + lxml (present in the deploy env; not in all local envs).

Usage:
    python scripts/refresh_sp500.py --dry-run   # show diff only (recommended first)
    python scripts/refresh_sp500.py             # write sp500.txt after reviewing
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "sp500.txt"


def fetch_symbols() -> list[str]:
    import pandas as pd  # deferred: needs pandas + lxml

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (hsfinest)"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - trusted URL
        html = resp.read().decode("utf-8")

    tables = pd.read_html(html)
    # The constituents table is the one with BOTH 'Symbol' and 'Security' columns
    # (the changes table has 'Added'/'Removed'/'Date' instead).
    constituents = None
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if "symbol" in cols and "security" in cols:
            constituents = t
            break
    if constituents is None:
        raise RuntimeError("Could not find the S&P 500 constituents table on the page.")

    sym_col = next(c for c in constituents.columns if str(c).strip().lower() == "symbol")
    symbols: set[str] = set()
    for raw in constituents[sym_col].astype(str):
        s = raw.strip().upper().replace(".", "-")
        if s and s != "NAN":
            symbols.add(s)
    return sorted(symbols)


def read_current() -> list[str]:
    if not OUT_FILE.exists():
        return []
    return [
        s.strip().upper()
        for s in OUT_FILE.read_text().splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    try:
        new = fetch_symbols()
    except Exception as e:
        print(f"ERROR fetching S&P 500 list: {type(e).__name__}: {e}")
        return 1
    if not (480 <= len(new) <= 520):
        print(f"ERROR: fetched {len(new)} symbols (expected ~500) — refusing to overwrite.")
        return 1

    old = read_current()
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))

    print(f"Current: {len(old)} tickers  |  Fetched: {len(new)} tickers")
    print(f"Added ({len(added)}): {', '.join(added) or '(none)'}")
    print(f"Removed ({len(removed)}): {', '.join(removed) or '(none)'}")

    if dry_run:
        print("\n--dry-run: sp500.txt NOT written. Review the diff above, then run without --dry-run.")
        return 0

    OUT_FILE.write_text("\n".join(new) + "\n")
    print(f"\nWrote {len(new)} tickers to {OUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
