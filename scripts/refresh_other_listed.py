#!/usr/bin/env python3
"""Refresh other_listed.txt: common stocks listed on NYSE, NYSE American, NYSE Arca
and Cboe (every U.S. exchange except Nasdaq, which nasdaq.txt covers).

Source: nasdaqtrader.com's ``otherlisted.txt``, the same official symbol directory
family as ``nasdaqlisted.txt``. Test issues and ETFs are flagged in their own
columns; funds, ETNs, preferreds, units, warrants and rights are dropped by name
or symbol. Class shares are written in the dash form the universe files use
(BRK.B -> BRK-B).

Safety: other_listed.txt is only rewritten when the parse looks like a real
exchange list (a minimum count and big NYSE names present), so a bad download can
never corrupt it. Non-zero exit on failure.

Usage:  python -m scripts.refresh_other_listed
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OTHER_PATH = ROOT / "other_listed.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_BAD_NAME = re.compile(
    r"(Warrant|\bUnits?\b|\bRights?\b|Depositary Shares|Preferred|\bNotes?\b|Debenture|Subordinated|"
    r"\bETN\b|Exchange[- ]Traded|\bFund\b|% )", re.I
)
# Plain tickers, optionally with a one-letter share class (BRK.B). "$" marks preferreds.
_SYMBOL = re.compile(r"[A-Z]{1,5}(\.[A-Z])?")
_MIN_TICKERS = 1500
_SENTINELS = ("JPM", "XOM", "WMT", "GME", "SNOW", "BRK-B")


def parse_other_listed(text: str) -> list[str]:
    """otherlisted.txt content -> sorted, de-duped common-stock tickers (dash class form).

    Columns: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"""
    out: set[str] = set()
    for line in text.splitlines()[1:]:
        if line.startswith("File Creation Time") or "|" not in line:
            continue
        f = line.split("|")
        if len(f) < 7:
            continue
        sym, name, _exch, _cqs, etf, _lot, test = (x.strip() for x in f[:7])
        if test != "N" or etf != "N" or _BAD_NAME.search(name):
            continue
        s = sym.upper()
        if not _SYMBOL.fullmatch(s):
            continue
        out.add(s.replace(".", "-"))
    return sorted(out)


def fetch_other_tickers() -> list[str]:
    import requests

    resp = requests.get(OTHER_LISTED_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return parse_other_listed(resp.text)


def main() -> int:
    try:
        tickers = fetch_other_tickers()
    except Exception as e:
        print(f"[refresh_other_listed] fetch/parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    missing = [s for s in _SENTINELS if s not in tickers]
    if len(tickers) < _MIN_TICKERS or missing:
        print(f"[refresh_other_listed] implausible result (n={len(tickers)}, missing={missing}); "
              "leaving other_listed.txt unchanged", file=sys.stderr)
        return 1

    old = {ln.strip() for ln in OTHER_PATH.read_text().splitlines() if ln.strip()} \
        if OTHER_PATH.exists() else set()
    new = set(tickers)
    OTHER_PATH.write_text("\n".join(tickers) + "\n")
    print(f"[refresh_other_listed] wrote {len(tickers)} tickers "
          f"(+{len(new - old)} added, -{len(old - new)} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
