"""Ticker -> sector, industry and market cap, from sectors.csv (scripts/refresh_sectors.py,
refreshed weekly).

Read once per process: about 7,000 short rows, well under a megabyte. A ticker the
file doesn't list has no sector (None), and a missing file means no sectors at all.
Older files carry only ticker,sector; industry and market cap are then None.
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SECTORS_PATH = Path(__file__).resolve().parents[1] / "sectors.csv"

# Size buckets by market cap (dollars), the usual U.S. cut-offs: (id, label, low, high).
CAP_BUCKETS: List[Tuple[str, str, float, Optional[float]]] = [
    ("micro", "Micro cap (under $300M)", 0.0, 3e8),
    ("small", "Small cap ($300M-$2B)", 3e8, 2e9),
    ("mid", "Mid cap ($2B-$10B)", 2e9, 1e10),
    ("large", "Large cap ($10B-$200B)", 1e10, 2e11),
    ("mega", "Mega cap ($200B+)", 2e11, None),
]
CAP_IDS = tuple(b[0] for b in CAP_BUCKETS)


def _key(ticker: object) -> str:
    """BRK-B, BRK.B and BRKB match."""
    return "".join(ch for ch in str(ticker or "").upper() if ch.isalnum())


def _cap(value: object) -> Optional[float]:
    try:
        v = float(str(value or "").strip())
    except ValueError:
        return None
    return v if v > 0 else None


@functools.lru_cache(maxsize=1)
def facts_map() -> Dict[str, Tuple[str, Optional[str], Optional[float]]]:
    """{ticker key: (sector, industry or None, market cap or None)}."""
    try:
        with SECTORS_PATH.open(newline="") as fh:
            return {_key(r.get("ticker")): (str(r.get("sector") or "").strip(),
                                            str(r.get("industry") or "").strip() or None,
                                            _cap(r.get("market_cap")))
                    for r in csv.DictReader(fh) if r.get("ticker") and r.get("sector")}
    except OSError:
        return {}


@functools.lru_cache(maxsize=1)
def sector_map() -> Dict[str, str]:
    return {k: v[0] for k, v in facts_map().items() if v[0]}


def sector_for(ticker: object) -> Optional[str]:
    return sector_map().get(_key(ticker)) or None


def market_cap_for(ticker: object) -> Optional[float]:
    f = facts_map().get(_key(ticker))
    return f[2] if f else None


def cap_bucket(cap: Optional[float]) -> Optional[str]:
    """The size bucket id for a market cap, or None when it's unknown."""
    if cap is None:
        return None
    for bid, _label, lo, hi in CAP_BUCKETS:
        if cap >= lo and (hi is None or cap < hi):
            return bid
    return None


def company_facts(ticker: object) -> Optional[Dict[str, object]]:
    """{sector, industry, market_cap, size} for the stock page; None when the file doesn't list it."""
    f = facts_map().get(_key(ticker))
    if not f:
        return None
    bucket = cap_bucket(f[2])
    label = next((b[1] for b in CAP_BUCKETS if b[0] == bucket), None)
    return {"sector": f[0] or None, "industry": f[1], "market_cap": f[2],
            "size": label.split(" (")[0] if label else None}
