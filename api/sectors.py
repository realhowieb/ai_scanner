"""Ticker -> sector, from sectors.csv (scripts/refresh_sectors.py, refreshed weekly).

Read once per process: about 7,000 short rows, well under a megabyte. A ticker the
file doesn't list has no sector (None), and a missing file means no sectors at all.
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Dict, Optional

SECTORS_PATH = Path(__file__).resolve().parents[1] / "sectors.csv"


def _key(ticker: object) -> str:
    """BRK-B, BRK.B and BRKB match."""
    return "".join(ch for ch in str(ticker or "").upper() if ch.isalnum())


@functools.lru_cache(maxsize=1)
def sector_map() -> Dict[str, str]:
    try:
        with SECTORS_PATH.open(newline="") as fh:
            return {_key(r.get("ticker")): str(r.get("sector") or "").strip()
                    for r in csv.DictReader(fh) if r.get("ticker") and r.get("sector")}
    except OSError:
        return {}


def sector_for(ticker: object) -> Optional[str]:
    return sector_map().get(_key(ticker)) or None
