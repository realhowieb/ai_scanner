"""Pre-scan liquidity trim for large manual scans (Combo, US market).

Before downloading 60 days of bars for thousands of symbols, one Alpaca
snapshot per symbol (batched, a few parallel requests) drops names that can't
pass the scan's own price and dollar-volume rules. The scan engine still applies
its exact rules afterwards (scan.breakout: last close in [min_price, max_price],
20-day average dollar volume >= the floor); this step only skips downloads.

It is deliberately loose so it never removes a stock the engine would keep:
  * price: last price within [min_price * 0.9, max_price * 1.1];
  * dollar volume: the better of today's and the previous session's
    close x volume must reach a quarter of the floor (the engine uses a
    20-day average; a quarter allows for a quiet day or two);
  * a symbol the snapshot response omits is dropped (no Alpaca data, so the
    bar download would fall back to slow one-by-one Yahoo calls);
  * a batch whose request fails keeps all of its symbols, and no Alpaca
    configuration keeps the whole list.
Volumes come from the same Alpaca feed the bar download uses, so the trim and
the engine measure the same thing.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

BATCH = 200          # symbols per snapshot request
WORKERS = 4          # parallel requests (well under Alpaca's per-minute limit)
PRICE_MARGIN = 0.10
VOLUME_FRACTION = 0.25

# fetch(batch) -> {symbol: snapshot dict}, or None when the request failed
SnapshotFetch = Callable[[List[str]], Optional[Dict[str, dict]]]


def _alpaca_fetch() -> Optional[SnapshotFetch]:
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_data_feed, get_alpaca_headers
    except Exception:
        return None
    cfg, headers = get_alpaca_config(), get_alpaca_headers()
    if not cfg or not headers:
        return None
    url = f"{cfg['data_url']}/v2/stocks/snapshots"
    feed = get_alpaca_data_feed()

    def fetch(batch: List[str]) -> Optional[Dict[str, dict]]:
        # Alpaca writes class shares with a dot (BRK.B); the universe uses a dash.
        to_orig = {s.replace("-", "."): s for s in batch}
        try:
            r = requests.get(url, headers=headers, timeout=15,
                             params={"symbols": ",".join(to_orig), "feed": feed})
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return {to_orig.get(k.upper(), k.upper()): v for k, v in data.items() if isinstance(v, dict)}

    return fetch


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None


def _last_price(snap: dict) -> Optional[float]:
    for key, field in (("latestTrade", "p"), ("dailyBar", "c"), ("prevDailyBar", "c")):
        p = _num((snap.get(key) or {}).get(field))
        if p:
            return p
    return None


def _best_dollar_volume(snap: dict) -> float:
    best = 0.0
    for key in ("dailyBar", "prevDailyBar"):
        bar = snap.get(key) or {}
        c, v = _num(bar.get("c")), _num(bar.get("v"))
        if c and v:
            best = max(best, c * v)
    return best


def keep_symbol(snap: dict, *, min_price: float, max_price: float, min_avg_dollar_vol: float) -> bool:
    """True when the engine could still keep this symbol (see module docstring)."""
    price = _last_price(snap)
    if price is None:
        return False
    if min_price and price < float(min_price) * (1 - PRICE_MARGIN):
        return False
    if max_price and price > float(max_price) * (1 + PRICE_MARGIN):
        return False
    if min_avg_dollar_vol and min_avg_dollar_vol > 0:
        return _best_dollar_volume(snap) >= float(min_avg_dollar_vol) * VOLUME_FRACTION
    return True


def apply_liquidity_filter_batch(
    tickers: Sequence[str],
    *,
    min_price: float,
    min_avg_dollar_vol: float,
    max_price: float = 0.0,
    fetch: Optional[SnapshotFetch] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[str]:
    """The tickers worth downloading, in their original order."""
    symbols = [s for s in dict.fromkeys(str(t).strip().upper() for t in tickers or []) if s]
    if not symbols:
        return []
    fetch = fetch or _alpaca_fetch()
    if fetch is None:
        return symbols
    batches = [symbols[i:i + BATCH] for i in range(0, len(symbols), BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(fetch, batches))

    keep: set = set()
    failed_batches = dropped_missing = dropped_rules = 0
    for batch, snaps in zip(batches, results):
        if snaps is None:          # request failed: keep the whole batch
            failed_batches += 1
            keep.update(batch)
            continue
        for sym in batch:
            snap = snaps.get(sym)
            if snap is None:
                dropped_missing += 1
            elif keep_symbol(snap, min_price=min_price, max_price=max_price,
                             min_avg_dollar_vol=min_avg_dollar_vol):
                keep.add(sym)
            else:
                dropped_rules += 1
    if failed_batches == len(batches):   # nothing answered: don't trim at all
        return symbols
    out = [s for s in symbols if s in keep]
    counts = {"requested": len(symbols), "kept": len(out), "failed_batches": failed_batches,
              "dropped_no_data": dropped_missing, "dropped_price_or_volume": dropped_rules}
    if stats is not None:
        stats.update(counts)
    logger.info("liquidity pre-filter: %s", counts)
    return out
