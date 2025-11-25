from ..db.runs import save_run
from ..data import fetch, prices, filters
from ..scan.breakout import breakout_scanner
from ..data.symbols import get_spy_history

import time
from datetime import datetime, timezone


def run_postmarket(settings) -> int:
    """
    Headless post-market job.
    Pure function: no Streamlit calls. Returns the saved run_id (or -1 on failure).

    Expected settings keys (with safe defaults):
      - universe_name: str (e.g., "sp500", "nasdaq", "custom")
      - min_price: float
      - max_price: float or None
      - include_ta: bool
      - fetch_period: str (e.g., "60d")
      - fetch_interval: str (e.g., "1d")
      - headless_max_workers: int
      - chunk_size: int
      - filter_min_vol: int or None
      - filter_min_avg_vol: int or None
      - filter_min_market_cap: float or None
      - symbols: list[str] (optional override)
    """
    t0 = time.perf_counter()

    # ---------- 1) Resolve universe ----------
    try:
        universe_name = settings.get("universe_name", "sp500")
        symbols_override = settings.get("symbols")
        if symbols_override:
            symbols = list(dict.fromkeys([s.strip().upper() for s in symbols_override if s]))
        else:
            # Try common universe providers in order
            symbols = None
            if hasattr(universe, "get_universe"):
                symbols = universe.get_universe(universe_name)  # type: ignore[attr-defined]
            if symbols is None and hasattr(universe, "sp500"):
                symbols = universe.sp500() if universe_name.lower() == "sp500" else None  # type: ignore[attr-defined]
            if symbols is None and hasattr(universe, "nasdaq100"):
                if universe_name.lower() in {"nasdaq", "nasdaq100", "ndx"}:
                    symbols = universe.nasdaq100()  # type: ignore[attr-defined]
            if symbols is None and hasattr(universe, "all_equities"):
                symbols = universe.all_equities()  # type: ignore[attr-defined]
            if symbols is None:
                symbols = []
        total_before = len(symbols)
    except Exception:
        # If we cannot resolve, fail fast but safely
        return -1

    if total_before == 0:
        return -1

    # ---------- 2) Fetch price data (headless-friendly) ----------
    fetch_period = settings.get("fetch_period", "60d")
    fetch_interval = settings.get("fetch_interval", "1d")
    headless_workers = settings.get("headless_max_workers", 4)
    chunk_size = settings.get("chunk_size", 70)

    # Pick fetch function that exists in data.fetch
    price_data = {}
    skipped = []

    try:
        # If there is a unified batch fetcher, prefer it
        if hasattr(fetch, "fetch_price_data_batch"):
            price_data, skipped = fetch.fetch_price_data_batch(
                symbols,
                period=fetch_period,
                interval=fetch_interval,
                batch_size=chunk_size,
                headless=True,
                logger=None,
            )
        elif hasattr(fetch, "fetch_price_data_parallel"):
            price_data, skipped = fetch.fetch_price_data_parallel(
                symbols,
                period=fetch_period,
                interval=fetch_interval,
                chunk_size=chunk_size,
                max_workers=headless_workers,
                headless=True,
                logger=None,
            )
        else:
            # Last resort: try a per-symbol function if available
            if hasattr(fetch, "fetch_price_data"):
                for s in symbols:
                    try:
                        df = fetch.fetch_price_data(s, period=fetch_period, interval=fetch_interval)
                        if df is not None and not df.empty:
                            price_data[s] = df
                        else:
                            skipped.append(s)
                    except Exception:
                        skipped.append(s)
            else:
                return -1
    except Exception:
        # fetching blew up
        return -1

    if not price_data:
        # Nothing fetched, nothing to do
        elapsed_s = round(time.perf_counter() - t0, 3)
        meta = {
            "run_kind": "postmarket",
            "universe_name": universe_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "total_before": total_before,
            "fetched": 0,
            "skipped": len(skipped),
        }
        try:
            run_id = save_run(
                run_type="postmarket",
                universe_name=universe_name,
                rows_df=None,
                meta=meta,
            )
            return int(run_id) if run_id is not None else -1
        except Exception:
            return -1

    # ---------- 3) Optional pre-filtering (price/volume/cap) ----------
    try:
        filtered = price_data
        if hasattr(filters, "prefilter"):
            filtered = filters.prefilter(
                price_data,
                min_price=settings.get("min_price"),
                max_price=settings.get("max_price"),
                min_vol=settings.get("filter_min_vol"),
                min_avg_vol=settings.get("filter_min_avg_vol"),
                min_market_cap=settings.get("filter_min_market_cap"),
            )
        elif hasattr(filters, "filter_price_volume_cap"):
            filtered = filters.filter_price_volume_cap(
                price_data,
                min_price=settings.get("min_price"),
                max_price=settings.get("max_price"),
                min_vol=settings.get("filter_min_vol"),
                min_avg_vol=settings.get("filter_min_avg_vol"),
                min_market_cap=settings.get("filter_min_market_cap"),
            )
    except Exception:
        filtered = price_data

    # ---------- 4) Load SPY history for RS calculations ----------
    try:
        spy_df = get_spy_history(period=fetch_period, interval=fetch_interval)
    except Exception:
        spy_df = None

    # ---------- 5) Run breakout scan ----------
    try:
        min_price = settings.get("min_price", 1.0)
        max_price = settings.get("max_price")
        include_ta = settings.get("include_ta", True)
        results_df = breakout_scanner(
            filtered,
            min_price=min_price,
            max_price=max_price,
            include_ta=include_ta,
            spy_df=spy_df,
        )
    except Exception:
        results_df = None

    # ---------- 6) Save results to DB ----------
    elapsed_s = round(time.perf_counter() - t0, 3)
    meta = {
        "run_kind": "postmarket",
        "universe_name": universe_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "total_before": total_before,
        "fetched": len(price_data),
        "skipped": len(skipped),
    }

    try:
        run_id = save_run(
            run_type="postmarket",
            universe_name=universe_name,
            rows_df=results_df,
            meta=meta,
        )
        return int(run_id) if run_id is not None else -1
    except Exception:
        return -1