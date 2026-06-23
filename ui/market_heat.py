"""Market heat provider helpers used by the pages UI."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

try:
    from ai_scanner.data.fetch import (
        fetch_hot_stocks,
        fetch_most_active_stocks,
        fetch_trending_stocks,
    )  # type: ignore
except ImportError:  # pragma: no cover - package import path varies locally
    try:
        from data.fetch import (
            fetch_hot_stocks,
            fetch_most_active_stocks,
            fetch_trending_stocks,
        )  # type: ignore
    except ImportError:  # pragma: no cover
        fetch_hot_stocks = None  # type: ignore
        fetch_most_active_stocks = None  # type: ignore
        fetch_trending_stocks = None  # type: ignore

try:
    import requests as _requests
    from requests import exceptions as _requests_exc
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore
    _requests_exc = None  # type: ignore


def _yahoo_to_df(items: List[Dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    rows = []
    for quote in items:
        rows.append(
            {
                "symbol": quote.get("symbol"),
                "name": quote.get("shortName") or quote.get("longName"),
                "price": quote.get("regularMarketPrice"),
                "change": quote.get("regularMarketChange"),
                "change_%": quote.get("regularMarketChangePercent"),
                "volume": quote.get("regularMarketVolume"),
                "marketCap": quote.get("marketCap"),
            }
        )
    df = pd.DataFrame(rows)
    for col in ["price", "change", "change_%", "volume", "marketCap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_predefined_screener(scr_id: str, count: int = 25) -> pd.DataFrame:
    if _requests is None:
        return pd.DataFrame()
    url = (
        "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
        f"?count={count}&scrIds={scr_id}"
    )
    try:
        resp = _requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ai-scanner/1.0)"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (_requests_exc.RequestException, ValueError):  # type: ignore[union-attr]
        return pd.DataFrame()

    quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    return _yahoo_to_df(quotes)


def _fetch_quotes(symbols: List[str]) -> pd.DataFrame:
    if _requests is None or not symbols:
        return pd.DataFrame()
    all_items: List[Dict] = []
    for index in range(0, len(symbols), 50):
        chunk = symbols[index : index + 50]
        url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ",".join(chunk)
        try:
            resp = _requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ai-scanner/1.0)"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (_requests_exc.RequestException, ValueError):  # type: ignore[union-attr]
            continue

        items = data.get("quoteResponse", {}).get("result", [])
        if items:
            all_items.extend(items)
    return _yahoo_to_df(all_items)


def _fetch_trending(region: str = "US", count: int = 25) -> pd.DataFrame:
    if _requests is None:
        return pd.DataFrame()

    df = _fetch_predefined_screener("trending_tickers", count=count)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    def _try_trending(base: str) -> List[str]:
        url = f"{base}/v1/finance/trending/{region}?count={count}"
        try:
            resp = _requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ai-scanner/1.0)",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://finance.yahoo.com/",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (_requests_exc.RequestException, ValueError):  # type: ignore[union-attr]
            return []

        raw = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        symbols: List[str] = []
        for quote in raw:
            if isinstance(quote, dict):
                symbol = quote.get("symbol")
                if symbol:
                    symbols.append(symbol)
            elif isinstance(quote, str):
                symbols.append(quote)
        return [symbol for symbol in symbols if symbol]

    symbols = _try_trending("https://query1.finance.yahoo.com")
    if not symbols:
        symbols = _try_trending("https://query2.finance.yahoo.com")

    if symbols:
        enriched = _fetch_quotes(symbols[:count])
        if isinstance(enriched, pd.DataFrame) and not enriched.empty:
            return enriched

    return _fetch_predefined_screener("most_actives", count=count)


if fetch_hot_stocks is None:
    def fetch_hot_stocks(count: int = 25) -> pd.DataFrame:  # type: ignore[no-redef]
        return _fetch_predefined_screener("day_gainers", count=count)


if fetch_most_active_stocks is None:
    def fetch_most_active_stocks(count: int = 25) -> pd.DataFrame:  # type: ignore[no-redef]
        return _fetch_predefined_screener("most_actives", count=count)


if fetch_trending_stocks is None:
    def fetch_trending_stocks(region: str = "US", count: int = 25) -> pd.DataFrame:  # type: ignore[no-redef]
        return _fetch_trending(region=region, count=count)
