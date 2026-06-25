from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from utils.symbols import as_ticker_list, sanitize_ticker_list

from .fetch import (
    fetch_hot_stocks,
    fetch_most_active_stocks,
    fetch_trending_stocks,
    load_sp500_tickers,
    load_sp600_tickers,
)


def fallback_universe(max_size: int = 1500) -> list[str]:
    for loader in (load_sp500_tickers, load_sp600_tickers):
        try:
            lst = sanitize_ticker_list(as_ticker_list(loader()))
            if lst: return lst[:max_size]
        except Exception: pass
    return []

def fetch_market_heat(kind: str, limit: int = 100) -> list[str]:
    kind = (kind or "").lower()
    try:
        if kind == "trending": lst = fetch_trending_stocks()
        elif kind == "most_active": lst = fetch_most_active_stocks()
        elif kind == "gainers": lst = fetch_hot_stocks()
        else: lst = []
        lst = sanitize_ticker_list(as_ticker_list(lst))
        if lst: return lst[:limit]
    except Exception: pass

    universe = fallback_universe()
    if not universe: return []
    df = yf.download(universe, period="2d", interval="1d",
                     auto_adjust=False, progress=False, threads=False, group_by='ticker')
    scores = []
    for t in universe:
        try:
            dft = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            if dft is None or dft.empty or len(dft) < 2: continue
            p0, p1 = float(dft['Close'].iloc[-2]), float(dft['Close'].iloc[-1])
            v1 = float(dft['Volume'].iloc[-1]) if 'Volume' in dft.columns else np.nan
            pct = ((p1 - p0) / p0) * 100 if p0 else np.nan
            dvol = (p1 * v1) if not (np.isnan(p1) or np.isnan(v1)) else np.nan
            if kind == 'gainers':
                if np.isnan(pct) or pct <= 0: continue
                score = pct
            elif kind == 'most_active':
                if np.isnan(dvol): continue
                score = dvol
            else:  # trending
                if np.isnan(pct): continue
                score = abs(pct)
            scores.append((t, score))
        except Exception:
            continue
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_ in scores[:limit]]