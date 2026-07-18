"""
Centralized market data helpers for the AI Scanner.

This module is responsible for talking to Alpaca's Market Data API and returning
a simple, app-friendly structure for latest quotes that can be reused across:
  - price ticker strip
  - market snapshot (SPY / QQQ, etc.)
  - liquidity filters for large universes
  - extended-hours scans (premarket / after-hours)

It is intentionally written to be:
  - resilient (graceful failure and empty results on config/network issues)
  - cache-friendly (Streamlit @st.cache_data wrappers)
  - independent of the scan engine or UI details
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ImportError:  # pragma: no cover - exercised by lean CI/test environments
    class _StreamlitFallback:
        @staticmethod
        def cache_data(*_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    st = _StreamlitFallback()  # type: ignore[assignment]

try:
    import requests  # type: ignore
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover - requests import failure handled at runtime
    requests = None  # type: ignore
    requests_exc = None  # type: ignore


# ------------------------- Internal config helpers -------------------------


def _get_alpaca_base_urls() -> Dict[str, str]:
    """Return Alpaca config (creds may be None). Resolution lives in
    data.alpaca_config (env-first, guarded secrets) so every reader agrees."""
    from data.alpaca_config import DEFAULT_BASE_URL, DEFAULT_DATA_URL, get_alpaca_config

    cfg = get_alpaca_config()
    if cfg is not None:
        return cfg
    # Preserve this module's historical shape: URLs always present, creds None.
    return {
        "api_key": None,
        "api_secret": None,
        "base_url": DEFAULT_BASE_URL,
        "data_url": DEFAULT_DATA_URL,
    }


def _get_alpaca_headers() -> Optional[Dict[str, str]]:
    """Return Alpaca auth headers if configured, otherwise None."""
    from data.alpaca_config import get_alpaca_headers

    return get_alpaca_headers()


# ------------------------------- Snapshots ---------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def fetch_alpaca_snapshots(symbols: List[str]) -> Dict[str, dict]:
    """
    Fetch snapshot data for a list of symbols from Alpaca.

    Returns a mapping:
      {
        "AAPL": { ...raw snapshot json... },
        "MSFT": { ... },
        ...
      }

    On any configuration or network error, returns an empty dict.
    """
    # Normalize and deduplicate symbols
    if not symbols:
        return {}

    symbols = [s.upper() for s in dict.fromkeys(symbols).keys()]

    headers = _get_alpaca_headers()
    if headers is None or requests is None:
        # Alpaca not configured or requests missing – caller should treat as "no data"
        return {}

    cfg = _get_alpaca_base_urls()
    url = f"{cfg['data_url']}/v2/stocks/snapshots"

    # Chunk the symbol list (~100 per call keeps URLs and responses sane) so
    # callers aren't capped by a single request's practical limit.
    normalized: Dict[str, dict] = {}
    for start in range(0, len(symbols), 100):
        chunk = symbols[start : start + 100]
        try:
            resp = requests.get(
                url, headers=headers, params={"symbols": ",".join(chunk)}, timeout=10
            )
        except requests_exc.RequestException:  # type: ignore[union-attr]
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, dict):
                normalized[k.upper()] = v

    return normalized


# ------------------------------ Public API ---------------------------------


def _sf(value: object) -> Optional[float]:
    """Best-effort float coercion (None on failure)."""
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_avg_daily_volume(symbols: List[str], lookback: int = 20) -> Dict[str, float]:
    """Average daily share volume over `lookback` sessions (for RVOL).

    Cached for 30 minutes because the denominator moves slowly. Returns an empty
    dict when Alpaca isn't configured — callers should treat missing keys as
    "RVOL unavailable" rather than an error.
    """
    if not symbols:
        return {}
    try:
        from data.price_alpaca import download_multi_alpaca
    except Exception:
        return {}
    try:
        frames = download_multi_alpaca(
            [s.upper() for s in symbols],
            period=f"{max(lookback + 5, 25)}d",
            interval="1d",
            prepost=False,
            timeout_s=15.0,
        )
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for sym, frame in (frames or {}).items():
        try:
            vol = frame["Volume"].tail(lookback)
            avg = float(vol.mean())
            if avg > 0:
                out[str(sym).upper()] = avg
        except Exception:
            continue
    return out


def ema_cross_label(frame: Any) -> Optional[str]:
    """Return a short EMA 9/21 cross label from the latest daily close."""
    if frame is None or not hasattr(frame, "columns") or "Close" not in frame.columns:
        return None
    try:
        import pandas as pd

        closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(closes) < 23:
            return None
        ema9 = closes.ewm(span=9, adjust=False).mean()
        ema21 = closes.ewm(span=21, adjust=False).mean()
        prev_delta = float(ema9.iloc[-2] - ema21.iloc[-2])
        curr_delta = float(ema9.iloc[-1] - ema21.iloc[-1])
        if prev_delta <= 0 < curr_delta:
            return "Golden"
        if prev_delta >= 0 > curr_delta:
            return "Death"
    except (ImportError, TypeError, ValueError, KeyError, AttributeError):
        return None
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ema_crosses(symbols: List[str]) -> Dict[str, str]:
    """Detect fresh daily EMA 9/21 crosses for a symbol list.

    Cached separately from live snapshots because this only needs daily bars and
    should not add repeated API pressure to the intraday monitor.
    """
    if not symbols:
        return {}
    try:
        from data.prices import fetch_price_data_parallel

        frames, _skipped = fetch_price_data_parallel(
            [s.upper() for s in symbols],
            period="90d",
            interval="1d",
            max_workers=4,
            chunk_size=25,
            timeout_s=10.0,
            rescue_missing=False,
            use_cache=True,
        )
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for sym, frame in (frames or {}).items():
        label = ema_cross_label(frame)
        if label:
            out[str(sym).upper()] = label
    return out


def build_day_trader_metrics(
    symbols: List[str],
    *,
    with_rvol: bool = True,
) -> List[Dict[str, Any]]:
    """Return per-symbol intraday day-trader metrics from Alpaca snapshots.

    One cached snapshot call yields today's move, gap, VWAP, and volume — no
    heavy bar downloads. RVOL layers on a slowly-changing cached average-volume
    lookup. Each row:
      { ticker, last, chg_pct, gap_pct, vwap, vs_vwap_pct, volume, rvol }
    Symbols with no usable price are dropped. Sorted by |chg_pct| descending.
    """
    if not symbols:
        return []
    syms = [s.upper() for s in dict.fromkeys(symbols).keys() if str(s).strip()]
    snapshots = fetch_alpaca_snapshots(syms)
    if not snapshots:
        return []

    avg_vol = fetch_avg_daily_volume(syms) if with_rvol else {}
    ema_crosses = fetch_ema_crosses(syms)

    rows: List[Dict[str, Optional[float]]] = []
    for sym in syms:
        snap = snapshots.get(sym)
        if not isinstance(snap, dict):
            continue
        latest_trade = snap.get("latestTrade") or {}
        minute_bar = snap.get("minuteBar") or {}
        daily_bar = snap.get("dailyBar") or {}
        prev_daily_bar = snap.get("prevDailyBar") or {}

        last = _sf(latest_trade.get("p")) or _sf(minute_bar.get("c")) or _sf(daily_bar.get("c"))
        if last is None:
            continue
        prev_close = _sf(prev_daily_bar.get("c"))
        today_open = _sf(daily_bar.get("o"))
        vwap = _sf(daily_bar.get("vw"))
        volume = _sf(daily_bar.get("v")) or _sf(minute_bar.get("v"))

        close_today = _sf(daily_bar.get("c"))
        chg_pct = ((last - prev_close) / prev_close * 100.0) if prev_close else None
        gap_pct = (
            (today_open - prev_close) / prev_close * 100.0
            if (today_open and prev_close)
            else None
        )
        vs_vwap_pct = ((last - vwap) / vwap * 100.0) if vwap else None
        avg = avg_vol.get(sym)
        rvol = (volume / avg) if (volume and avg) else None

        rows.append(
            {
                "ticker": sym,
                "last": round(last, 2),
                "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
                "vwap": round(vwap, 2) if vwap is not None else None,
                "vs_vwap_pct": round(vs_vwap_pct, 2) if vs_vwap_pct is not None else None,
                "volume": int(volume) if volume else None,
                "rvol": round(rvol, 2) if rvol is not None else None,
                "close_today": round(close_today, 2) if close_today is not None else None,
                "ema_cross": ema_crosses.get(sym),
            }
        )

    rows.sort(key=lambda r: abs(r["chg_pct"]) if r.get("chg_pct") is not None else -1.0, reverse=True)
    return rows


def get_latest_quotes(
    symbols: List[str],
    session_mode: str = "regular",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Return the latest quotes for a list of symbols using Alpaca snapshots.

    Structure:
      {
        "AAPL": {
          "last": 178.23,
          "prev_close": 176.80,
          "volume": 12345678,
        },
        ...
      }

    Notes:
      - `session_mode` is accepted for future use (e.g., premkt/after-hours tuning)
        but currently we always take Alpaca's latestTrade price, which should
        already reflect extended-hours when available.
      - If Alpaca is not configured or returns no data, this function returns {}.
        Callers may layer their own yfinance fallback on top of this.
    """
    if not symbols:
        return {}

    snapshots = fetch_alpaca_snapshots(symbols)
    if not snapshots:
        return {}

    results: Dict[str, Dict[str, Optional[float]]] = {}

    for raw_symbol in symbols:
        sym = raw_symbol.upper()
        snap = snapshots.get(sym)
        if not isinstance(snap, dict):
            continue

        latest_trade = snap.get("latestTrade") or {}
        minute_bar = snap.get("minuteBar") or {}
        daily_bar = snap.get("dailyBar") or {}
        prev_daily_bar = snap.get("prevDailyBar") or {}

        # Determine "last" price preference:
        # - latestTrade.p (most real-time)
        # - else minuteBar.c
        # - else dailyBar.c
        last: Optional[float] = None
        for candidate in (
            latest_trade.get("p"),
            minute_bar.get("c"),
            daily_bar.get("c"),
        ):
            try:
                if candidate is not None:
                    last = float(candidate)
                    break
            except (TypeError, ValueError):
                continue

        # Previous close from prevDailyBar.c, if available
        prev_close: Optional[float] = None
        try:
            if prev_daily_bar.get("c") is not None:
                prev_close = float(prev_daily_bar["c"])
        except (TypeError, ValueError):
            prev_close = None

        # Volume preference: minuteBar.v, else dailyBar.v
        volume: Optional[float] = None
        for candidate in (minute_bar.get("v"), daily_bar.get("v")):
            try:
                if candidate is not None:
                    volume = float(candidate)
                    break
            except (TypeError, ValueError):
                continue

        if last is None:
            # No usable price – skip this symbol
            continue

        results[sym] = {
            "last": last,
            "prev_close": prev_close,
            "volume": volume,
        }

    return results
