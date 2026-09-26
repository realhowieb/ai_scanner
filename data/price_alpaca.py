"""Alpaca price-provider helpers for historical OHLCV downloads."""
from __future__ import annotations

import datetime as _dt
import email.utils as _email_utils
import logging
import random
import time
from typing import Dict, Sequence

import pandas as pd

from .price_utils import chunks, normalize_price_frame

try:
    import requests
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    requests_exc = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# Run 51 (maturation hardening) — bounded retry policy for Alpaca data requests.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRY_AFTER_S = 60.0


class AlpacaRateLimitError(RuntimeError):
    """HTTP 429 persisted after bounded retries (throttling, NOT missing data)."""

    rate_limited = True


class AlpacaRequestStats:
    """Per-run request instrumentation shared across Alpaca calls."""

    def __init__(self) -> None:
        self.requests = 0        # HTTP attempts actually sent (incl. retries)
        self.rate_limited = 0    # HTTP 429 responses received
        self.retries = 0         # attempts re-sent after a retryable failure
        self.last_request_at = 0.0

    def as_dict(self) -> Dict[str, int]:
        return {"alpaca_requests": self.requests,
                "alpaca_429_count": self.rate_limited,
                "alpaca_retry_count": self.retries}


def _retry_after_seconds(resp) -> float | None:
    """Server-requested wait: Retry-After (seconds or HTTP-date), else Alpaca's
    X-RateLimit-Reset (unix epoch seconds). None when absent/unparseable."""
    headers = getattr(resp, "headers", None) or {}
    try:
        ra = headers.get("Retry-After")
    except Exception:
        ra = None
    if ra:
        try:
            return max(0.0, float(ra))
        except (TypeError, ValueError):
            try:
                when = _email_utils.parsedate_to_datetime(str(ra))
                return max(0.0, (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds())
            except Exception:
                pass
    try:
        reset = headers.get("X-RateLimit-Reset")
        if reset:
            return max(0.0, float(reset) - time.time())
    except (TypeError, ValueError):
        pass
    return None


def _backoff_delay(attempt: int, base_s: float, cap_s: float) -> float:
    """Exponential backoff with jitter: uniform in [d/2, d], d = min(cap, base*2^n)."""
    d = min(cap_s, base_s * (2 ** attempt))
    return d / 2.0 + random.uniform(0.0, d / 2.0)


def _alpaca_get(url: str, *, headers: Dict[str, str], params: Dict, timeout_s: float,
                stats: AlpacaRequestStats | None = None, max_retries: int = 5,
                base_delay_s: float = 1.0, max_delay_s: float = 30.0,
                min_interval_s: float = 0.0, sleep=None):
    """GET + JSON with bounded retries on 429 / 5xx / timeouts / connection errors.

    429 honours Retry-After (capped) plus jitter, otherwise exponential backoff with
    jitter. Raises AlpacaRateLimitError if 429 persists past `max_retries`; other
    non-retryable failures raise as before. `min_interval_s` paces consecutive
    requests sharing `stats` so a run stays under the provider's per-minute quota.
    """
    sleep = sleep or time.sleep
    retryable_exc = ()
    if requests_exc is not None:
        retryable_exc = (requests_exc.Timeout, requests_exc.ConnectionError)
    attempt = 0
    while True:
        if stats is not None and min_interval_s > 0 and stats.last_request_at:
            wait = stats.last_request_at + min_interval_s - time.monotonic()
            if wait > 0:
                sleep(wait)
        if stats is not None:
            stats.requests += 1
            stats.last_request_at = time.monotonic()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)  # type: ignore[union-attr]
        except retryable_exc:
            if attempt >= max_retries:
                raise
            if stats is not None:
                stats.retries += 1
            sleep(_backoff_delay(attempt, base_delay_s, max_delay_s))
            attempt += 1
            continue
        status = getattr(resp, "status_code", 200)
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            if status == 429 and stats is not None:
                stats.rate_limited += 1
            if attempt >= max_retries:
                if status == 429:
                    raise AlpacaRateLimitError(
                        f"Alpaca HTTP 429 persisted after {max_retries} retries")
                resp.raise_for_status()
                raise RuntimeError(f"Alpaca HTTP {status} persisted after {max_retries} retries")
            delay =_retry_after_seconds(resp) if status == 429 else None
            if delay is not None:
                delay = min(delay, _MAX_RETRY_AFTER_S) + random.uniform(0.0, 0.25 * base_delay_s)
            else:
                delay = _backoff_delay(attempt, base_delay_s, max_delay_s)
            if stats is not None:
                stats.retries += 1
            sleep(delay)
            attempt += 1
            continue
        resp.raise_for_status()
        return resp.json()


def get_alpaca_config() -> Dict[str, str] | None:
    """Return Alpaca Market Data configuration if it is available.

    Resolution (env-first, then guarded Streamlit secrets) is centralized in
    data.alpaca_config so every reader agrees; this wrapper only adds the
    requests-availability check that bar downloads need.
    """
    from data.alpaca_config import get_alpaca_config as _shared_config

    cfg = _shared_config()
    if cfg is None or requests is None:
        return None
    return cfg


def get_alpaca_data_feed() -> str:
    from data.alpaca_config import get_alpaca_data_feed as _shared_feed

    return _shared_feed()


def alpaca_timeframe_from_interval(interval: str) -> str | None:
    """Map yfinance-style intervals to Alpaca timeframes."""
    return {
        "1d": "1Day",
        "1D": "1Day",
    }.get(interval)


def alpaca_limit_from_period(period: str) -> int:
    """Convert a yfinance-style period string into an approximate bar limit."""
    default = 60
    try:
        normalized = period.lower()
        if normalized.endswith("d"):
            return max(1, int(normalized[:-1]))
        if normalized.endswith("mo"):
            return max(1, int(normalized[:-2]) * 21)
        if normalized.endswith("y"):
            return max(1, int(normalized[:-1]) * 252)
    except (AttributeError, TypeError, ValueError):
        pass
    return default


def alpaca_start_from_period(period: str) -> str:
    """Return an ISO date (UTC) far enough back to cover `period` trading days.

    Alpaca's /v2/stocks/bars endpoint requires a `start` date; without it the
    request 400s or returns no bars. We use ~1.7x calendar days per requested
    trading day (plus a buffer) so weekends/holidays don't starve the window.
    """
    days = 60
    try:
        n = (period or "").lower()
        if n.endswith("d"):
            days = max(1, int(n[:-1]))
        elif n.endswith("mo"):
            days = max(1, int(n[:-2])) * 31
        elif n.endswith("y"):
            days = max(1, int(n[:-1])) * 366
    except (AttributeError, TypeError, ValueError):
        pass
    start = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=int(days * 1.7) + 5)
    return start.strftime("%Y-%m-%d")


def download_multi_alpaca(
    tickers: Sequence[str],
    period: str,
    interval: str,
    prepost: bool,
    timeout_s: float,
    feed: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Download bars for multiple symbols from Alpaca Market Data."""
    del prepost  # Daily Alpaca bars ignore extended-hours selection.

    cfg = get_alpaca_config()
    if cfg is None:
        raise RuntimeError("Alpaca configuration is not available.")

    timeframe = alpaca_timeframe_from_interval(interval)
    if timeframe is None:
        raise RuntimeError(f"Unsupported interval for Alpaca bars: {interval!r}")

    symbols = [symbol for symbol in dict.fromkeys(tickers) if isinstance(symbol, str) and symbol.strip()]
    if not symbols:
        return {}

    # Alpaca uses a dot for class shares (BRK.B), while our universe uses
    # Yahoo's dash form (BRK-B). Send the dot form and map responses back so a
    # single class-share symbol doesn't 400 the whole batch.
    orig_by_alpaca: Dict[str, str] = {}
    alpaca_symbols: list[str] = []
    for sym in symbols:
        a = str(sym).upper().replace("-", ".")
        orig_by_alpaca[a] = str(sym).upper()
        alpaca_symbols.append(a)
    symbols = alpaca_symbols

    url = f"{cfg['data_url']}/v2/stocks/bars"
    data_feed = str(feed or get_alpaca_data_feed()).strip().lower()
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }

    start = alpaca_start_from_period(period)
    out: Dict[str, pd.DataFrame] = {}
    for chunk in chunks(symbols, 150):
        symbols_param = ",".join(sorted(chunk))
        # Accumulate bars across pages (a 150-symbol × ~60-bar response can
        # exceed Alpaca's 1000-bar page limit, so we must follow next_page_token).
        bars_by_symbol: Dict[str, list] = {}
        page_token: str | None = None
        failed = False
        for _ in range(50):  # hard page cap as a safety bound
            params = {
                "symbols": symbols_param,
                "timeframe": timeframe,
                "start": start,
                "limit": 10000,
                "adjustment": "raw",
                "feed": data_feed,
            }
            if page_token:
                params["page_token"] = page_token

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)  # type: ignore[union-attr]
                resp.raise_for_status()
                payload = resp.json()
            except requests_exc.RequestException as exc:  # type: ignore[union-attr]
                logger.warning("Alpaca request failed for %s symbols (%s): %s", len(chunk), symbols_param, exc)
                failed = True
                break
            except ValueError as exc:
                logger.warning("Alpaca returned invalid JSON for %s symbols (%s): %s", len(chunk), symbols_param, exc)
                failed = True
                break

            if not isinstance(payload, dict):
                logger.warning("Alpaca returned non-dict payload for %s symbols", len(chunk))
                failed = True
                break

            page_bars = payload.get("bars") or {}
            for sym, bars in page_bars.items():
                if bars:
                    bars_by_symbol.setdefault(sym, []).extend(bars)

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if failed:
            continue
        if not bars_by_symbol:
            logger.info("Alpaca returned no bars for %s symbols (%s)", len(chunk), symbols_param)
            continue

        for symbol, bars in (bars_by_symbol or {}).items():
            if not bars:
                logger.debug("Alpaca returned empty bars for symbol %s", symbol)
                continue
            df = pd.DataFrame(bars)
            if df.empty:
                logger.debug("Alpaca bars converted to empty DataFrame for symbol %s", symbol)
                continue

            if "t" in df.columns:
                df["t"] = pd.to_datetime(df["t"], errors="coerce")
                df = df.set_index("t")

            rename = {
                old: new
                for old, new in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("v", "Volume"))
                if old in df.columns
            }
            if rename:
                df = df.rename(columns=rename)

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            # Map Alpaca's dot form back to the caller's original (dash) symbol.
            out_sym = orig_by_alpaca.get(str(symbol).upper(), str(symbol).upper())

            normalized = normalize_price_frame(df)
            try:
                normalized.attrs["source"] = "alpaca_multi"
                normalized.attrs["feed"] = data_feed
                normalized.attrs["symbol"] = out_sym
            except (AttributeError, TypeError, ValueError):
                pass
            out[out_sym] = normalized

    return out


def fetch_minute_bars(
    symbol: str,
    start: str,
    end: str | None = None,
    *,
    timeframe: str = "1Min",
    feed: str | None = None,
    timeout_s: float = 15.0,
    max_pages: int = 50,
) -> list[dict]:
    """Fetch historical intraday bars for ONE symbol from Alpaca Market Data.

    Returns ascending bars [{"t","o","h","l","c","v"}, ...] (RFC3339 `t`). Used by
    the off-Streamlit DT Score validation harness — NOT the production Day Trader
    path. `start`/`end` are RFC3339 / ISO date strings. Respects the existing
    IEX/SIP feed selection. Fails safe: returns [] with no config / on any error
    (never raises into a caller).
    """
    if requests is None:
        return []
    cfg = get_alpaca_config()
    if cfg is None:
        return []
    sym = str(symbol or "").upper().replace("-", ".")
    if not sym:
        return []
    url = f"{cfg['data_url']}/v2/stocks/{sym}/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }
    data_feed = str(feed or get_alpaca_data_feed()).strip().lower()
    out: list[dict] = []
    page_token: str | None = None
    for _ in range(max_pages):
        params = {
            "timeframe": timeframe, "start": start, "limit": 10000,
            "adjustment": "raw", "feed": data_feed,
        }
        if end:
            params["end"] = end
        if page_token:
            params["page_token"] = page_token
        try:
            payload = _alpaca_get(url, headers=headers, params=params, timeout_s=timeout_s)
        except Exception as exc:  # network / parse / persistent 429 — fail safe
            logger.warning("Alpaca minute bars failed for %s: %s", sym, exc)
            break
        for b in (payload.get("bars") or []):
            if isinstance(b, dict) and b.get("t") is not None:
                out.append({k: b.get(k) for k in ("t", "o", "h", "l", "c", "v")})
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return out


def fetch_minute_bars_multi(
    symbols: Sequence[str],
    start: str,
    end: str | None = None,
    *,
    timeframe: str = "1Min",
    feed: str | None = None,
    timeout_s: float = 30.0,
    max_pages: int = 200,
    stats: AlpacaRequestStats | None = None,
    max_retries: int = 5,
    min_interval_s: float = 0.35,
    sleep=None,
) -> Dict[str, list[dict]]:
    """Fetch intraday bars for MANY symbols in one paginated request series.

    Uses the multi-symbol /v2/stocks/bars endpoint (10k bars per page shared across
    symbols) instead of one request per symbol. Returns {caller_symbol: ascending
    bars [{"t","o","h","l","c","v"}]}; a symbol with no bars is simply absent.

    Unlike `fetch_minute_bars` this does NOT fail safe: it raises so the caller can
    tell throttling (AlpacaRateLimitError) from provider errors, and never returns
    a partially-paginated result (a mid-series failure raises).
    """
    if requests is None:
        raise RuntimeError("requests is not installed")
    cfg = get_alpaca_config()
    if cfg is None:
        raise RuntimeError("Alpaca configuration is not available.")
    orig_by_alpaca: Dict[str, str] = {}
    for sym in symbols:
        s = str(sym or "").strip().upper()
        if s:
            orig_by_alpaca.setdefault(s.replace("-", "."), s)
    if not orig_by_alpaca:
        return {}
    url = f"{cfg['data_url']}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }
    data_feed = str(feed or get_alpaca_data_feed()).strip().lower()
    out: Dict[str, list[dict]] = {}
    page_token: str | None = None
    for _ in range(max_pages):
        params = {
            "symbols": ",".join(sorted(orig_by_alpaca)), "timeframe": timeframe,
            "start": start, "limit": 10000, "adjustment": "raw", "feed": data_feed,
        }
        if end:
            params["end"] = end
        if page_token:
            params["page_token"] = page_token
        payload = _alpaca_get(url, headers=headers, params=params, timeout_s=timeout_s,
                              stats=stats, max_retries=max_retries,
                              min_interval_s=min_interval_s, sleep=sleep)
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca returned a non-dict payload")
        for sym, bars in (payload.get("bars") or {}).items():
            key = orig_by_alpaca.get(str(sym).upper(), str(sym).upper())
            for b in bars or []:
                if isinstance(b, dict) and b.get("t") is not None:
                    out.setdefault(key, []).append({k: b.get(k) for k in ("t", "o", "h", "l", "c", "v")})
        page_token = payload.get("next_page_token")
        if not page_token:
            return out
    raise RuntimeError(f"Alpaca pagination exceeded {max_pages} pages")
