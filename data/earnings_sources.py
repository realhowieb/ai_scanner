"""Earnings-calendar fetchers: Financial Modeling Prep + Finnhub + Nasdaq (bulk), and
Yahoo (one symbol at a time, for names the bulk calendars miss).

Both providers expose a bulk date-range endpoint, so one request covers the whole
universe in a window — far cheaper than per-symbol calls (FMP free tier is ~250
calls/day). Returns {SYMBOL: (date, time)} where time is 'AMC' | 'BMO' | None.

No-ops cleanly (returns {}) when the relevant API key isn't configured, so the
caller falls back to the next source. Keys are read env-first with a guarded
Streamlit-secrets fallback, so this works headlessly (cron) and in the app.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

# FMP retired the v3 calendar (403 on every call since late Oct 2026), so only the
# "stable" endpoint is used. On the free plan it covers ~87 sample symbols only.
FMP_BASES = ("https://financialmodelingprep.com/stable/earnings-calendar",)
FINNHUB_BASE = "https://finnhub.io/api/v1/calendar/earnings"
# Nasdaq's public calendar, one trading day per request, no key. Finnhub's free
# calendar skips most foreign and cross-listed filers (TD, RY, NVO, RACE, ENB...);
# Nasdaq lists the US-listed ones with their before-open / after-close timing.
NASDAQ_BASE = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_MAX_DAYS = 45          # Nasdaq rarely lists dates further out than this
NASDAQ_MAX_ERRORS = 3         # stop after this many failed days in a row
NASDAQ_HEADERS = {
    # Nasdaq refuses requests without a browser-like User-Agent.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                  "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "application/json, text/plain, */*",
}

EarnMap = Dict[str, Tuple[_dt.date, Optional[str]]]

# Both providers cap how many rows one request returns, and Finnhub drops the
# nearest dates first when a range is too big: one 120-day request came back
# with ~1,500 rows and only 6 earnings in the coming week (no JPM, WFC, BAC...).
# So the window is fetched in slices, and a Finnhub slice that comes back near
# the cap is split again.
FMP_SLICE_DAYS = 30
FINNHUB_SLICE_DAYS = 7
FINNHUB_SPLIT_ROWS = 1000


def _secret(key: str) -> Optional[str]:
    """Read a key env-first, then Streamlit secrets (guarded), else None."""
    val = os.getenv(key)
    if val:
        return val
    try:
        from utils.st_compat import st

        v = getattr(st, "secrets", {}).get(key)
        return str(v) if v else None
    except Exception:
        return None


def _norm_symbol(sym: object) -> str:
    return str(sym or "").strip().upper().replace(".", "-")


def _norm_time(raw: object) -> Optional[str]:
    s = str(raw or "").strip().lower()
    if s in ("amc", "after market close", "aftermarket"):
        return "AMC"
    if s in ("bmo", "before market open", "premarket"):
        return "BMO"
    return None


def _parse_date(raw: object) -> Optional[_dt.date]:
    s = str(raw or "").strip()[:10]
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s)
    except ValueError:
        return None


def _get_json(url: str, timeout: float = 20.0, headers: Optional[dict] = None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "hsfinest-earnings"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - trusted host
        return _json.loads(resp.read().decode("utf-8"))


def _merge_earliest(out: EarnMap, sym: str, d: _dt.date, t: Optional[str], today: _dt.date) -> None:
    """Keep the earliest future (>= today) date per symbol."""
    if not sym or d is None or d < today:
        return
    cur = out.get(sym)
    if cur is None or d < cur[0]:
        out[sym] = (d, t)


def _slices(start_iso: str, end_iso: str, days: int):
    """[start, end] as consecutive (from, to) ISO pairs of at most `days` days each."""
    start, end = _parse_date(start_iso), _parse_date(end_iso)
    if start is None or end is None:
        return [(start_iso, end_iso)]
    out = []
    while start <= end:
        stop = min(end, start + _dt.timedelta(days=max(1, days) - 1))
        out.append((start.isoformat(), stop.isoformat()))
        start = stop + _dt.timedelta(days=1)
    return out


def _fetch_fmp_rows(key: str, start_iso: str, end_iso: str) -> list:
    # The stable endpoint only returns "time" (bmo/amc) with includeReportTimes;
    # v3 ignores the extra parameter.
    params = urllib.parse.urlencode({"from": start_iso, "to": end_iso, "apikey": key,
                                     "includeReportTimes": "true"})
    for base in FMP_BASES:
        try:
            data = _get_json(f"{base}?{params}")
            if isinstance(data, list) and data:
                return data
        except Exception as e:
            print(f"[earnings] FMP {start_iso}..{end_iso} failed: {type(e).__name__}: {e}")
    return []


def fetch_earnings_window_fmp(start_iso: str, end_iso: str) -> EarnMap:
    """Bulk earnings calendar from FMP for [start_iso, end_iso]. {} if no key/error."""
    key = _secret("FMP_API_KEY")
    if not key:
        return {}

    data: list = []
    for lo, hi in _slices(start_iso, end_iso, FMP_SLICE_DAYS):
        data.extend(_fetch_fmp_rows(key, lo, hi))
    if not data:
        return {}

    out: EarnMap = {}
    today = _dt.date.today()
    for row in data:
        if not isinstance(row, dict):
            continue
        _merge_earliest(
            out,
            _norm_symbol(row.get("symbol")),
            _parse_date(row.get("date")),
            # v3 and stable (with includeReportTimes) both use "time".
            _norm_time(row.get("time")),
            today,
        )
    return out


def _fetch_finnhub_rows(key: str, start_iso: str, end_iso: str) -> list:
    """One Finnhub request; a range that comes back near the row cap is split in half
    and fetched again, so a busy week is not silently truncated."""
    params = urllib.parse.urlencode({"from": start_iso, "to": end_iso, "token": key})
    try:
        data = _get_json(f"{FINNHUB_BASE}?{params}")
    except Exception as e:
        print(f"[earnings] Finnhub {start_iso}..{end_iso} failed: {type(e).__name__}: {e}")
        return []
    rows = (data or {}).get("earningsCalendar") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    start, end = _parse_date(start_iso), _parse_date(end_iso)
    if len(rows) >= FINNHUB_SPLIT_ROWS and start and end and end > start:
        mid = start + (end - start) // 2
        return (_fetch_finnhub_rows(key, start.isoformat(), mid.isoformat())
                + _fetch_finnhub_rows(key, (mid + _dt.timedelta(days=1)).isoformat(), end.isoformat()))
    return rows


def fetch_earnings_window_finnhub(start_iso: str, end_iso: str) -> EarnMap:
    """Bulk earnings calendar from Finnhub for [start_iso, end_iso]. {} if no key/error."""
    key = _secret("FINNHUB_API_KEY")
    if not key:
        return {}
    rows: list = []
    for lo, hi in _slices(start_iso, end_iso, FINNHUB_SLICE_DAYS):
        rows.extend(_fetch_finnhub_rows(key, lo, hi))
    if not rows:
        return {}
    out: EarnMap = {}
    today = _dt.date.today()
    for row in rows:
        if not isinstance(row, dict):
            continue
        _merge_earliest(
            out,
            _norm_symbol(row.get("symbol")),
            _parse_date(row.get("date")),
            _norm_time(row.get("hour")),
            today,
        )
    return out


def _nasdaq_time(raw: object) -> Optional[str]:
    s = str(raw or "").strip().lower()
    if "pre-market" in s:
        return "BMO"
    if "after-hours" in s:
        return "AMC"
    return None


def nasdaq_enabled() -> bool:
    return os.getenv("HSF_NASDAQ_EARNINGS", "1").strip().lower() not in ("0", "false", "no", "off")


def fetch_earnings_window_nasdaq(start_iso: str, end_iso: str, max_days: int = NASDAQ_MAX_DAYS) -> EarnMap:
    """Nasdaq's earnings calendar for each weekday in [start, min(end, start+max_days)].
    {} when turned off (HSF_NASDAQ_EARNINGS=0) or unreachable."""
    if not nasdaq_enabled():
        return {}
    start, end = _parse_date(start_iso), _parse_date(end_iso)
    if start is None or end is None:
        return {}
    end = min(end, start + _dt.timedelta(days=max(0, max_days)))
    out: EarnMap = {}
    today = _dt.date.today()
    errors = 0
    day = start
    while day <= end and errors < NASDAQ_MAX_ERRORS:
        if day.weekday() < 5:
            try:
                data = _get_json(f"{NASDAQ_BASE}?date={day.isoformat()}", headers=NASDAQ_HEADERS)
                rows = ((data or {}).get("data") or {}).get("rows") if isinstance(data, dict) else None
                errors = 0
                for row in rows or []:
                    if isinstance(row, dict):
                        _merge_earliest(out, _norm_symbol(row.get("symbol")), day,
                                        _nasdaq_time(row.get("time")), today)
            except Exception as e:
                errors += 1
                print(f"[earnings] Nasdaq {day} failed: {type(e).__name__}: {e}")
        day += _dt.timedelta(days=1)
    return out


def fetch_next_earnings_yahoo(symbol: str) -> Optional[_dt.date]:
    """Next earnings date from Yahoo (yfinance Ticker.calendar), or None.

    Yahoo keeps showing the last report until the next one is announced, so only
    dates today or later count."""
    try:
        import yfinance as yf

        cal = yf.Ticker(str(symbol).strip().upper()).calendar
    except Exception as e:
        print(f"[earnings] Yahoo {symbol} failed: {type(e).__name__}")
        return None
    raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if not isinstance(raw, (list, tuple)):
        raw = [raw] if raw else []
    today = _dt.date.today()
    future = []
    for v in raw:
        d = v.date() if isinstance(v, _dt.datetime) else v
        if isinstance(d, _dt.date) and d >= today:
            future.append(d)
    return min(future) if future else None


def fetch_earnings_window(start_iso: str, end_iso: str, *, nasdaq: bool = False) -> Tuple[EarnMap, str]:
    """Merge FMP (primary) + Finnhub (secondary) for max coverage.

    Both free tiers are limited in different ways (FMP returns few rows; Finnhub
    a broader subset), so we union them: Finnhub provides the base, FMP overlays
    on top (primary precedence on conflicts). With nasdaq=True (the daily cron),
    Nasdaq fills symbols Finnhub lacks, and a missing Finnhub time on the same date.
    Returns (map, source).
    """
    fmp = fetch_earnings_window_fmp(start_iso, end_iso)
    finnhub = fetch_earnings_window_finnhub(start_iso, end_iso)
    nas = fetch_earnings_window_nasdaq(start_iso, end_iso) if nasdaq else {}
    if fmp:
        print(f"[earnings] FMP window {start_iso}..{end_iso}: {len(fmp)} symbols")
    if finnhub:
        print(f"[earnings] Finnhub window {start_iso}..{end_iso}: {len(finnhub)} symbols")
    if nas:
        added = sum(1 for k in nas if k not in finnhub)
        print(f"[earnings] Nasdaq window {start_iso}..{end_iso}: {len(nas)} symbols, {added} not in Finnhub")

    if not fmp and not finnhub and not nas:
        return {}, "none"

    merged: EarnMap = dict(finnhub)
    for sym, (d, t) in nas.items():
        cur = merged.get(sym)
        if cur is None:
            merged[sym] = (d, t)
        elif cur[1] is None and cur[0] == d and t is not None:
            merged[sym] = (d, t)
    for sym, (d, t) in fmp.items():
        # FMP wins on overlap (primary), but a missing FMP time must not hide
        # Finnhub's for the same date: JPM/WFC/C/UNH/JNJ showed "After close"
        # because FMP's None kept a stale stored time (both providers say BMO).
        other = finnhub.get(sym)
        if t is None and other is not None and other[0] == d:
            t = other[1]
        merged[sym] = (d, t)
    source = "+".join(n for n, m in (("fmp", fmp), ("finnhub", finnhub), ("nasdaq", nas)) if m)
    return merged, source
