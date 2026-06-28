"""Earnings-calendar fetchers: Financial Modeling Prep (primary) + Finnhub (fallback).

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

# FMP deprecated the v3 legacy calendar; try the newer "stable" endpoint first,
# then fall back to v3 (some keys/plans still serve one but not the other).
FMP_BASES = (
    "https://financialmodelingprep.com/stable/earnings-calendar",
    "https://financialmodelingprep.com/api/v3/earning_calendar",
)
FINNHUB_BASE = "https://finnhub.io/api/v1/calendar/earnings"

EarnMap = Dict[str, Tuple[_dt.date, Optional[str]]]


def _secret(key: str) -> Optional[str]:
    """Read a key env-first, then Streamlit secrets (guarded), else None."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st  # type: ignore

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


def _get_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "hsfinest-earnings"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - trusted host
        return _json.loads(resp.read().decode("utf-8"))


def _merge_earliest(out: EarnMap, sym: str, d: _dt.date, t: Optional[str], today: _dt.date) -> None:
    """Keep the earliest future (>= today) date per symbol."""
    if not sym or d is None or d < today:
        return
    cur = out.get(sym)
    if cur is None or d < cur[0]:
        out[sym] = (d, t)


def fetch_earnings_window_fmp(start_iso: str, end_iso: str) -> EarnMap:
    """Bulk earnings calendar from FMP for [start_iso, end_iso]. {} if no key/error."""
    key = _secret("FMP_API_KEY")
    if not key:
        return {}
    params = urllib.parse.urlencode({"from": start_iso, "to": end_iso, "apikey": key})

    data = None
    for base in FMP_BASES:
        try:
            data = _get_json(f"{base}?{params}")
            if isinstance(data, list) and data:
                print(f"[earnings] FMP endpoint OK: {base}")
                break
        except Exception as e:
            print(f"[earnings] FMP {base} failed: {type(e).__name__}: {e}")
            data = None
    if not isinstance(data, list):
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
            # v3 used "time"; stable may omit it.
            _norm_time(row.get("time")),
            today,
        )
    return out


def fetch_earnings_window_finnhub(start_iso: str, end_iso: str) -> EarnMap:
    """Bulk earnings calendar from Finnhub for [start_iso, end_iso]. {} if no key/error."""
    key = _secret("FINNHUB_API_KEY")
    if not key:
        return {}
    params = urllib.parse.urlencode({"from": start_iso, "to": end_iso, "token": key})
    try:
        data = _get_json(f"{FINNHUB_BASE}?{params}")
    except Exception as e:
        print(f"[earnings] Finnhub fetch failed: {type(e).__name__}: {e}")
        return {}
    rows = (data or {}).get("earningsCalendar") if isinstance(data, dict) else None
    if not isinstance(rows, list):
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


def fetch_earnings_window(start_iso: str, end_iso: str) -> Tuple[EarnMap, str]:
    """Merge FMP (primary) + Finnhub (secondary) for max coverage.

    Both free tiers are limited in different ways (FMP returns few rows; Finnhub
    a broader subset), so we union them: Finnhub provides the base, FMP overlays
    on top (primary precedence on conflicts). Returns (map, source).
    """
    fmp = fetch_earnings_window_fmp(start_iso, end_iso)
    finnhub = fetch_earnings_window_finnhub(start_iso, end_iso)
    if fmp:
        print(f"[earnings] FMP window {start_iso}..{end_iso}: {len(fmp)} symbols")
    if finnhub:
        print(f"[earnings] Finnhub window {start_iso}..{end_iso}: {len(finnhub)} symbols")

    if not fmp and not finnhub:
        return {}, "none"

    merged: EarnMap = dict(finnhub)
    merged.update(fmp)  # FMP wins on overlap (primary)
    source = "fmp+finnhub" if (fmp and finnhub) else ("fmp" if fmp else "finnhub")
    return merged, source
