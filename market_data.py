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


def _get_alpaca_data_feed() -> str:
    from data.alpaca_config import get_alpaca_data_feed

    return get_alpaca_data_feed()


def _alpaca_source(feed: str) -> str:
    return f"alpaca_{str(feed or '').strip().lower() or 'unknown'}"


# ------------------------------- Snapshots ---------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def fetch_alpaca_snapshots(symbols: List[str], feed: Optional[str] = None) -> Dict[str, dict]:
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
    feed = str(feed or _get_alpaca_data_feed()).strip().lower()

    # Alpaca uses a dot for class shares (BRK.B) while our universe uses Yahoo's
    # dash form (BRK-B). Send the dot form and map responses back — otherwise a
    # single class-share symbol 400s the whole request and every ticker in that
    # chunk silently returns no quote.
    alpaca_syms: List[str] = []
    orig_by_alpaca: Dict[str, str] = {}
    for s in symbols:
        a = s.replace("-", ".")
        alpaca_syms.append(a)
        orig_by_alpaca[a] = s

    # Chunk the symbol list (~100 per call keeps URLs and responses sane) so
    # callers aren't capped by a single request's practical limit.
    normalized: Dict[str, dict] = {}
    for start in range(0, len(alpaca_syms), 100):
        chunk = alpaca_syms[start : start + 100]
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"symbols": ",".join(chunk), "feed": feed},
                timeout=10,
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
                key = orig_by_alpaca.get(k.upper(), k.upper())  # map BRK.B -> BRK-B
                normalized[key] = v

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


def calculate_day_trader_row_audit(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recalculate Day Trader display math from row primitives.

    This is a developer/debug helper. It does not change ranking or rendering;
    it exists so tests and diagnostics can prove displayed values match their
    claimed formulas.
    """
    ticker = str(row.get("ticker") or row.get("Ticker") or "").upper()
    open_px = _sf(row.get("open") if "open" in row else row.get("Open"))
    prev_close = _sf(
        row.get("previous_close")
        if "previous_close" in row
        else row.get("Previous Close")
    )
    last = _sf(row.get("last") if "last" in row else row.get("Last"))
    displayed_gap = _sf(row.get("gap_pct") if "gap_pct" in row else row.get("Gap %"))
    displayed_change = _sf(
        row.get("change_dollar")
        if "change_dollar" in row
        else row.get("Change $")
    )
    vwap = _sf(row.get("vwap") if "vwap" in row else row.get("VWAP"))

    expected_gap = (
        round((open_px - prev_close) / prev_close * 100.0, 2)
        if open_px is not None and prev_close
        else None
    )
    expected_change = (
        round(last - open_px, 2)
        if last is not None and open_px is not None
        else None
    )
    expected_vs_vwap_pct = (
        round((last - vwap) / vwap * 100.0, 2)
        if last is not None and vwap
        else None
    )
    if last is None or vwap is None:
        expected_vwap_state = None
    elif last > vwap:
        expected_vwap_state = "above"
    elif last < vwap:
        expected_vwap_state = "below"
    else:
        expected_vwap_state = "at VWAP"

    gap_difference = (
        round(abs(float(displayed_gap) - float(expected_gap)), 4)
        if displayed_gap is not None and expected_gap is not None
        else None
    )
    change_difference = (
        round(abs(float(displayed_change) - float(expected_change)), 4)
        if displayed_change is not None and expected_change is not None
        else None
    )

    return {
        "ticker": ticker,
        "open": open_px,
        "previous_close": prev_close,
        "last": last,
        "displayed_gap_pct": displayed_gap,
        "expected_gap_pct": expected_gap,
        "gap_difference": gap_difference,
        "gap_pass": bool(gap_difference is not None and gap_difference <= 0.02),
        "displayed_change_dollar": displayed_change,
        "expected_change_dollar": expected_change,
        "change_difference": change_difference,
        "change_pass": bool(change_difference is not None and change_difference <= 0.01),
        "vwap": vwap,
        "expected_vs_vwap_pct": expected_vs_vwap_pct,
        "expected_vwap_state": expected_vwap_state,
        "open_source": row.get("open_source"),
        "previous_close_source": row.get("previous_close_source")
        or row.get("prev_close_source"),
        "vwap_source": row.get("vwap_source"),
        "volume_source": row.get("volume_source"),
        "rvol_source": row.get("rvol_source"),
    }


def build_day_trader_validation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return formula-validation rows for developer diagnostics/tests."""
    return [calculate_day_trader_row_audit(row) for row in rows or []]


def day_trader_metric_timeframes() -> List[Dict[str, str]]:
    """Document the current Day Trader metric timeframe/source semantics."""
    return [
        {"metric": "Open", "timeframe": "snapshot daily bar", "source": "Alpaca dailyBar.o"},
        {"metric": "Last", "timeframe": "latest snapshot", "source": "Alpaca latestTrade/minuteBar/dailyBar"},
        {"metric": "Change $", "timeframe": "session", "source": "Last - Open"},
        {"metric": "Gap %", "timeframe": "daily/session", "source": "Open vs prevDailyBar.c"},
        {"metric": "ADX", "timeframe": "1d", "source": "daily OHLC bars, ADX(14)"},
        {"metric": "VWAP", "timeframe": "snapshot daily bar", "source": "Alpaca dailyBar.vw"},
        {"metric": "vs VWAP", "timeframe": "latest vs daily VWAP", "source": "Last relative to VWAP"},
        {"metric": "RVOL", "timeframe": "session vs 20 daily bars", "source": "current volume / 20d avg volume"},
        {"metric": "Volume", "timeframe": "snapshot daily bar cumulative", "source": "Alpaca dailyBar.v"},
        {"metric": "SuperTrend", "timeframe": "1d", "source": "daily OHLC bars, SuperTrend(13,2)"},
        {"metric": "EWO", "timeframe": "1d", "source": "daily close bars, SMA(5)-SMA(35)"},
    ]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_avg_daily_volume(
    symbols: List[str],
    lookback: int = 20,
    feed: Optional[str] = None,
) -> Dict[str, float]:
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
            feed=feed or _get_alpaca_data_feed(),
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
    """Return a short EMA 9/21 cross label ('Golden'/'Death') from daily closes."""
    from scan.indicators import ema_cross_detail

    detail = ema_cross_detail(frame)
    if not detail:
        return None
    return "Golden" if detail["direction"] == "bullish" else "Death"


# Daily-bar enrichments run only for a display-sized symbol set (the Day Trader
# table caps at 150). Anything larger is a screening/universe call that must not
# trigger a universe-wide daily-bar fetch.
_DAILY_ENRICH_MAX = 200


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ema_crosses(symbols: List[str]) -> Dict[str, str]:
    """Detect fresh daily EMA 9/21 crosses for a symbol list.

    Cached separately from live snapshots because this only needs daily bars and
    should not add repeated API pressure to the intraday monitor.
    """
    # Never fetch daily bars for more than a display-sized set — a hard stop
    # against any path accidentally requesting the whole universe.
    if not symbols or len(symbols) > _DAILY_ENRICH_MAX:
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


def _range_metrics(frame) -> Optional[Dict[str, Any]]:
    """Daily technical enrichments for the display-sized Day Trader table."""
    try:
        import pandas as pd

        from scan.indicators import adx, atr, bollinger, donchian, ewo, ewo_pct, supertrend

        if frame is None or "Close" not in getattr(frame, "columns", []):
            return None
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(close) < 25:
            return None
        last = float(close.iloc[-1])
        if last <= 0:
            return None
        indicator_source = str(getattr(frame, "attrs", {}).get("source") or "daily_bars")
        indicator_feed = getattr(frame, "attrs", {}).get("feed")

        try:
            atr_pct = float(atr(frame, 14).iloc[-1]) / last * 100.0
        except Exception:
            atr_pct = None

        # Donchian: position within the 20-day range + fresh-breakout flag
        # (today's high/low vs the *prior* 20-day channel).
        up, lo = donchian(frame, 20)
        u, low_b = float(up.iloc[-1]), float(lo.iloc[-1])
        donch_pos = ((last - low_b) / (u - low_b) * 100.0) if u > low_b else None
        prior_up = float(up.iloc[-2]) if len(up) >= 2 and up.iloc[-2] == up.iloc[-2] else u
        prior_lo = float(lo.iloc[-2]) if len(lo) >= 2 and lo.iloc[-2] == lo.iloc[-2] else low_b
        hi_t = float(pd.to_numeric(frame["High"], errors="coerce").iloc[-1])
        lo_t = float(pd.to_numeric(frame["Low"], errors="coerce").iloc[-1])
        breakout = "up" if hi_t >= prior_up else "down" if lo_t <= prior_lo else None

        # Bollinger %B + squeeze (band-width in its own recent low quantile).
        mid, bu, bl = bollinger(close, 20, 2.0)
        bu_l, bl_l = float(bu.iloc[-1]), float(bl.iloc[-1])
        pctb = ((last - bl_l) / (bu_l - bl_l) * 100.0) if bu_l > bl_l else None
        width = ((bu - bl) / mid * 100.0).dropna()
        squeeze = bool(len(width) >= 20 and float(width.iloc[-1]) <= float(width.tail(100).quantile(0.15)))

        def _finite_latest(series) -> Optional[float]:
            try:
                val = float(series.iloc[-1])
                return val if val == val else None
            except Exception:
                return None

        try:
            adx_val = _finite_latest(adx(frame, 14))
        except Exception:
            adx_val = None

        try:
            st_frame = supertrend(frame, 13, 2.0)
            st_val = _finite_latest(st_frame["supertrend"])
            st_dir = st_frame["direction"].iloc[-1]
            st_dir = str(st_dir).lower() if pd.notna(st_dir) else None
            if st_dir not in ("green", "red"):
                st_dir = None
        except Exception:
            st_val = None
            st_dir = None

        try:
            ewo_val = _finite_latest(ewo(frame, 5, 35))
        except Exception:
            ewo_val = None
        try:
            ewo_pct_val = _finite_latest(ewo_pct(frame, 5, 35))
        except Exception:
            ewo_pct_val = None

        return {
            "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
            "donchian_pos": round(donch_pos) if donch_pos is not None else None,
            "donchian_breakout": breakout,
            "bb_pctb": round(pctb) if pctb is not None else None,
            "bb_squeeze": squeeze,
            "adx": round(adx_val, 1) if adx_val is not None else None,
            "adx_period": 14,
            "adx_timeframe": "1d",
            "adx_source": indicator_source,
            "adx_feed": indicator_feed,
            "supertrend": round(st_val, 2) if st_val is not None else None,
            "supertrend_direction": st_dir,
            "supertrend_period": 13,
            "supertrend_multiplier": 2.0,
            "supertrend_timeframe": "1d",
            "supertrend_source": indicator_source,
            "supertrend_feed": indicator_feed,
            "ewo": round(ewo_val, 2) if ewo_val is not None else None,
            "ewo_pct": round(ewo_pct_val, 2) if ewo_pct_val is not None else None,
            "ewo_fast": 5,
            "ewo_slow": 35,
            "ewo_timeframe": "1d",
            "ewo_source": indicator_source,
            "ewo_feed": indicator_feed,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_daily_range_metrics(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """ATR/Donchian/Bollinger per symbol from daily bars. Cached like EMA crosses."""
    if not symbols or len(symbols) > _DAILY_ENRICH_MAX:
        return {}
    try:
        from data.prices import fetch_price_data_parallel

        frames, _skipped = fetch_price_data_parallel(
            [s.upper() for s in symbols],
            period="150d", interval="1d", max_workers=4, chunk_size=25,
            timeout_s=10.0, rescue_missing=False, use_cache=True,
        )
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym, frame in (frames or {}).items():
        m = _range_metrics(frame)
        if m:
            out[str(sym).upper()] = m
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
      { ticker, open, last, change_dollar, chg_pct, gap_pct, adx, vwap,
        vs_vwap_pct, rvol, volume, supertrend, supertrend_direction, ewo }
    Symbols with no usable price are dropped. Sorted by |chg_pct| descending.
    """
    if not symbols:
        return []
    syms = [s.upper() for s in dict.fromkeys(symbols).keys() if str(s).strip()]
    data_feed = _get_alpaca_data_feed()
    snapshots = fetch_alpaca_snapshots(syms, feed=data_feed)
    if not snapshots:
        return []

    avg_vol = fetch_avg_daily_volume(syms, feed=data_feed) if with_rvol else {}
    # Daily-bar enrichments (EMA cross + ATR/Donchian/Bollinger) run on the
    # DISPLAY path only. The universe-wide movers *screening* pass (with_rvol=
    # False) scores from snapshots alone and never uses these — fetching daily
    # bars for thousands of tickers there was pure waste and, when Alpaca is slow,
    # a flood of timeouts.
    ema_crosses = fetch_ema_crosses(syms) if with_rvol else {}
    range_metrics = fetch_daily_range_metrics(syms) if with_rvol else {}
    source = _alpaca_source(data_feed)

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
        change_dollar = (last - today_open) if today_open is not None else None
        vs_vwap_pct = ((last - vwap) / vwap * 100.0) if vwap else None
        avg = avg_vol.get(sym)
        rvol = (volume / avg) if (volume and avg) else None

        row = {
                "ticker": sym,
                "open": round(today_open, 2) if today_open is not None else None,
                "last": round(last, 2),
                "previous_close": round(prev_close, 2) if prev_close is not None else None,
                "change_dollar": (
                    round(change_dollar, 2) if change_dollar is not None else None
                ),
                "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
                "vwap": round(vwap, 2) if vwap is not None else None,
                "vs_vwap_pct": round(vs_vwap_pct, 2) if vs_vwap_pct is not None else None,
                "volume": int(volume) if volume else None,
                "volume_source": source if volume else None,
                "rvol": round(rvol, 2) if rvol is not None else None,
                "rvol_source": (
                    f"{source}_current_vs_20d_{source}_avg" if rvol is not None else None
                ),
                "close_today": round(close_today, 2) if close_today is not None else None,
                "ema_cross": ema_crosses.get(sym),
                "open_source": source if today_open is not None else None,
                "open_feed": data_feed if today_open is not None else None,
                "open_timeframe": "snapshot_daily_bar",
                "last_source": source,
                "last_feed": data_feed,
                "last_timeframe": "latest_trade_or_minute_or_daily_snapshot",
                "prev_close_source": source if prev_close is not None else None,
                "previous_close_source": source if prev_close is not None else None,
                "previous_close_feed": data_feed if prev_close is not None else None,
                "previous_close_timeframe": "previous_completed_regular_session_daily_bar",
                "vwap_source": source if vwap is not None else None,
                "vwap_feed": data_feed if vwap is not None else None,
                "vwap_timeframe": "snapshot_daily_bar",
                "vwap_formula": "provider_supplied_daily_bar_vw",
                "volume_timeframe": "snapshot_daily_bar_session_cumulative",
                "rvol_period": 20,
                "rvol_formula": "current_session_volume / average_daily_volume_20",
                "market_data_feed": data_feed,
                # Latest activity timestamp — lets callers drop stale/delisted
                # names (a delisted ticker's last trade is days/weeks old).
                "trade_ts": latest_trade.get("t") or minute_bar.get("t") or daily_bar.get("t"),
                # ATR% / Donchian / Bollinger (daily). Empty dict when not fetched.
                **(range_metrics.get(sym) or {}),
            }
        row.update(calculate_day_trader_row_audit(row))
        rows.append(row)

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
