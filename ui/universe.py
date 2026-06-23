from __future__ import annotations

import os
import time
from typing import List

import pandas as pd
import streamlit as st
from ui.universe_db import db_get_universe, db_upsert_universe, try_import

try:
    import requests
except ImportError:
    requests = None

# Liquidity filter (repo-local import first; packaged import fallback)
try:
    from scan.liquidity import apply_liquidity_filter_batch as _core_liquidity_filter
except ImportError:
    try:
        from ai_scanner.scan.liquidity import apply_liquidity_filter_batch as _core_liquidity_filter
    except ImportError:
        _core_liquidity_filter = None  # type: ignore


UNIVERSE_PROVIDER_ERRORS = (
    RuntimeError,
    TimeoutError,
    ConnectionError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
)


# ---------- Universe loaders ----------
# Try your real loaders first, fallback to tiny defaults.

_load_sp500 = (
    try_import("ai_scanner.ui.universe", "load_sp500_universe")
    or try_import("ai_scanner.ui.universe", "get_sp500")
    or try_import("ui.universe", "load_sp500_universe")
    or try_import("ui.universe", "get_sp500")
)

_load_nasdaq = (
    try_import("ai_scanner.ui.universe", "load_nasdaq_universe")
    or try_import("ai_scanner.ui.universe", "get_nasdaq")
    or try_import("ui.universe", "load_nasdaq_universe")
    or try_import("ui.universe", "get_nasdaq")
)


def _fetch_yahoo_universe(scr_id: str, count: int = 1000) -> List[str]:
    """Fetch predefined Yahoo Finance screener tickers."""
    if requests is None:
        raise RuntimeError("requests not available")

    url = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"scrIds": scr_id, "count": count, "start": 0}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.yahoo.com/",
    }

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=12)
            if r.status_code == 429:
                # backoff and retry
                time.sleep(1.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            tickers = [q.get("symbol") for q in quotes if q.get("symbol")]
            # De-dupe while preserving order
            seen = set()
            out: List[str] = []
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
        except UNIVERSE_PROVIDER_ERRORS as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))

    raise last_err or RuntimeError("Yahoo universe fetch failed")


def _note_yahoo_fail(which: str, err: Exception):
    key = f"yahoo_fail_noted_{which}"
    if not st.session_state.get(key):
        st.session_state[key] = True
        st.caption(
            f"Yahoo {which} universe fallback unavailable (rate-limited). Using Wikipedia instead."
        )


def _fetch_wikipedia_table(url: str, col: str) -> List[str]:
    """Fallback if Yahoo endpoint changes; pulls ticker lists from Wikipedia."""
    try:
        tables = pd.read_html(url)
        for t in tables:
            if col in t.columns:
                tickers = (
                    t[col]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                return tickers
    except UNIVERSE_PROVIDER_ERRORS:
        return []
    return []


def _fetch_nasdaq_official_listings() -> List[str]:
    """Fetch NASDAQ-listed tickers from NASDAQ Trader official symbol directory (nasdaqlisted.txt only)."""
    if requests is None:
        raise RuntimeError("requests not available")

    # NASDAQ universe should be NASDAQ-listed only.
    # Do NOT include otherlisted.txt here (NYSE/AMEX/etc + preferred/warrants/units -> yfinance spam/timeouts)
    urls = [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ]

    tickers: List[str] = []
    for url in urls:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        lines = r.text.splitlines()
        if not lines:
            continue

        header = lines[0].split("|")
        for line in lines[1:]:
            if line.startswith("File Creation Time"):
                break
            parts = line.split("|")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))

            # Skip known non-tradable / test issues when the column exists
            # (nasdaqlisted.txt includes "Test Issue"; otherlisted.txt may not).
            test_issue = (row.get("Test Issue") or "").strip().upper()
            if test_issue == "Y":
                continue

            sym_raw = row.get("Symbol") or row.get("ACT Symbol")
            if sym_raw:
                sym_raw = sym_raw.strip()

                # Skip preferred/special series that NASDAQ denotes with "$" (e.g., "AGM$D").
                # These spam yfinance with "no data" and slow scans.
                if "$" in sym_raw:
                    continue

                # Skip obvious non-common-equity instruments by name when available
                sec_name = (
                    row.get("Security Name") or row.get("SecurityName") or ""
                ).strip().lower()
                if sec_name:
                    if any(
                        k in sec_name for k in ("warrant", "unit", "right", "preferred")
                    ):
                        continue

                # Normalize: dot class shares -> dash
                sym = sym_raw.replace(".", "-")

                # Basic sanity: must start with an alnum
                if sym and sym[0].isalnum():
                    tickers.append(sym)

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_sp500_universe() -> List[str]:
    # 0) Preferred: local sp500.txt (same folder as app.py or in ./data)
    local_paths = [
        os.path.join(os.path.dirname(__file__), "sp500.txt"),
        os.path.join(os.path.dirname(__file__), "data", "sp500.txt"),
        os.path.join(os.getcwd(), "sp500.txt"),
        os.path.join(os.getcwd(), "data", "sp500.txt"),
    ]
    for path in local_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    tickers = [
                        ln.strip()
                        for ln in f
                        if ln.strip() and not ln.startswith("#")
                    ]
                seen = set()
                out: List[str] = []
                for t in tickers:
                    if t not in seen:
                        seen.add(t)
                        out.append(t)
                if out and len(out) >= 400:
                    st.caption(
                        f"Loaded SP500 universe from {os.path.basename(path)} ({len(out)} tickers)."
                    )
                    return out
                else:
                    st.caption(
                        f"Local {os.path.basename(path)} returned {len(out)} tickers; expecting full list."
                    )
        except UNIVERSE_PROVIDER_ERRORS as e:
            st.caption(f"Failed loading local SP500 file at {path}: {e}")

    # 1) Prefer your local/custom loader if it exists
    if callable(_load_sp500):
        try:
            local = list(_load_sp500())
            if local and len(local) >= 100:
                return local
            else:
                st.caption(
                    f"Local SP500 universe returned {len(local) if local else 0} tickers; using Wikipedia instead."
                )
        except UNIVERSE_PROVIDER_ERRORS as e:
            st.caption(
                f"Local SP500 universe loader failed: {e}. Using Wikipedia instead."
            )

    # 2) Stable primary fallback: Wikipedia S&P 500 list
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        col="Symbol",
    )
    if wiki and len(wiki) >= 450:
        return wiki

    # 3) Optional Yahoo fallback (may rate-limit)
    try:
        tickers = _fetch_yahoo_universe("sp500", count=520)
        if tickers:
            return tickers
    except UNIVERSE_PROVIDER_ERRORS as e:
        _note_yahoo_fail("SP500", e)

    # 4) Tiny last-resort default
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_nasdaq_universe() -> List[str]:
    """NASDAQ universe loader.

    Order:
      1) DB cache (if refreshed recently)
      2) Your local/custom loader (if present)
      3) NASDAQ Trader official listings (cleaned)
      4) Yahoo (Nasdaq-100) / Wikipedia fallbacks

    If we fetch from the network, we attempt to persist back into DB.
    """
    # 1) DB-first
    db_tickers, meta = db_get_universe("nasdaq", max_age_hours=24.0)
    if db_tickers and len(db_tickers) >= 500:
        # Optional tiny caption for diagnostics (won't spam once cached)
        st.caption(f"Loaded NASDAQ universe from DB ({len(db_tickers)} tickers).")
        return filter_universe(db_tickers)

    # 2) Prefer your local/custom loader if it exists
    if callable(_load_nasdaq):
        try:
            local = list(_load_nasdaq())
            if local and len(local) >= 100:
                # Best-effort persist
                db_upsert_universe("nasdaq", local, source="local_loader")
                return filter_universe(local)
        except UNIVERSE_PROVIDER_ERRORS as e:
            st.caption(f"Local NASDAQ universe loader failed: {e}. Falling back.")

    # 3) Official NASDAQ Trader listings (NASDAQ Composite-ish universe)
    try:
        tickers = _fetch_nasdaq_official_listings()
        if tickers:
            # Persist cleaned list
            db_upsert_universe("nasdaq", tickers, source="nasdaqtrader")
            return filter_universe(tickers)
    except UNIVERSE_PROVIDER_ERRORS as e:
        st.caption(f"Official NASDAQ listings fallback failed: {e}")

    # 4) Yahoo Finance predefined screener fallback (Nasdaq 100)
    try:
        tickers = _fetch_yahoo_universe("nasdaq100", count=120)
        if tickers:
            db_upsert_universe("nasdaq", tickers, source="yahoo_nasdaq100")
            return filter_universe(tickers)
    except UNIVERSE_PROVIDER_ERRORS as e:
        _note_yahoo_fail("NASDAQ", e)

    # 5) Wikipedia fallback (Nasdaq-100)
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        col="Ticker",
    )
    if wiki:
        db_upsert_universe("nasdaq", wiki, source="wikipedia_nasdaq100")
        return filter_universe(wiki)

    return ["TSLA", "PLTR", "AMD", "SOFI", "SNOW", "CRWD"]


def filter_universe(tickers: List[str]) -> List[str]:
    """Basic cleaning for ticker universes (remove obvious junk + preferred/warrant noise).

    This is provider-oriented cleaning (yfinance/quotes). We aggressively drop
    instruments that frequently fail lookups (preferred series, units, warrants),
    while preserving a small set of well-known class-share tickers.
    """

    # Bases that legitimately have common-share class suffixes like "-A"/"-B"
    # (avoid accidentally dropping BF-B, BRK-B, etc.)
    CLASS_SHARE_BASES = {
        "BF",   # BF.A / BF.B
        "BRK",  # BRK.A / BRK.B
        "HEI",  # HEI.A
        "LEN",  # LEN.B
        "LBRD", "LBRK",  # Liberty tracking stocks
    }

    cleaned: List[str] = []

    # Pre-pass: normalize + strip obvious prefix noise
    normalized: List[str] = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        sym = t.strip().upper()
        if not sym:
            continue

        # Some feeds/logging include a leading "$" (e.g., "$ADC-A"). Strip it.
        # If the symbol still contains "$" after this, treat it as non-common.
        sym = sym.lstrip("$")

        normalized.append(sym)

    # Build a base set for preferred-series heuristics
    bases = {s.split("-", 1)[0] for s in normalized if s and "-" in s}
    bases |= {s for s in normalized if s and "-" not in s}

    for sym in normalized:
        # Drop any remaining "$" (preferred series often encoded with "$")
        if "$" in sym:
            continue

        # Ignore symbols that contain spaces or obvious non-equity prefixes
        if " " in sym:
            continue
        if sym.startswith("^"):
            continue

        # Drop symbols with characters that frequently cause provider lookups to fail
        if any(ch in sym for ch in ["/", "\\", ":", "="]):
            continue

        # Common NASDAQ/NYSE suffixes that are often not regular common shares
        # (Units/Warrants/Rights). These frequently fail provider lookups.
        if (
            sym.endswith("-W")
            or sym.endswith("-WS")
            or sym.endswith("-U")
            or sym.endswith("-R")
        ):
            continue

        # Preferred-series / odd-lot heuristic:
        # If we see BASE-SUFFIX where suffix is a single letter and BASE also exists,
        # it's often a preferred series (e.g., ADC-A, AGM-D, etc.).
        # Keep only for a few known common-share class bases.
        if "-" in sym:
            base, suf = sym.split("-", 1)
            if base and suf and len(suf) == 1 and base in bases:
                if base not in CLASS_SHARE_BASES:
                    continue

        if len(sym) > 10:
            continue

        cleaned.append(sym)

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for sym in cleaned:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def apply_liquidity_filter_batch(
    tickers: List[str],
    *,
    min_price: float,
    min_avg_dollar_vol: float,
) -> List[str]:
    """Filter tickers by approximate price and 20D dollar volume.

    Delegates to the shared scan.liquidity.apply_liquidity_filter_batch,
    which is now Alpaca-first with yfinance fallback.
    """
    if not tickers:
        return []

    if _core_liquidity_filter is None:
        # Core liquidity filter is unavailable (import failed); do not block the app.
        return tickers

    try:
        return _core_liquidity_filter(
            tickers,
            min_price=min_price,
            min_avg_dollar_vol=min_avg_dollar_vol,
        )
    except UNIVERSE_PROVIDER_ERRORS:
        # If anything goes wrong in the core filter, fall back to returning the
        # unfiltered universe rather than failing the entire app.
        return tickers
