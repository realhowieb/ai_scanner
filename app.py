from __future__ import annotations

import importlib
import inspect
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import json
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

# --- Simple local DB for scan history (SQLite). ---
DB_PATH = Path(__file__).with_name("scanner.sqlite")

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_tables() -> None:
    conn = _get_conn()
    cur = conn.cursor()

    # Check if the runs table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    exists = cur.fetchone() is not None

    if exists:
        # Inspect the existing schema to ensure it has the expected columns
        cur.execute("PRAGMA table_info(runs)")
        cols = [row[1] for row in cur.fetchall()]  # row[1] is the column name
        required_cols = {"id", "name", "results_json", "created_at"}
        if not required_cols.issubset(set(cols)):
            # Old or incompatible schema: drop and recreate
            cur.execute("DROP TABLE IF EXISTS runs")
            conn.commit()

    # Create the expected schema if it doesn't exist (or after dropping old one)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

_ensure_tables()

def save_run(name: str, results_json: str) -> None:
    """Save a scan run to the local SQLite file."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO runs (name, results_json) VALUES (?, ?)",
            (name, results_json),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Fail silently; history is a convenience feature.
        pass

def list_runs(limit: int = 50):
    """Return a list of recent runs (id, name, created_at)."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, created_at FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "name": r["name"],
                "timestamp": r["created_at"],
            })
        return out
    except Exception:
        return []

def load_run_results(run_id: int) -> str:
    """Return the JSON payload for a given run id."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT results_json FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No run found with id {run_id}")
    return row["results_json"]

# Optional for Yahoo Finance universe fallback
try:
    import requests
except Exception:  # requests may not be installed in some runtimes
    requests = None

# ============================================
# Breakout Stock Scanner — Subscription Ready
# Single-file entrypoint (replaces bootstrapper)
# ============================================

# ---------- Safe import helpers ----------

def _try_import(path: str, attr: str | None = None):
    """Import a module by dotted path; optionally return a named attribute."""
    try:
        mod = importlib.import_module(path)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


def banner(msg: str, level: str = "info"):
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def safe_call(
    fn,
    *args,
    retries: int = 2,
    sleep_s: float = 0.8,
    label: str = "",
    **kwargs,
):
    """Retry wrapper to harden flaky providers (yfinance, etc.). Supports kwargs."""
    last_err = None
    for i in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            st.caption(
                f"⚠️ {label or fn.__name__} failed (attempt {i+1}/{retries+1}): {e}"
            )
            time.sleep(sleep_s)
    raise last_err


# ---------- Helper to override last prices from yfinance ----------

def _override_last_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Override df['Last'] with live-ish last trade prices."""
    if yf is None or df is None or df.empty or "Ticker" not in df.columns:
        return df
    last_map = {}
    for t in df["Ticker"].astype(str).tolist():
        try:
            tk = yf.Ticker(t)
            price = None
            try:
                fi = getattr(tk, "fast_info", {}) or {}
                price = fi.get("last_price")
            except Exception:
                price = None
            if price is None:
                try:
                    info = tk.info or {}
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
                except Exception:
                    price = None
            if price is not None and np.isfinite(price):
                last_map[t] = float(price)
        except Exception:
            continue
    if last_map:
        out = df.copy()
        if "Last" not in out.columns:
            out["Last"] = np.nan
        out["Last"] = out["Ticker"].map(last_map).fillna(out["Last"])
        return out
    return df


# ---------- Scan output coercion helper ----------
def _coerce_scan_output(out, tickers: List[str]) -> pd.DataFrame:
    """Coerce various real-scan return types into a DataFrame."""
    if out is None:
        return pd.DataFrame()
    if isinstance(out, pd.DataFrame):
        return out
    # Common patterns: list of dict rows, dict of rows, or list of tickers
    try:
        if isinstance(out, list):
            if len(out) == 0:
                return pd.DataFrame()
            if isinstance(out[0], dict):
                return pd.DataFrame(out)
            if isinstance(out[0], str):
                return pd.DataFrame({"Ticker": out})
        if isinstance(out, dict):
            # dict of ticker->score or ticker->row
            if all(isinstance(v, (int, float)) for v in out.values()):
                return pd.DataFrame({"Ticker": list(out.keys()), "BreakoutScore": list(out.values())})
            if all(isinstance(v, dict) for v in out.values()):
                rows = []
                for k, v in out.items():
                    r = {"Ticker": k}
                    r.update(v)
                    rows.append(r)
                return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


# ---------- Page config ----------

st.set_page_config(
    page_title="Breakout Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Tiers / Plans ----------

@dataclass
class Tier:
    name: str
    can_scan_sp500: bool
    can_scan_nasdaq: bool
    can_premarket: bool
    can_afterhours: bool
    can_unusual_volume: bool
    can_export_csv: bool
    can_ai_notes: bool
    max_results: int


TIERS: Dict[str, Tier] = {
    "basic": Tier(
        name="Basic",
        can_scan_sp500=True,
        can_scan_nasdaq=False,
        can_premarket=False,
        can_afterhours=False,
        can_unusual_volume=False,
        can_export_csv=False,
        can_ai_notes=False,
        max_results=25,
    ),
    "pro": Tier(
        name="Pro",
        can_scan_sp500=True,
        can_scan_nasdaq=True,
        can_premarket=True,
        can_afterhours=True,
        can_unusual_volume=True,
        can_export_csv=True,
        can_ai_notes=False,
        max_results=75,
    ),
    "premium": Tier(
        name="Premium",
        can_scan_sp500=True,
        can_scan_nasdaq=True,
        can_premarket=True,
        can_afterhours=True,
        can_unusual_volume=True,
        can_export_csv=True,
        can_ai_notes=True,
        max_results=200,
    ),
}


# ---------- Local demo user store ----------
# Replace with your real user DB / Firestore later.
# Password hashes should be bcrypt from streamlit-authenticator.

USERS_DB = {
    "howard": {
        "name": "Howard",
        # Example bcrypt hash ONLY. Replace with your own.
        "password": "$2b$12$9e7Gq1l8Jc0aZ5g2QFHOiO3nHvxT7O1s2W5U2nZfA5c7xU2p1Jk9C",
        "tier": "premium",
    }
}


def get_user_tier(username: str) -> Tier:
    tier_key = USERS_DB.get(username, {}).get("tier", "basic")
    return TIERS.get(tier_key, TIERS["basic"])


# ---------- Stripe payment links (placeholders) ----------
# Replace with real Stripe Payment Links.

STRIPE_LINKS = {
    "basic": "https://buy.stripe.com/test_basic_link",
    "pro": "https://buy.stripe.com/test_pro_link",
    "premium": "https://buy.stripe.com/test_premium_link",
}


# ---------- Auth ----------
try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None
import streamlit as st

# Optional live price override for the 'Last' column
try:
    import yfinance as yf
except Exception:
    yf = None


# Built-in candlestick fallback (no extra deps beyond plotly)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Matplotlib fallback if Plotly isn't available
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def auth_ui() -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (authenticated, username, display_name)."""
    if stauth is None:
        banner("streamlit-authenticator not installed. Running in DEMO mode.", "warning")
        return True, "howard", "Howard"

    usernames = list(USERS_DB.keys())
    authenticator = stauth.Authenticate(
        {"usernames": {u: {"name": USERS_DB[u]["name"], "password": USERS_DB[u]["password"]} for u in usernames}},
        "breakout_scanner_cookie",
        "breakout_scanner_signature",
        cookie_expiry_days=7,
    )

    name, auth_status, username = authenticator.login("Login", "main")

    if auth_status is False:
        banner("Username/password incorrect", "error")
        return False, None, None
    if auth_status is None:
        banner("Please enter your credentials.", "info")
        return False, None, None

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    return True, username, name


def pricing_sidebar():
    st.sidebar.markdown("## 💳 Upgrade")
    cols = st.sidebar.columns(3)
    for i, key in enumerate(["basic", "pro", "premium"]):
        t = TIERS[key]
        with cols[i]:
            st.markdown(f"**{t.name}**")
            st.markdown(f"- SP500: {'✅' if t.can_scan_sp500 else '❌'}")
            st.markdown(f"- NASDAQ: {'✅' if t.can_scan_nasdaq else '❌'}")
            st.markdown(f"- Export: {'✅' if t.can_export_csv else '❌'}")
            st.link_button(f"Subscribe {t.name}", STRIPE_LINKS[key])


# ---------- Universe filtering helper ----------

def filter_universe(tickers: List[str]) -> List[str]:
    """Drop symbols Yahoo commonly can't serve (preferreds, warrants, units, rights, weird junk)."""
    if not tickers:
        return []

    bad_suffixes = ("-W", "-WS", "-U", "-R")
    allowed = re.compile(r"^[A-Z][A-Z0-9.\-]*$")

    out: List[str] = []
    for t in tickers:
        if not t:
            continue
        ts = str(t).strip().upper()

        # Super-short symbols are usually junk/noise
        if len(ts) < 2:
            continue

        # Preferred/share classes like BRK$A or BAC$E
        if "$" in ts:
            continue

        # Warrants/units/rights
        if ts.endswith(bad_suffixes):
            continue

        # Extra pattern skip
        if re.search(r"\bWARRANT\b|\bRIGHT\b", ts):
            continue

        # Only keep clean ticker character set
        if not allowed.match(ts):
            continue

        out.append(ts)

    # De-dupe preserving order
    seen = set()
    deduped: List[str] = []
    for ts in out:
        if ts not in seen:
            seen.add(ts)
            deduped.append(ts)
    return deduped


def apply_liquidity_filter_batch(
    tickers: List[str],
    min_price: float = 2.0,
    min_volume: int = 300_000,
) -> List[str]:
    """Filter tickers by basic liquidity using a single yfinance batch call.

    Keeps only symbols with last close >= min_price and last daily volume >= min_volume.
    If anything goes wrong, returns the original list so scans still work.
    """
    if yf is None or not tickers:
        return tickers

    try:
        batch = yf.download(
            " ".join(tickers),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return tickers

    if batch is None or batch.empty:
        return tickers

    liquid: List[str] = []

    if isinstance(batch.columns, pd.MultiIndex):
        # Multi-ticker case: columns like (ticker, field) or (field, ticker)
        for t in tickers:
            price_series = None
            vol_series = None
            try:
                # Orientation 1: (ticker, field)
                price_series = batch[(t, "Close")]
                vol_series = batch[(t, "Volume")]
            except Exception:
                try:
                    # Orientation 2: (field, ticker)
                    price_series = batch[("Close", t)]
                    vol_series = batch[("Volume", t)]
                except Exception:
                    continue

            price_series = price_series.dropna()
            vol_series = vol_series.dropna()
            if price_series.empty or vol_series.empty:
                continue

            last_price = float(price_series.iloc[-1])
            last_vol = float(vol_series.iloc[-1])
            if last_price >= min_price and last_vol >= min_volume:
                liquid.append(t)
    else:
        # Single-ticker case
        price_series = batch.get("Close")
        vol_series = batch.get("Volume")
        if price_series is not None and vol_series is not None:
            price_series = price_series.dropna()
            vol_series = vol_series.dropna()
            if not price_series.empty and not vol_series.empty:
                last_price = float(price_series.iloc[-1])
                last_vol = float(vol_series.iloc[-1])
                if last_price >= min_price and last_vol >= min_volume:
                    # Only one ticker here, keep it
                    return tickers

    return liquid or tickers

# ---------- Universe loaders ----------
# Try your real loaders first, fallback to tiny defaults.

_load_sp500 = (
    _try_import("ai_scanner.ui.universe", "load_sp500_universe")
    or _try_import("ai_scanner.ui.universe", "get_sp500")
    or _try_import("ui.universe", "load_sp500_universe")
    or _try_import("ui.universe", "get_sp500")
)

_load_nasdaq = (
    _try_import("ai_scanner.ui.universe", "load_nasdaq_universe")
    or _try_import("ai_scanner.ui.universe", "get_nasdaq")
    or _try_import("ui.universe", "load_nasdaq_universe")
    or _try_import("ui.universe", "get_nasdaq")
)


def _fetch_yahoo_universe(scr_id: str, count: int = 1000) -> List[str]:
    """Fetch predefined Yahoo Finance screener tickers.

    Yahoo can rate-limit (429). We send a browser UA and retry a couple times with
    exponential backoff. If still limited, caller should fall back to Wikipedia.
    """
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
            quotes = (
                data.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
            )
            tickers = [q.get("symbol") for q in quotes if q.get("symbol")]
            # De-dupe while preserving order
            seen = set()
            out = []
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
        except Exception as e:
            last_err = e
            # brief pause before next attempt
            time.sleep(0.8 * (attempt + 1))

    raise last_err or RuntimeError("Yahoo universe fetch failed")
def _note_yahoo_fail(which: str, err: Exception):
    key = f"yahoo_fail_noted_{which}"
    if not st.session_state.get(key):
        st.session_state[key] = True
        st.caption(f"Yahoo {which} universe fallback unavailable (rate-limited). Using Wikipedia instead.")


def _fetch_wikipedia_table(url: str, col: str) -> List[str]:
    """Fallback if Yahoo endpoint changes; pulls ticker lists from Wikipedia."""
    try:
        tables = pd.read_html(url)
        for t in tables:
            if col in t.columns:
                tickers = t[col].astype(str).str.replace(".", "-", regex=False).tolist()
                return tickers
    except Exception:
        return []
    return []


# Official NASDAQ Trader listings fallback
def _fetch_nasdaq_official_listings() -> List[str]:
    """Fetch NASDAQ-listed tickers from NASDAQ Trader official symbol directory.

    These files are pipe-delimited and maintained by NASDAQ. This is a stable way to
    get the NASDAQ Composite universe without Yahoo 429s.
    """
    if requests is None:
        raise RuntimeError("requests not available")

    urls = [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
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
            sym = row.get("Symbol") or row.get("ACT Symbol")
            if sym:
                sym = sym.strip().replace(".", "-")
                if sym and sym[0].isalnum():
                    tickers.append(sym)

    # De-dupe while preserving order
    seen = set()
    out = []
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
                    tickers = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
                # de-dupe while preserving order
                seen = set(); out = []
                for t in tickers:
                    if t not in seen:
                        seen.add(t); out.append(t)
                if out and len(out) >= 400:
                    st.caption(f"Loaded SP500 universe from {os.path.basename(path)} ({len(out)} tickers).")
                    return out
                else:
                    st.caption(f"Local {os.path.basename(path)} returned {len(out)} tickers; expecting full list.")
        except Exception as e:
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
        except Exception as e:
            st.caption(f"Local SP500 universe loader failed: {e}. Using Wikipedia instead.")

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
    except Exception as e:
        _note_yahoo_fail("SP500", e)

    # 4) Tiny last-resort default
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_nasdaq_universe() -> List[str]:
    if callable(_load_nasdaq):
        return list(_load_nasdaq())

    # ✅ Official NASDAQ Trader listings (NASDAQ Composite universe)
    try:
        tickers = _fetch_nasdaq_official_listings()
        if tickers:
            return tickers
    except Exception as e:
        st.caption(f"Official NASDAQ listings fallback failed: {e}")

    # Yahoo Finance predefined screener fallback (Nasdaq 100)
    try:
        tickers = _fetch_yahoo_universe("nasdaq100", count=120)
        if tickers:
            return tickers
    except Exception as e:
        _note_yahoo_fail("NASDAQ", e)

    # Wikipedia fallback (Nasdaq-100)
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        col="Ticker",
    )
    if wiki:
        return wiki

    return ["TSLA", "PLTR", "AMD", "SOFI", "SNOW", "CRWD"]


# ---------- Scan engine ----------
# Try your real scan function first; fallback to a safe stub.

_real_scan = (
    # Preferred locations (your project)
    _try_import("ai_scanner.scan.breakout_scanner", "run_breakout_scan")
    or _try_import("ai_scanner.scan.breakout", "run_breakout_scan")
    or _try_import("ai_scanner.breakout", "run_breakout_scan")
    or _try_import("ai_scanner.breakout_scanner", "run_breakout_scan")

    # Alternate common names
    or _try_import("ai_scanner.scan.breakout_scanner", "breakout_scanner")
    or _try_import("ai_scanner.scan.breakout", "breakout_scanner")
    or _try_import("ai_scanner.breakout", "breakout_scanner")
    or _try_import("ai_scanner.breakout_scanner", "breakout_scanner")

    # Root-level / legacy paths
    or _try_import("scan.breakout_scanner", "run_breakout_scan")
    or _try_import("scan.breakout", "run_breakout_scan")
    or _try_import("breakout_scanner", "run_breakout_scan")
    or _try_import("breakout", "run_breakout_scan")
    or _try_import("breakout_scanner", "breakout_scanner")
    or _try_import("breakout", "breakout_scanner")
)


def run_breakout_scan(
    tickers: List[str],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool = True,
) -> pd.DataFrame:
    use_stub = False
    if callable(_real_scan):
        # Try to match your real scan function signature safely.
        try:
            sig = inspect.signature(_real_scan)
            accepted = set(sig.parameters.keys())

            call_kwargs = {
                "premarket": premarket,
                "afterhours": afterhours,
                "unusual_volume": unusual_volume,
                "min_gap": min_gap,
                "min_price": min_price,
                "max_price": max_price,
                "top_n": top_n,
                "diagnostics": diagnostics,
            }
            filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

            # Preferred call: tickers as first positional arg.
            try:
                t_start = time.time()
                out = _real_scan(tickers, **filtered_kwargs) if accepted else _real_scan(tickers)
                df_out = _coerce_scan_output(out, tickers)
                # If the real scanner returns empty almost instantly, treat as a no-op and fall back.
                if df_out.empty and (time.time() - t_start) < 0.5:
                    st.warning(
                        "Real breakout scanner returned 0 rows instantly. Falling back to stub. "
                        "This usually means a signature/universe mismatch or an internal early-exit."
                    )
                    use_stub = True
                    raise RuntimeError("real_scan_empty_fast")
                return df_out
            except AttributeError as e:
                # Some real scanners expect a dict-like universe and call .items().
                if "items" in str(e) and isinstance(tickers, list):
                    retry_universes = [
                        {t: {} for t in tickers},
                        {t: None for t in tickers},
                    ]
                    last_retry_err = None
                    for uni in retry_universes:
                        try:
                            t_start = time.time()
                            out = _real_scan(uni, **filtered_kwargs) if accepted else _real_scan(uni)
                            df_out = _coerce_scan_output(out, tickers)
                            if df_out.empty and (time.time() - t_start) < 0.5:
                                use_stub = True
                                raise RuntimeError("real_scan_empty_fast")
                            return df_out
                        except Exception as re:
                            last_retry_err = re
                            continue
                    # If all retries failed, raise the last retry error (don’t mask it).
                    if last_retry_err is not None:
                        raise last_retry_err
                raise

        except TypeError as e:
            # Fallback 1: maybe your function wants no kwargs at all.
            try:
                return _real_scan(tickers)
            except Exception:
                raise e
        except Exception as e:
            if str(e) == "real_scan_empty_fast" or "real_scan_empty_fast" in str(e):
                use_stub = True
            else:
                raise

    if callable(_real_scan) and not use_stub:
        # Real scan succeeded and returned rows.
        return pd.DataFrame()  # Should be unreachable, but keeps type checkers happy.

    if use_stub and diagnostics and not st.session_state.get("noted_stub_scan"):
        st.session_state["noted_stub_scan"] = True
        st.warning(
            "Real breakout scanner not found. Using random stub results. "
            "Check your module path for run_breakout_scan/breakout_scanner."
        )
    # ---------- Fallback stub (used when real scan not found or no-ops) ----------
    rows = []
    for t in tickers:
        price = float(np.random.uniform(min_price, max_price))
        vol = int(np.random.randint(1_000_000, 80_000_000))
        gap = float(np.random.uniform(-5, 12))
        score = float(np.random.uniform(0, 100))
        rows.append(
            {
                "Ticker": t,
                "BreakoutScore": round(score, 2),
                "Last": round(price, 2),
                "Volume": vol,
                "Gap%": round(gap, 2),
                "Premarket": premarket,
                "AfterHours": afterhours,
                "UnusualVol": unusual_volume and vol > 20_000_000,
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["Last"].between(min_price, max_price)]
    df = df[df["Gap%"] >= min_gap]
    df = df.sort_values("BreakoutScore", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------- Cached scan wrapper ----------
@st.cache_data(ttl=600, show_spinner=False)
def cached_real_scan(
    tickers: Tuple[str, ...],
    *,
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    diagnostics: bool,
) -> pd.DataFrame:
    """Cached wrapper around run_breakout_scan.

    Uses a tuple of tickers so Streamlit can hash the arguments. This makes
    re-running the same scan (same universe + filters) much faster.
    """
    return run_breakout_scan(
        list(tickers),
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        diagnostics=diagnostics,
    )


# ---------- Chart renderer ----------
# Custom chart renderer disabled; always use built‑in charts.
_real_chart = None


def _fetch_unadjusted_ohlc(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch unadjusted OHLCV for charting with multiple retries and normalization."""
    if yf is None:
        return None

    def _norm(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df = df.reset_index()
        cols_lower = {c.lower() for c in df.columns}
        if "date" not in cols_lower and "datetime" in cols_lower:
            for c in df.columns:
                if c.lower() == "datetime":
                    df.rename(columns={c: "Date"}, inplace=True)
                    break
        if "date" not in {c.lower() for c in df.columns} and "Date" not in df.columns:
            return None
        return df

    attempts = []
    attempts.append((ticker, period, interval))
    attempts.append((ticker, "3mo", interval))
    attempts.append((ticker, "1mo", interval))
    if "." in ticker:
        attempts.append((ticker.replace(".", "-"), period, interval))
    if ticker.endswith((".NS", ".TO", ".L", ".AX", ".SA", ".HK", ".F")):
        attempts.append((ticker.split(".")[0], period, interval))

    for sym, per, inter in attempts:
        try:
            df = yf.download(
                sym,
                period=per,
                interval=inter,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            out = _norm(df)
            if out is not None:
                return out
        except Exception:
            continue

    try:
        hist = yf.Ticker(ticker).history(period="6mo", interval=interval, auto_adjust=False)
        out = _norm(hist)
        if out is not None:
            return out
    except Exception:
        pass

    return None


def _render_builtin_candlestick(ticker: str):
    """Render a candlestick chart using unadjusted OHLC.

    Uses Plotly if available; otherwise falls back to a simple matplotlib line chart
    with EMA overlays and resistance.
    """
    df = _fetch_unadjusted_ohlc(ticker)
    if df is None or df.empty:
        st.warning(
            f"No OHLC data available for {ticker}. This may occur if the symbol is OTC, delisted, newly listed, "
            "or not supported by Yahoo Finance. Try another ticker."
        )
        return

    # Indicators
    try:
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["Res20"] = df["High"].rolling(20, min_periods=1).max()
        if "Volume" in df.columns:
            df["VolSMA20"] = df["Volume"].rolling(20, min_periods=1).mean()
    except Exception:
        pass

    # --- Matplotlib fallback if Plotly is missing ---
    if go is None or plt is None:
        if plt is None:
            st.write("No chart backend available (Plotly and Matplotlib missing).")
            return
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Close"], label="Close")
        if "EMA9" in df.columns:
            ax.plot(df["Date"], df["EMA9"], label="EMA9")
        if "EMA21" in df.columns:
            ax.plot(df["Date"], df["EMA21"], label="EMA21")
        if "Res20" in df.columns:
            ax.plot(df["Date"], df["Res20"], label="Resistance 20D High")
        ax.set_title(f"{ticker} Price (unadjusted) with EMAs")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        st.pyplot(fig, use_container_width=True)

        if "Volume" in df.columns:
            vfig, vax = plt.subplots()
            vax.bar(df["Date"], df["Volume"], label="Volume")
            if "VolSMA20" in df.columns:
                vax.plot(df["Date"], df["VolSMA20"], label="Vol SMA20")
            vax.set_title(f"{ticker} Volume")
            vax.legend(loc="upper left")
            st.pyplot(vfig, use_container_width=True)
        return

    # Expected columns from yfinance: Date, Open, High, Low, Close, Volume
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df.get("Open"),
            high=df.get("High"),
            low=df.get("Low"),
            close=df.get("Close"),
            name=ticker,
        )
    )
    if "EMA9" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA9"], name="EMA9", mode="lines"))
    if "EMA21" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA21"], name="EMA21", mode="lines"))
    if "Res20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Res20"], name="Resistance(20D High)", mode="lines"))
    fig.update_layout(
        title=f"{ticker} Candlestick (unadjusted) • EMA9/EMA21 • 20D Resistance",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional volume bar
    if "Volume" in df.columns:
        st.caption("Volume")
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
        if "VolSMA20" in df.columns:
            vol_fig.add_trace(go.Scatter(x=df["Date"], y=df["VolSMA20"], name="Vol SMA20", mode="lines"))
        vol_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(vol_fig, use_container_width=True)


def render_chart_for_ticker(ticker: str, force_builtin: bool = False):
    """Always render built‑in unadjusted candlestick charts."""
    _render_builtin_candlestick(ticker)


# ---------- Main UI ----------

def main():
    st.title("📈 Breakout Stock Scanner")
    st.caption("Money Moves • AI Breakout Score • Subscription Ready")

    authed, username, display_name = auth_ui()
    if not authed:
        st.stop()

    tier = get_user_tier(username)

    st.sidebar.markdown(f"### 👤 {display_name}")
    st.sidebar.markdown(f"**Plan:** `{tier.name}`")

    pricing_sidebar()

    # Sidebar filters
    st.sidebar.markdown("## Filters")
    min_gap = st.sidebar.slider("Min Gap %", -10.0, 20.0, 2.0, 0.5)
    min_price = st.sidebar.number_input("Min Price", 0.5, 500.0, 3.0, 0.5)
    max_price = st.sidebar.number_input("Max Price", 1.0, 5000.0, 50.0, 1.0)
    top_n = st.sidebar.slider("Top N Results", 5, tier.max_results, min(25, tier.max_results), 5)

    max_nasdaq_scan = st.sidebar.number_input(
        "Max NASDAQ tickers to scan",
        min_value=100,
        max_value=6000,
        value=1200,
        step=100,
        help="Caps NASDAQ universe to speed up scans. Applied to NASDAQ + Combo scans.",
    )

    max_combo_scan = st.sidebar.number_input(
        "Max Combo tickers to scan",
        min_value=100,
        max_value=6000,
        value=1000,
        step=100,
        help="Caps SP500+NASDAQ universe for Combo scans.",
    )

    premarket = st.sidebar.checkbox("Include Premarket Scan", value=False, disabled=not tier.can_premarket)
    afterhours = st.sidebar.checkbox("Include After-hours Scan", value=False, disabled=not tier.can_afterhours)
    unusual_vol = st.sidebar.checkbox("Unusual Volume Filter", value=True, disabled=not tier.can_unusual_volume)

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox("Show diagnostics", value=True)

    # Load universes
    sp500 = safe_call(load_sp500_universe, label="SP500 universe")
    nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")

    # Filter out Yahoo-incompatible symbols (preferreds/warrants/units/rights)
    sp500 = filter_universe(sp500)
    nasdaq = filter_universe(nasdaq)

    nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]

    combo_universe = sp500 + nasdaq_capped

    # Batch liquidity filter for Combo: drop thin/penny names before scanning
    combo_liquid = apply_liquidity_filter_batch(combo_universe)
    combo_capped = combo_liquid[: int(max_combo_scan)]

    # Universe diagnostics (your preference)
    with st.expander("Universe Info", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**SP500 size:** {len(sp500)}")
            st.caption(f"Sample: {', '.join(sp500[:20])}")
        with c2:
            st.markdown(f"**NASDAQ size:** {len(nasdaq_capped)} (capped from {len(nasdaq)})")
            st.caption(f"Sample: {', '.join(nasdaq_capped[:20])}")

    # Buttons (hard-wired universes)
    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        run_sp500_btn = st.button("Run SP500 Scan", use_container_width=True, disabled=not tier.can_scan_sp500)
        st.caption("Runs SP500 regardless of sidebar universe.")

    with b2:
        run_nasdaq_btn = st.button("Run NASDAQ Scan", use_container_width=True, disabled=not tier.can_scan_nasdaq)
        st.caption("Runs NASDAQ regardless of sidebar universe.")

    with b3:
        run_combo_btn = st.button(
            "Run Combo Scan (SP500+NASDAQ)",
            use_container_width=True,
            disabled=not (tier.can_scan_sp500 and tier.can_scan_nasdaq),
        )
        st.caption("Pro/Premium only.")

    # Session state for results
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    def do_scan(tickers: List[str], label: str):
        def _run_scan_body():
            t0 = time.time()
            try:
                st.caption(f"🔎 Scanning {len(tickers)} tickers for {label}...")
                if len(tickers) < 50:
                    st.warning(
                        f"{label} universe is very small ({len(tickers)} tickers). "
                        "This usually means a fallback/stub universe is still being used."
                    )

                df = safe_call(
                    cached_real_scan,
                    tuple(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    diagnostics=diagnostics,
                    label="cached_real_scan",
                )

                # Apply Top N cap here to avoid doing last-price overrides on hundreds of rows.
                if df is not None and not df.empty:
                    df = df.head(top_n).reset_index(drop=True)
                    df = _override_last_prices(df)

                st.caption(f"✅ {label}: {len(df)} results returned from scan.")
                dt = time.time() - t0
                st.session_state.results_df = df
                banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")
                # Persist this scan to the runs DB (local SQLite history)
                try:
                    results_json = df.to_json(orient="records") if df is not None else "[]"
                    run_name = f"{label} | {len(df)} results | {dt:.1f}s"
                    save_run(run_name, results_json)
                except Exception:
                    # Never fail the UI just because DB logging failed
                    pass
            except Exception as e:
                banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

        # Some environments (e.g., restricted sandboxes, Python 3.13 runtimes) may not
        # allow starting new threads, which Streamlit's spinner uses internally.
        # Wrap the spinner in a try/except and fall back to running without it.
        try:
            with st.spinner(f"Scanning {label}..."):
                _run_scan_body()
        except Exception:
            _run_scan_body()

    if run_sp500_btn:
        do_scan(sp500, "SP500")
    if run_nasdaq_btn:
        do_scan(nasdaq_capped, "NASDAQ")
    if run_combo_btn:
        do_scan(combo_capped, "Combo")

    df = st.session_state.results_df

    if df is not None and not df.empty:
        st.subheader("Results")
        st.caption(
            f"Showing {len(df)} results. Increase 'Top N Results' in the sidebar to see more, "
            "or relax filters (Min Gap %, price range, Unusual Volume)."
        )
        st.dataframe(df, use_container_width=True, height=420)

        # Export (tier-gated)
        if tier.can_export_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="breakout_results.csv",
                mime="text/csv",
                use_container_width=False,
            )
        else:
            st.info("CSV export is available on Pro/Premium.")

        # Chart picker
        st.subheader("Charts")
        pick = st.selectbox("Select ticker to chart", df["Ticker"].tolist())
        render_chart_for_ticker(pick)

        # AI notes placeholder (tier-gated)
        if tier.can_ai_notes:
            st.subheader("AI Notes (Premium)")
            st.write("Add your AI commentary here.")
        else:
            st.caption("AI Notes are Premium-only.")
    else:
        st.caption("Run a scan to see results.")

    # --- Scan History (DB-backed via local SQLite) ---
    with st.expander("📜 Scan History", expanded=False):
        runs_list = []
        try:
            runs_list = list_runs()
        except Exception:
            st.caption("History unavailable (DB error).")

        if runs_list:
            options = []
            for r in runs_list:
                # Expect dict-like rows from list_runs
                rid = r.get("id") if isinstance(r, dict) else None
                name = r.get("name") if isinstance(r, dict) else str(r)
                ts = r.get("timestamp") if isinstance(r, dict) else None

                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                elif ts is not None:
                    ts_str = str(ts)
                else:
                    ts_str = ""

                label_str = f"#{rid} — {name}"
                if ts_str:
                    label_str += f" — {ts_str}"
                options.append((label_str, rid))

            if options:
                labels = [lbl for (lbl, _rid) in options]
                selected_label = st.selectbox("Select a past scan to load:", labels, index=0)
                selected_id = None
                for lbl, _rid in options:
                    if lbl == selected_label:
                        selected_id = _rid
                        break

                col_hist1, col_hist2 = st.columns([1, 1])
                with col_hist1:
                    if st.button("Load Selected Scan") and selected_id is not None:
                        try:
                            payload = load_run_results(int(selected_id))
                            hist_df = pd.read_json(payload)
                            st.session_state.results_df = hist_df
                            st.success(f"Loaded scan #{selected_id} from history with {len(hist_df)} rows.")
                        except Exception as e:
                            st.error(f"Failed to load scan #{selected_id}: {e}")
                with col_hist2:
                    st.caption(
                        "History is stored in a local scanner.sqlite file next to app.py."
                    )
            else:
                st.caption("No past scans saved yet.")
        else:
            st.caption("No past scans saved yet.")

    st.divider()
    st.caption("⚠️ Not financial advice. Educational tool only.")


if __name__ == "__main__":
    main()