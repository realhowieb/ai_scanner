from __future__ import annotations

import importlib
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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


def safe_call(fn, *args, retries: int = 2, sleep_s: float = 0.8, label: str = "", **kwargs):
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
    if callable(_load_sp500):
        return list(_load_sp500())

    # ✅ Stable primary fallback: Wikipedia S&P 500 list
    wiki = _fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        col="Symbol",
    )
    if wiki:
        return wiki

    # Optional Yahoo fallback (may rate-limit)
    try:
        tickers = _fetch_yahoo_universe("sp500", count=520)
        if tickers:
            return tickers
    except Exception as e:
        _note_yahoo_fail("SP500", e)

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
    _try_import("ai_scanner.scan.breakout_scanner", "run_breakout_scan")
    or _try_import("ai_scanner.scan.breakout", "run_breakout_scan")
    or _try_import("scan.breakout_scanner", "run_breakout_scan")
    or _try_import("scan.breakout", "run_breakout_scan")
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
    if callable(_real_scan):
        # Your real scanner should accept these kwargs. If not, adapt here.
        return _real_scan(
            tickers,
            premarket=premarket,
            afterhours=afterhours,
            unusual_volume=unusual_volume,
            min_gap=min_gap,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
            diagnostics=diagnostics,
        )

    # ---------- Fallback stub so app still runs ----------
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


# ---------- Chart renderer ----------
_real_chart = (
    _try_import("ai_scanner.ui.components", "render_candlestick")
    or _try_import("ai_scanner.ui.components", "render_chart")
    or _try_import("ui.components", "render_candlestick")
    or _try_import("ui.components", "render_chart")
)


def render_chart_for_ticker(ticker: str):
    if callable(_real_chart):
        return _real_chart(ticker)
    st.write(f"📊 Chart placeholder for **{ticker}**")


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

    premarket = st.sidebar.checkbox("Include Premarket Scan", value=False, disabled=not tier.can_premarket)
    afterhours = st.sidebar.checkbox("Include After-hours Scan", value=False, disabled=not tier.can_afterhours)
    unusual_vol = st.sidebar.checkbox("Unusual Volume Filter", value=True, disabled=not tier.can_unusual_volume)

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox("Show diagnostics", value=True)

    # Load universes
    sp500 = safe_call(load_sp500_universe, label="SP500 universe")
    nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")

    # Universe diagnostics (your preference)
    with st.expander("Universe Info", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**SP500 size:** {len(sp500)}")
            st.caption(f"Sample: {', '.join(sp500[:20])}")
        with c2:
            st.markdown(f"**NASDAQ size:** {len(nasdaq)}")
            st.caption(f"Sample: {', '.join(nasdaq[:20])}")

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
        with st.spinner(f"Scanning {label}..."):
            t0 = time.time()
            try:
                df = safe_call(
                    run_breakout_scan,
                    tickers,
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    diagnostics=diagnostics,
                    label="run_breakout_scan",
                )
                dt = time.time() - t0
                st.session_state.results_df = df
                banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")
            except Exception as e:
                banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

    if run_sp500_btn:
        do_scan(sp500, "SP500")
    if run_nasdaq_btn:
        do_scan(nasdaq, "NASDAQ")
    if run_combo_btn:
        do_scan(sp500 + nasdaq, "Combo")

    df = st.session_state.results_df

    if df is not None and not df.empty:
        st.subheader("Results")
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

    st.divider()
    st.caption("⚠️ Not financial advice. Educational tool only.")


if __name__ == "__main__":
    main()