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
from datetime import datetime, date

import json
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

import psycopg
from psycopg.rows import dict_row

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
        required_cols = {"id", "name", "results_json", "label", "username", "row_count", "duration_sec", "is_snapshot", "created_at"}
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
            label TEXT,
            username TEXT,
            row_count INTEGER,
            duration_sec REAL,
            is_snapshot INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

_ensure_tables()


def get_neon_conn():
    """Return a new Neon PostgreSQL connection using Streamlit secrets.

    Expects st.secrets["neon"]["database_url"] to contain a full Postgres URI, e.g.:
    postgresql://USER:PASSWORD@host/dbname?sslmode=require
    """
    try:
        db_url = st.secrets["neon"]["database_url"]
    except Exception:
        # Secrets not configured; Neon not available in this environment.
        return None

    try:
        # psycopg3: use row_factory=dict_row to return dict-like rows
        conn = psycopg.connect(db_url, row_factory=dict_row)
        return conn
    except Exception as e:
        # Surface an info caption but don't hard-fail the app.
        st.caption(f"⚠️ Neon connection failed: {e}")
        return None


# --- Neon DB Schema Helper ---
def _ensure_neon_runs_schema(conn) -> None:
    """Ensure the Neon 'runs' table exists with the extended schema.

    This is idempotent and safe to call before inserts. It upgrades older
    tables that only had (id, name, results_json, created_at) by adding the
    newer metadata columns as needed.
    """
    cur = conn.cursor()
    # Base table (older deployments may already have this minimal schema)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Incremental column adds for newer metadata
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS label TEXT")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS username TEXT")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS row_count INT")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS duration_sec REAL")
    cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS is_snapshot BOOLEAN DEFAULT FALSE")
    # created_at may already exist; this is just defensive if older schema omitted it
    cur.execute(
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    )


# --- Neon users schema and helpers ---
def _ensure_neon_users_schema(conn):
    """Create Neon users table if missing."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'basic',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def seed_neon_users_from_local():
    """
    If Neon.users is empty, seed it using the hard-coded USERS_DB.
    This runs once, then never again unless Neon is wiped.
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            return

        _ensure_neon_users_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
        row = cur.fetchone()
        count = row[0] if row is not None and len(row) > 0 else 0

        # Nothing in table? → seed from USERS_DB
        if count == 0:
            for uname, cfg in USERS_DB.items():
                cur.execute(
                    "INSERT INTO users (username, full_name, password, tier) VALUES (%s, %s, %s, %s)",
                    (uname, cfg["name"], cfg["password"], cfg["tier"])
                )
            conn.commit()

        cur.close()
        conn.close()
    except Exception:
        # Seeding is best-effort only; never break the app here.
        pass


def load_users():
    """
    Load active users from Neon.
    If Neon is offline or empty, fallback to in-memory USERS_DB.
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            return USERS_DB

        _ensure_neon_users_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT username, full_name, password, tier FROM users WHERE is_active = TRUE")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        out: Dict[str, Dict[str, str]] = {}
        for r in rows:
            uname = r.get("username") if isinstance(r, dict) else r[0]
            full_name = r.get("full_name") if isinstance(r, dict) else r[1]
            password = r.get("password") if isinstance(r, dict) else r[2]
            tier = r.get("tier") if isinstance(r, dict) else r[3]
            if not uname:
                continue
            out[uname] = {
                "name": full_name or uname,
                "password": password or "",
                "tier": tier or "basic",
            }
        return out if out else USERS_DB
    except Exception:
        return USERS_DB


# --- Helper: fetch all users (including inactive) as DataFrame ---
def fetch_all_users() -> pd.DataFrame:
    """Return all users from Neon (including inactive) as a DataFrame.

    Falls back to a DataFrame built from the local USERS_DB if Neon is
    unavailable or the table is empty.
    """
    try:
        conn = get_neon_conn()
        if conn is None:
            # Fallback to USERS_DB
            rows = []
            for uname, cfg in USERS_DB.items():
                rows.append(
                    {
                        "id": None,
                        "username": uname,
                        "full_name": cfg.get("name", uname),
                        "tier": cfg.get("tier", "basic"),
                        "is_active": True,
                        "created_at": None,
                    }
                )
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        _ensure_neon_users_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, full_name, tier, is_active, created_at FROM users ORDER BY id ASC"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return pd.DataFrame()

        out_rows = []
        for r in rows:
            if isinstance(r, dict):
                out_rows.append(
                    {
                        "id": r.get("id"),
                        "username": r.get("username"),
                        "full_name": r.get("full_name"),
                        "tier": r.get("tier"),
                        "is_active": r.get("is_active"),
                        "created_at": r.get("created_at"),
                    }
                )
            else:
                out_rows.append(
                    {
                        "id": r[0],
                        "username": r[1],
                        "full_name": r[2],
                        "tier": r[3],
                        "is_active": r[4],
                        "created_at": r[5],
                    }
                )
        return pd.DataFrame(out_rows)
    except Exception:
        return pd.DataFrame()


# --- DB Status Helper ---
@st.cache_data(show_spinner=False, ttl=60)
def get_db_status() -> str:
    """Return current DB status: 'neon', 'sqlite', or 'none'.

    This is a lightweight, config-based check to keep startup fast. We avoid
    live connections here and rely on real DB calls in save_run/list_runs.
    """
    # If Neon is configured in secrets, treat it as the primary backend.
    try:
        _ = st.secrets["neon"]["database_url"]  # type: ignore[index]
        return "neon"
    except Exception:
        pass

    # If the local SQLite file exists, report sqlite as available.
    try:
        if DB_PATH.exists():
            return "sqlite"
    except Exception:
        pass

    return "none"

def save_run(
    name: str,
    results_json: str,
    *,
    label: Optional[str] = None,
    username: Optional[str] = None,
    row_count: Optional[int] = None,
    duration_sec: Optional[float] = None,
    is_snapshot: bool = False,
) -> None:
    """Save a scan run to Neon PostgreSQL if available, else local SQLite.

    Extra metadata is stored in dedicated columns so history can be queried by
    label/username and used for analytics.
    """
    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            # Ensure Neon table is present and upgraded to the extended schema
            _ensure_neon_runs_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO runs (name, results_json, label, username, row_count, duration_sec, is_snapshot)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (name, results_json, label, username, row_count, duration_sec, is_snapshot),
            )
            conn.commit()
            cur.close()
            conn.close()
            return
    except Exception as e:
        st.caption(f"⚠️ Neon DB write failed, falling back to SQLite: {e}")

    # Fallback to local SQLite
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (name, results_json, label, username, row_count, duration_sec, is_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, results_json, label, username, row_count, duration_sec, int(is_snapshot)),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Fail silently; history is a convenience feature.
        pass


# --- Daily snapshot helper ---
def save_daily_snapshot(label: str, results_json: str, username: Optional[str] = None) -> None:
    """Save a once-per-day snapshot for a given label (e.g., 'SP500', 'NASDAQ').

    This creates a run named 'Daily snapshot YYYY-MM-DD | LABEL' in Neon if available,
    else in local SQLite. If a snapshot with that name already exists for today, it is
    not duplicated.
    """
    # Build deterministic snapshot name for today + label
    today_str = date.today().isoformat()
    snapshot_name = f"Daily snapshot {today_str} | {label}"

    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            # Ensure Neon table is present and upgraded to the extended schema
            _ensure_neon_runs_schema(conn)
            cur = conn.cursor()
            # Check if snapshot already exists
            cur.execute("SELECT 1 FROM runs WHERE name = %s LIMIT 1", (snapshot_name,))
            if cur.fetchone():
                cur.close()
                conn.close()
                return
            # Insert snapshot (marked as is_snapshot)
            cur.execute(
                """
                INSERT INTO runs (name, results_json, label, username, row_count, duration_sec, is_snapshot)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (snapshot_name, results_json, label, username, None, None, True),
            )
            conn.commit()
            cur.close()
            conn.close()
            return
    except Exception:
        # Fall through to SQLite if Neon unavailable or failing
        pass

    # Fallback to local SQLite snapshot storage
    try:
        conn = _get_conn()
        cur = conn.cursor()
        # Ensure table exists (schema already enforced by _ensure_tables)
        cur.execute("SELECT 1 FROM runs WHERE name = ? LIMIT 1", (snapshot_name,))
        if cur.fetchone():
            conn.close()
            return
        cur.execute(
            """
            INSERT INTO runs (name, results_json, label, username, row_count, duration_sec, is_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (snapshot_name, results_json, label, username, None, None, 1),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Snapshots are convenience only; never break the app.
        pass

def list_runs(limit: int = 50):
    """Return a list of recent runs (id, name, created_at), preferring Neon."""
    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, created_at FROM runs ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            out = []
            for r in rows:
                out.append(
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "timestamp": r["created_at"],
                    }
                )
            return out
    except Exception as e:
        st.caption(f"⚠️ Neon DB read failed, falling back to SQLite: {e}")

    # Fallback to local SQLite
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
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "timestamp": r["created_at"],
                }
            )
        return out
    except Exception:
        return []

def load_run_results(run_id: int) -> str:
    """Return the JSON payload for a given run id, preferring Neon."""
    # Try Neon first
    try:
        conn = get_neon_conn()
        if conn is not None:
            cur = conn.cursor()
            cur.execute("SELECT results_json FROM runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return row["results_json"]
    except Exception:
        # Fall through to SQLite
        pass

    # Fallback to local SQLite
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


# ---------- Safe yfinance batch download helper ----------
def safe_yf_download(
    tickers: List[str],
    *,
    period: str = "1mo",
    interval: str = "1d",
    group_by: str = "ticker",
) -> pd.DataFrame:
    """Batch yfinance.download with retries and minimal overhead."""
    if yf is None or not tickers:
        return pd.DataFrame()

    tickers_str = " ".join(ticker for ticker in tickers if isinstance(ticker, str) and ticker)

    def _download():
        return yf.download(
            tickers=tickers_str,
            period=period,
            interval=interval,
            group_by=group_by,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    try:
        return safe_call(_download, label=f"yfinance batch ({len(tickers)} symbols)")
    except Exception:
        return pd.DataFrame()


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
    # -----------------------
    # BASIC TIER USERS
    # -----------------------
    "basic1": {
        "name": "Basic User 1",
        "password": "test123",
        "tier": "basic",
    },
    "basic2": {
        "name": "Basic User 2",
        "password": "test123",
        "tier": "basic",
    },
    "basic3": {
        "name": "Basic User 3",
        "password": "test123",
        "tier": "basic",
    },

    # -----------------------
    # PRO TIER USERS
    # -----------------------
    "pro1": {
        "name": "Pro User 1",
        "password": "pro123",
        "tier": "pro",
    },
    "pro2": {
        "name": "Pro User 2",
        "password": "pro123",
        "tier": "pro",
    },
    "pro3": {
        "name": "Pro User 3",
        "password": "pro123",
        "tier": "pro",
    },

    # -----------------------
    # PREMIUM TIER USERS
    # -----------------------
    "premium1": {
        "name": "Premium User 1",
        "password": "premium123",
        "tier": "premium",
    },
    "premium2": {
        "name": "Premium User 2",
        "password": "premium123",
        "tier": "premium",
    },
    "premium3": {
        "name": "Premium User 3",
        "password": "premium123",
        "tier": "premium",
    },
}
ADMIN_USERS = {"premium1","howard"}  # usernames allowed to access the Admin Users page


def get_user_tier(username: str) -> Tier:
    users = load_users()
    tier_key = users.get(username, {}).get("tier", "basic")
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

    users_map = load_users()
    usernames = list(users_map.keys())
    authenticator = stauth.Authenticate(
        {"usernames": {u: {"name": users_map[u]["name"], "password": users_map[u]["password"]} for u in usernames}},
        "breakout_scanner_cookie",
        "breakout_scanner_signature",
        cookie_expiry_days=7,
    )

    # New API (v0.3+): login() returns None for rendered locations; values are in st.session_state
    try:
        authenticator.login(
            "main",
            fields={
                "Form name": "Login",
                "Username": "Username",
                "Password": "Password",
                "Login": "Login",
            },
        )
    except Exception as e:
        banner(f"Auth error: {e}", "error")
        return False, None, None

    auth_status = st.session_state.get("authentication_status", None)
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        banner("Username/password incorrect", "error")
        return False, None, None
    if auth_status is None:
        banner("Please enter your credentials.", "info")
        return False, None, None

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    return True, username, name


def pricing_sidebar(current_username: Optional[str]):
    """Show upgrade options only for tiers above the user's current plan.

    - Basic users see Pro + Premium
    - Pro users see Premium
    - Premium users see a small thank-you message (no upgrade cards)
    """
    tiers_order = ["basic", "pro", "premium"]
    users = load_users()
    current_key = users.get(current_username or "", {}).get("tier", "basic")
    try:
        start_idx = tiers_order.index(current_key) + 1
    except ValueError:
        start_idx = 1  # default: treat as basic if unknown

    upsell_keys = tiers_order[start_idx:]

    if not upsell_keys:
        st.sidebar.markdown("## 💳 Upgrade")
        st.sidebar.caption("You're on the top Premium plan. Thank you for subscribing!")
        return

    st.sidebar.markdown("## 💳 Upgrade")
    cols = st.sidebar.columns(len(upsell_keys))
    for i, key in enumerate(upsell_keys):
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

    batch = safe_yf_download(tickers, period="5d", interval="1d", group_by="ticker")
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
# Real AI-style breakout scanner (no external module required).


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
    """Premium Breakout v2.1 Engine (Balanced mode).

    Signals:
      - Gap% vs previous close
      - BreakoutPos20D: last close vs 20-day high
      - Trend20D%: 20-day trend
      - Trend10D%: shorter-term trend (multi-timeframe check)
      - VolRel20: last volume / 20-day average
      - DollarVol20: avg volume * last price (liquidity)
      - Volatility20D%: std dev of daily returns
      - RS_Rank: percentile rank of Trend20D% across the universe

    Goals:
      - De-emphasize illiquid / junky microcaps
      - Reward liquid names with strong, sustainable momentum
      - Provide extra columns to interpret each score (PatternTag, ScoreNote, RS_Rank).
    """

    # ---------- Stub path if yfinance is unavailable ----------
    if yf is None or not tickers:
        rows = []
        for t in tickers:
            price = float(np.random.uniform(min_price, max_price))
            vol = int(np.random.randint(300_000, 20_000_000))
            gap = float(np.random.uniform(min_gap, min_gap + 10))
            score = float(np.random.uniform(0, 100))
            rows.append(
                {
                    "Ticker": t,
                    "BreakoutScore": round(score, 2),
                    "Last": round(price, 2),
                    "Volume": vol,
                    "Gap%": round(gap, 2),
                    "BreakoutPos20D": np.nan,
                    "Trend20D%": np.nan,
                    "Trend10D%": np.nan,
                    "VolRel20": np.nan,
                    "DollarVol20": np.nan,
                    "Volatility20D%": np.nan,
                    "RS_Rank": np.nan,
                    "PatternTag": "Stub",
                    "ScoreNote": "Stub/no yfinance.",
                    "Premarket": premarket,
                    "AfterHours": afterhours,
                    "UnusualVol": False,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df[df["Last"].between(min_price, max_price)]
        df = df[df["Gap%"] >= min_gap]
        df = df.sort_values("BreakoutScore", ascending=False).head(top_n).reset_index(drop=True)
        return df

    # ---------- Batch download (1mo daily) ----------
    batch = safe_yf_download(tickers, period="1mo", interval="1d", group_by="ticker")
    if batch is None or batch.empty:
        return pd.DataFrame()

    rows: List[Dict] = []

    def _get_series(sym: str, field: str) -> Optional[pd.Series]:
        if batch is None or batch.empty:
            return None
        if isinstance(batch.columns, pd.MultiIndex):
            # Try both orientations: (ticker, field) and (field, ticker)
            try:
                return batch[(sym, field)].dropna()
            except Exception:
                try:
                    return batch[(field, sym)].dropna()
                except Exception:
                    return None
        # Single-ticker case
        try:
            return batch[field].dropna()
        except Exception:
            return None

    # ---------- Per-ticker feature extraction ----------
    for sym in tickers:
        try:
            close = _get_series(sym, "Close")
            high = _get_series(sym, "High")
            vol = _get_series(sym, "Volume")
            if close is None or high is None or vol is None:
                continue
            if len(close) < 5 or len(vol) < 5:
                continue

            close = close.dropna()
            high = high.dropna()
            vol = vol.dropna()
            if close.empty or high.empty or vol.empty:
                continue

            last_close = float(close.iloc[-1])
            if last_close <= 0:
                continue

            # Basic price band filter
            if not (min_price <= last_close <= max_price):
                continue

            # Gap% vs previous close
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
            gap_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

            # 20-day high breakout position
            window_h = min(20, len(high))
            high20 = float(high.tail(window_h).max()) if window_h > 0 else last_close
            breakout_pos = (last_close / high20) if high20 > 0 else 0.0  # ~1.0 near 20D high

            # 20-day and 10-day trend (multi-timeframe)
            if len(close) >= 20:
                past20 = float(close.iloc[-20])
            else:
                past20 = float(close.iloc[0])
            trend20 = ((last_close - past20) / past20) * 100 if past20 > 0 else 0.0

            if len(close) >= 10:
                past10 = float(close.iloc[-10])
            else:
                past10 = float(close.iloc[0])
            trend10 = ((last_close - past10) / past10) * 100 if past10 > 0 else 0.0

            # Volume relative to 20-day average + dollar volume
            window_v = min(20, len(vol))
            avg_vol20 = float(vol.tail(window_v).mean()) if window_v > 0 else float(vol.iloc[-1])
            last_vol = float(vol.iloc[-1])
            vol_rel20 = (last_vol / avg_vol20) if avg_vol20 > 0 else 1.0
            dollar_vol20 = avg_vol20 * last_close

            # Volatility: std of daily returns over last 20 bars (in %)
            returns = close.pct_change().dropna()
            if not returns.empty:
                tail = returns.tail(min(20, len(returns)))
                vol20_pct = float(tail.std() * 100.0)
            else:
                vol20_pct = 0.0

            # ---------- Balanced liquidity + quality filters ----------
            # Raise the bar from v2.0: target liquid, institution-grade names.
            min_avg_vol = 200_000
            min_dollar_vol = 10_000_000  # ~$10M+
            if avg_vol20 < min_avg_vol:
                continue
            if dollar_vol20 < min_dollar_vol:
                continue
            # Avoid super-thin moves where current volume is way below normal
            if vol_rel20 < 0.8:
                continue

            # Unusual volume filter (if enabled)
            if unusual_volume and vol_rel20 < 1.5:
                continue

            # Min gap filter
            if gap_pct < min_gap:
                continue

            # ---------- Premium BreakoutScore v2.1 ----------
            # Normalize components. Heuristic scaling so each term lives in a
            # moderate range before weighting.
            comp_gap = max(0.0, gap_pct) / 4.0                 # strong gaps rewarded
            comp_breakout = max(0.0, breakout_pos - 0.9) * 15.0   # near 20D high
            comp_trend20 = max(0.0, trend20) / 6.0            # sustained trend
            comp_trend10 = max(0.0, trend10) / 4.0            # short-term momentum
            comp_vol_rel = max(0.0, vol_rel20 - 1.0) * 3.0
            price_factor = np.clip(last_close / 20.0, 0.2, 1.5)   # de-weight sub-$5, reward $20+
            dv_component = np.clip((np.log10(dollar_vol20 + 1) - 5.5), 0.0, 4.0)

            # Volatility penalty: stronger than v2.0 for wild names
            vol_penalty = np.clip((vol20_pct - 10.0) / 3.0, 0.0, 5.0)

            raw_score = (
                0.20 * comp_gap
                + 0.22 * comp_breakout
                + 0.18 * comp_trend20
                + 0.14 * comp_trend10
                + 0.14 * comp_vol_rel
                + 0.12 * dv_component
            )

            raw_score = raw_score * float(price_factor) - 0.15 * vol_penalty
            score = float(np.clip(raw_score * 10.0, 0.0, 100.0))

            # ---------- Simple pattern tagging ----------
            pattern_tags = []
            if breakout_pos >= 0.98 and trend20 > 0 and gap_pct >= min_gap:
                pattern_tags.append("BreakoutHigh")
            if trend20 > 20 and vol_rel20 >= 1.3:
                pattern_tags.append("Momentum")
            if trend20 > 10 and vol20_pct <= 8:
                pattern_tags.append("SteadyClimb")
            if not pattern_tags and vol20_pct >= 20 and gap_pct >= 5:
                pattern_tags.append("HighVolRunner")
            if not pattern_tags:
                pattern_tags.append("Base/Neutral")
            pattern_tag = ",".join(pattern_tags)

            # Short note to explain score drivers
            score_factors = []
            if comp_gap > 0:
                score_factors.append("Gap")
            if comp_breakout > 0:
                score_factors.append("NearHigh")
            if comp_trend20 > 0:
                score_factors.append("Trend20D")
            if comp_trend10 > 0:
                score_factors.append("Trend10D")
            if comp_vol_rel > 0:
                score_factors.append("Volume")
            if dv_component > 0:
                score_factors.append("Liquidity")
            if vol_penalty > 0:
                score_factors.append("VolPenalty")
            score_note = "+".join(score_factors) if score_factors else "Neutral mix"

            rows.append(
                {
                    "Ticker": sym,
                    "BreakoutScore": round(score, 2),
                    "Last": round(last_close, 2),
                    "Volume": int(last_vol),
                    "Gap%": round(gap_pct, 2),
                    "BreakoutPos20D": round(breakout_pos, 3),
                    "Trend20D%": round(trend20, 2),
                    "Trend10D%": round(trend10, 2),
                    "VolRel20": round(vol_rel20, 2),
                    "DollarVol20": round(dollar_vol20, 2),
                    "Volatility20D%": round(vol20_pct, 2),
                    "PatternTag": pattern_tag,
                    "ScoreNote": score_note,
                    # Placeholder; RS_Rank filled after DataFrame construction.
                    "RS_Rank": np.nan,
                    "Premarket": premarket,
                    "AfterHours": afterhours,
                    "UnusualVol": unusual_volume and vol_rel20 >= 1.5,
                }
            )
        except Exception:
            # Skip broken symbols silently; diagnostics mode can be extended later.
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---------- RS Rank across universe ----------
    try:
        if "Trend20D%" in df.columns and len(df) > 1:
            df["RS_Rank"] = df["Trend20D%"].rank(pct=True) * 100.0
            df["RS_Rank"] = df["RS_Rank"].round(1)
    except Exception:
        pass

    # Final sorting and Top N cap.
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


def generate_ai_note(row: pd.Series) -> str:
    """Heuristic, on-device "AI" note using the breakout metrics for one ticker.

    This keeps the app self-contained (no external API calls) and explains why a
    name is scoring well or poorly based on its columns.
    """
    def _fmt(val, nd=2, suffix=""):
        try:
            if pd.isna(val):
                return "N/A"
            return f"{float(val):.{nd}f}{suffix}"
        except Exception:
            return "N/A"

    ticker = row.get("Ticker", "?")
    score = row.get("BreakoutScore", np.nan)
    pattern = row.get("PatternTag", "") or "Neutral"
    gap = row.get("Gap%", np.nan)
    trend20 = row.get("Trend20D%", np.nan)
    trend10 = row.get("Trend10D%", np.nan)
    vol_rel = row.get("VolRel20", np.nan)
    rs = row.get("RS_Rank", np.nan)
    vol20 = row.get("Volatility20D%", np.nan)
    dollar_vol = row.get("DollarVol20", np.nan)

    parts = []

    # High-level summary
    parts.append(
        f"**{ticker}** currently has a BreakoutScore of **{_fmt(score, 1)}** with pattern tag **{pattern}**."
    )

    # Trend + relative strength
    trend_bits = []
    if not pd.isna(trend20):
        trend_bits.append(f"~{_fmt(trend20, 1, '%')} over the last 20 days")
    if not pd.isna(trend10):
        trend_bits.append(f"~{_fmt(trend10, 1, '%')} over the last 10 days")
    if trend_bits:
        line = "Price trend: " + ", ".join(trend_bits) + "."
        parts.append(line)

    if not pd.isna(rs):
        try:
            rs_val = float(rs)
            if rs_val >= 80:
                rs_comment = "strong relative strength vs the universe (top 20%)."
            elif rs_val >= 60:
                rs_comment = "above-average relative strength (top 40%)."
            elif rs_val >= 40:
                rs_comment = "roughly middle-of-the-pack relative strength."
            else:
                rs_comment = "weak relative strength vs peers right now."
            parts.append(f"RS Rank is **{_fmt(rs, 1)}**, indicating {rs_comment}")
        except Exception:
            pass

    # Gap + volume behaviour
    gap_bits = []
    if not pd.isna(gap):
        gap_bits.append(f"gap of {_fmt(gap, 1, '%')} vs the prior close")
    if not pd.isna(vol_rel):
        gap_bits.append(f"volume running at roughly {_fmt(vol_rel, 2)}x the 20D average")
    if gap_bits:
        parts.append("Today it is showing a " + " and ".join(gap_bits) + ".")

    if not pd.isna(dollar_vol):
        parts.append(
            f"Liquidity check: 20D avg dollar volume is around **${_fmt(dollar_vol/1_000_000, 1)}M**, "
            "which helps with entries and exits."
        )

    if not pd.isna(vol20):
        try:
            vol_val = float(vol20)
            if vol_val <= 8:
                vol_comment = "Price action has been relatively quiet (low volatility)."
            elif vol_val <= 18:
                vol_comment = "Volatility is moderate and tradable for most setups."
            else:
                vol_comment = "This is a high-volatility name; position sizing and risk management are critical."
            parts.append(f"Volatility (20D) sits near **{_fmt(vol20, 1, '%')}**, {vol_comment}")
        except Exception:
            pass

    parts.append(
        "This is not a trade recommendation. Consider support/resistance on the chart, overall market context, "
        "and your own risk management rules before acting."
    )

    return "\n\n".join(parts)

# ---------- Main UI ----------

def main():
    st.title("📈 Breakout Stock Scanner")
    st.caption("Money Moves • AI Breakout Score • Subscription Ready")

    authed, username, display_name = auth_ui()
    if not authed:
        st.stop()

    # Seed Neon users table once (no-op if already populated or Neon unavailable)
    try:
        seed_neon_users_from_local()
    except Exception:
        pass

    tier = get_user_tier(username)

    st.sidebar.markdown(f"### 👤 {display_name}")
    st.sidebar.markdown(f"**Plan:** `{tier.name}`")

    # DB status badge
    try:
        db_status = get_db_status()
    except Exception:
        db_status = "none"

    if db_status == "neon":
        st.sidebar.markdown("🟢 **DB:** Neon (cloud)")
    elif db_status == "sqlite":
        st.sidebar.markdown("🟡 **DB:** Local SQLite")
    else:
        st.sidebar.markdown("🔴 **DB:** Unavailable")

    pricing_sidebar(username)

    # Sidebar filters
    st.sidebar.markdown("## Filters")
    min_gap = st.sidebar.slider("Min Gap %", -10.0, 20.0, 1.0, 0.5)
    min_price = st.sidebar.number_input("Min Price", 0.5, 500.0, 1.0, 0.5)
    max_price = st.sidebar.number_input("Max Price", 1.0, 5000.0, 1000.0, 1.0)
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
    unusual_vol = st.sidebar.checkbox("Unusual Volume Filter", value=False, disabled=not tier.can_unusual_volume)

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox("Show diagnostics", value=True)

    # Universe state (lazy-loaded on first scan to keep startup fast)
    if "sp500_universe" not in st.session_state:
        st.session_state["sp500_universe"] = []
    if "nasdaq_universe" not in st.session_state:
        st.session_state["nasdaq_universe"] = []
    if "nasdaq_capped" not in st.session_state:
        st.session_state["nasdaq_capped"] = []
    if "combo_capped" not in st.session_state:
        st.session_state["combo_capped"] = []

    # Universe diagnostics (lazy; based on last scan)
    with st.expander("Universe Info", expanded=True):
        sp500 = st.session_state.get("sp500_universe", [])
        nasdaq_full = st.session_state.get("nasdaq_universe", [])
        nasdaq_capped = st.session_state.get("nasdaq_capped", [])

        if sp500 or nasdaq_full:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**SP500 size:** {len(sp500)}" if sp500 else "**SP500 size:** (not loaded yet)")
                if sp500:
                    st.caption(f"Sample: {', '.join(sp500[:20])}")
            with c2:
                if nasdaq_full:
                    st.markdown(f"**NASDAQ size:** {len(nasdaq_capped) or len(nasdaq_full)}"
                                f"{' (capped)' if nasdaq_capped else ''}")
                    st.caption(f"Sample: {', '.join((nasdaq_capped or nasdaq_full)[:20])}")
                else:
                    st.markdown("**NASDAQ size:** (not loaded yet)")
                    st.caption("Run a NASDAQ or Combo scan to populate NASDAQ universe.")
        else:
            st.caption("Universes will appear here after you run your first scan (SP500, NASDAQ, or Combo).")

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
            n_input = len(tickers)
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

                    if premarket or afterhours:
                        df = _override_last_prices(df)

                filtered_count = len(df) if df is not None else 0
                if diagnostics:
                    st.caption(f"📊 Filtered down from {n_input} tickers to {filtered_count} results after filters.")

                st.caption(f"✅ {label}: {len(df)} results returned from scan.")
                dt = time.time() - t0
                st.session_state.results_df = df
                banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")
                # Persist this scan to the runs DB (history + optional daily snapshot)
                try:
                    results_json = df.to_json(orient="records") if df is not None else "[]"
                    row_count = len(df) if df is not None else 0
                    run_name = f"{label} | {row_count} results | {dt:.1f}s"
                    save_run(
                        run_name,
                        results_json,
                        label=label,
                        username=username,
                        row_count=row_count,
                        duration_sec=dt,
                        is_snapshot=False,
                    )

                    # Morning snapshot: one per day per label (approx. before noon server time)
                    try:
                        current_hour = datetime.now().hour
                        if current_hour < 12:
                            save_daily_snapshot(label, results_json, username=username)
                    except Exception:
                        # Snapshot is best-effort only
                        pass
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
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        sp500 = filter_universe(sp500)
        st.session_state["sp500_universe"] = sp500
        do_scan(sp500, "SP500")

    if run_nasdaq_btn:
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        do_scan(nasdaq_capped, "NASDAQ")

    if run_combo_btn:
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        sp500 = filter_universe(sp500)
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        combo_universe = sp500 + nasdaq_capped
        combo_liquid = apply_liquidity_filter_batch(combo_universe)
        combo_capped = combo_liquid[: int(max_combo_scan)]

        st.session_state["sp500_universe"] = sp500
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        st.session_state["combo_capped"] = combo_capped

        do_scan(combo_capped, "Combo")

    df = st.session_state.results_df

    if df is not None and not df.empty:
        st.subheader("Results")
        st.caption(
            f"Showing {len(df)} results. Increase 'Top N Results' in the sidebar to see more, "
            "or relax filters (Min Gap %, price range, Unusual Volume Filter). "
            "If you see 0 results, try lowering Min Gap or turning off the Unusual Volume Filter."
        )

        # --- Pro styling for results table ---
        styled = df.style

        # Heatmap for BreakoutScore
        if "BreakoutScore" in df.columns:
            styled = styled.background_gradient(axis=None, cmap="RdYlGn", subset=["BreakoutScore"])

        # Conditional formatting for RS_Rank (0-100)
        if "RS_Rank" in df.columns:
            styled = styled.background_gradient(axis=None, cmap="Greens", subset=["RS_Rank"])

        # Bold / color trend markers
        def _trend_style(series: pd.Series):
            styles = []
            for v in series:
                try:
                    val = float(v)
                except Exception:
                    styles.append("")
                    continue
                if val >= 20:
                    styles.append("font-weight: bold; color: #006400;")  # strong uptrend
                elif val <= -10:
                    styles.append("font-weight: bold; color: #8B0000;")  # strong downtrend
                else:
                    styles.append("")
            return styles

        if "Trend20D%" in df.columns:
            styled = styled.apply(_trend_style, subset=["Trend20D%"])
        if "Trend10D%" in df.columns:
            styled = styled.apply(_trend_style, subset=["Trend10D%"])

        st.dataframe(styled, use_container_width=True, height=420)

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

        # AI notes (tier-gated)
        if tier.can_ai_notes:
            st.subheader("AI Notes (Premium)")
            try:
                # Use the same ticker the user selected for the chart
                row = df[df["Ticker"] == pick].iloc[0]
                auto_note = generate_ai_note(row)
                st.markdown(auto_note)
                st.text_area(
                    "Edit or copy these notes (Premium only):",
                    value=auto_note,
                    height=220,
                )
            except Exception:
                st.caption("AI notes are unavailable for the selected row.")
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
                    if db_status == "neon":
                        st.caption(
                            "History is stored in Neon (cloud Postgres). Local scanner.sqlite is used as a fallback."
                        )
                    elif db_status == "sqlite":
                        st.caption(
                            "History is stored in a local scanner.sqlite file next to app.py."
                        )
                    else:
                        st.caption(
                            "History storage backend is currently unavailable."
                        )
            else:
                st.caption("No past scans saved yet.")
        else:
            st.caption("No past scans saved yet.")

    # --- Admin Users Page ---
    if username in ADMIN_USERS:
        with st.expander("👑 Admin: Manage Users", expanded=False):
            # --- Create New User ---
            st.subheader("➕ Create New User")

            new_username = st.text_input("New Username")
            new_full_name = st.text_input("Full Name")
            new_password = st.text_input("Password", type="password")
            new_tier_create = st.selectbox("Tier", ["basic", "pro", "premium"], key="create_user_tier")
            new_active_create = st.checkbox("Active", value=True, key="create_user_active")

            if st.button("Create User"):
                if not new_username or not new_full_name or not new_password:
                    st.error("All fields are required.")
                else:
                    try:
                        conn = get_neon_conn()
                        if conn is None:
                            st.error("Neon connection unavailable; cannot create user.")
                        else:
                            _ensure_neon_users_schema(conn)
                            cur = conn.cursor()
                            cur.execute(
                                """
                                INSERT INTO users (username, full_name, password, tier, is_active)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (username) DO NOTHING
                                """,
                                (new_username, new_full_name, new_password, new_tier_create, new_active_create),
                            )
                            conn.commit()
                            cur.close()
                            conn.close()

                            # Clear cache so new user is available immediately
                            try:
                                load_users.clear()  # type: ignore
                            except Exception:
                                pass

                            st.success(f"User '{new_username}' created successfully!")
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to create user: {e}")

            # --- Existing Users Table + Edit UI ---
            users_df = fetch_all_users()
            if users_df is None or users_df.empty:
                st.caption("No users found in Neon users table.")
            else:
                st.caption("View and edit user tiers. Changes apply to Neon-backed accounts.")
                st.dataframe(
                    users_df[["id", "username", "full_name", "tier", "is_active", "created_at"]],
                    use_container_width=True,
                    height=260,
                )

                usernames_list = users_df["username"].tolist()
                selected_user = st.selectbox("Select user to edit", usernames_list)
                row = users_df[users_df["username"] == selected_user].iloc[0]

                new_tier = st.selectbox(
                    "Tier",
                    ["basic", "pro", "premium"],
                    index=["basic", "pro", "premium"].index(
                        row["tier"] if row["tier"] in ["basic", "pro", "premium"] else "basic"
                    ),
                    key="edit_user_tier",
                )
                new_active = st.checkbox("Active", value=bool(row["is_active"]), key="edit_user_active")

                if st.button("Save User Changes"):
                    try:
                        conn = get_neon_conn()
                        if conn is None:
                            st.error("Neon connection unavailable; cannot update users.")
                        else:
                            _ensure_neon_users_schema(conn)
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE users SET tier = %s, is_active = %s WHERE username = %s",
                                (new_tier, new_active, selected_user),
                            )
                            conn.commit()
                            cur.close()
                            conn.close()
                            # Clear cached users so changes are picked up immediately
                            try:
                                load_users.clear()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            st.success(
                                f"Updated user '{selected_user}' to tier '{new_tier}' (active={new_active})."
                            )
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to update user: {e}")

    st.divider()
    st.caption("⚠️ Not financial advice. Educational tool only.")


if __name__ == "__main__":
    main()