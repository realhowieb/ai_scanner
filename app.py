from __future__ import annotations

import importlib
import inspect
import os
import re
import time
import traceback
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date

import json

import numpy as np
import pandas as pd
import streamlit as st


# Optional for Yahoo Finance universe fallback
try:
    import requests
except Exception:  # requests may not be installed in some runtimes
    requests = None

# Ensure local project directory is on sys.path so sibling modules (charts, ui, db, etc.) can be imported
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from charts import render_chart_for_ticker
except ModuleNotFoundError:
    try:
        from ui.charts import render_chart_for_ticker  # type: ignore
    except ModuleNotFoundError:
        try:
            from ai_scanner.charts import render_chart_for_ticker  # type: ignore
        except ModuleNotFoundError:
            def render_chart_for_ticker(ticker: str, *args, **kwargs):
                import streamlit as st  # local import to avoid circulars
                st.info("Chart module not available; add charts.py to enable charts.")
try:
    from ai_notes import generate_ai_note
except ModuleNotFoundError:
    try:
        from ui.ai_notes import generate_ai_note  # type: ignore
    except ModuleNotFoundError:
        def generate_ai_note(row: pd.Series) -> str:
            """Heuristic, on-device 'AI' note using the breakout metrics for one ticker.

            Local fallback implementation used when ai_notes module is not available.
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
                f"**{ticker}** currently has a BreakoutScore of **{_fmt(score, 1)}** "
                f"with pattern tag **{pattern}**."
            )

            # Trend + relative strength
            trend_bits = []
            if not pd.isna(trend20):
                trend_bits.append(f"~{_fmt(trend20, 1, '%')} over the last 20 days")
            if not pd.isna(trend10):
                trend_bits.append(f"~{_fmt(trend10, 1, '%')} over the last 10 days")
            if trend_bits:
                parts.append("Price trend: " + ", ".join(trend_bits) + ".")

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
                    "Liquidity check: 20D avg dollar volume is around "
                    f"**${_fmt(dollar_vol/1_000_000, 1)}M**, which helps with entries and exits."
                )

            if not pd.isna(vol20):
                try:
                    vol_val = float(vol20)
                    if vol_val <= 8:
                        vol_comment = "Price action has been relatively quiet (low volatility)."
                    elif vol_val <= 18:
                        vol_comment = "Volatility is moderate and tradable for most setups."
                    else:
                        vol_comment = (
                            "This is a high-volatility name; position sizing and risk management are critical."
                        )
                    parts.append(
                        f"Volatility (20D) sits near **{_fmt(vol20, 1, '%')}**, {vol_comment}"
                    )
                except Exception:
                    pass

            parts.append(
                "This is not a trade recommendation. Consider support/resistance on the chart, "
                "overall market context, and your own risk management rules before acting."
            )

            return "\n\n".join(parts)
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




# ---------- Page config ----------

st.set_page_config(
    page_title="Breakout Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Tiers / Plans ----------

from config import TIERS_CONFIG, STRIPE_MONTHLY_LINKS, STRIPE_YEARLY_LINKS

# Try to import tiering module from different locations
_tiering_mod = _try_import("tiering")
if _tiering_mod is None:
    _tiering_mod = _try_import("auth.tiering")
if _tiering_mod is None:
    _tiering_mod = _try_import("ai_scanner.tiering")
if _tiering_mod is None:
    _tiering_mod = _try_import("ai_scanner.auth.tiering")

if _tiering_mod is not None:
    USERS_DB = getattr(_tiering_mod, "USERS_DB", {})
    ADMIN_USERS = getattr(_tiering_mod, "ADMIN_USERS", set())
    get_user_tier = getattr(_tiering_mod, "get_user_tier")
    Tier = getattr(_tiering_mod, "Tier")
else:
    # Minimal safe fallback so the app can still run in environments
    # where tiering.py is not importable.
    @dataclass
    class Tier:  # type: ignore
        key: str = "basic"
        name: str = "Basic"
        features: list = None
        max_results: int = 25
        is_premium: bool = False

        def __post_init__(self):
            if self.features is None:
                self.features = []

        # Backwards-compatible properties used in the app
        @property
        def can_scan_sp500(self) -> bool:
            return "SP500 Scan" in self.features or "SP500" in self.features

        @property
        def can_scan_nasdaq(self) -> bool:
            return "NASDAQ" in self.features

        @property
        def can_premarket(self) -> bool:
            return "Premarket" in self.features

        @property
        def can_afterhours(self) -> bool:
            return "AfterHours" in self.features

        @property
        def can_unusual_volume(self) -> bool:
            return "UnusualVolume" in self.features

        @property
        def can_export_csv(self) -> bool:
            return "ExportCSV" in self.features

        @property
        def can_ai_notes(self) -> bool:
            return "AI Notes" in self.features

    USERS_DB: Dict[str, Dict[str, str]] = {}
    ADMIN_USERS = set()

    def get_user_tier(username: str, users: Dict[str, Dict[str, str]]) -> Tier:  # type: ignore
        cfg = TIERS_CONFIG.get("basic", TIERS_CONFIG[next(iter(TIERS_CONFIG))])
        return Tier(
            key="basic",
            name=cfg.get("name", "Basic"),
            features=cfg.get("features", []),
            max_results=cfg.get("max_results", 25),
            is_premium=False,
        )

# ---------- Auth ----------
from db.users import seed_neon_users_from_local, load_users, fetch_all_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

from db.engine import get_db_status, get_neon_conn
from ui.auth import auth_ui
from ui.pricing import pricing_sidebar
from ui.admin_users import render_admin_users_panel
from ui.history import render_history_expander
from ui.results import render_results
from ui.scans import render_scan_controls
from ui.universe_panel import render_universe_panel
from ui.filters import render_filters
from ui.db_status import render_db_status_badge

try:
    from ui.universe import (
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )
except ModuleNotFoundError:
    # Fallback for environments where `ai_scanner` is the package root
    from ai_scanner.ui.universe import (
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )
from scan.engine import safe_call, cached_real_scan, _override_last_prices, safe_yf_download

# Optional live price override for the 'Last' column
try:
    import yfinance as yf
except Exception:
    yf = None


# ---------- Main UI ----------

def main():
    st.title("📈 Breakout Stock Scanner")
    st.caption("Money Moves • AI Breakout Score • Subscription Ready")

    authed, username, display_name = auth_ui()
    if not authed:
        st.stop()

    # Seed Neon users table once (no-op if already populated or Neon unavailable)
    try:
        if not st.session_state.get("seeded_neon_users"):
            seed_neon_users_from_local()
            st.session_state["seeded_neon_users"] = True
    except Exception:
        pass

    # Load users once per rerun to avoid redundant Neon hits
    users_map = load_users()

    tier = get_user_tier(username, users_map)

    # Safely derive capability flags from Tier object or its features list
    tier_features = getattr(tier, "features", []) or []

    def _tier_flag(attr_name: str, feature_name: str) -> bool:
        """Return a boolean capability flag for the given tier.

        Prefer an explicit attribute (e.g., tier.can_premarket). If it doesn't exist,
        fall back to checking the feature name inside the tier_features list.
        """
        if hasattr(tier, attr_name):
            try:
                return bool(getattr(tier, attr_name))
            except Exception:
                pass
        # Fallback to feature-name based check
        if feature_name == "SP500 Scan":
            # Some configs might use 'SP500' instead of 'SP500 Scan'
            return ("SP500 Scan" in tier_features) or ("SP500" in tier_features)
        return feature_name in tier_features

    can_scan_sp500 = _tier_flag("can_scan_sp500", "SP500 Scan")
    can_scan_nasdaq = _tier_flag("can_scan_nasdaq", "NASDAQ")
    can_premarket = _tier_flag("can_premarket", "Premarket")
    can_afterhours = _tier_flag("can_afterhours", "AfterHours")
    can_unusual_volume = _tier_flag("can_unusual_volume", "UnusualVolume")
    can_export_csv = _tier_flag("can_export_csv", "ExportCSV")
    can_ai_notes = _tier_flag("can_ai_notes", "AI Notes")

    st.sidebar.markdown(f"### 👤 {display_name}")
    if username in ADMIN_USERS:
        st.sidebar.markdown("**Plan:** `Admin`")
    else:
        st.sidebar.markdown(f"**Plan:** `{tier.name}`")

    # DB status badge
    db_status = render_db_status_badge()

    pricing_sidebar(username, users_map)

    (
        min_gap,
        min_price,
        max_price,
        top_n,
        max_nasdaq_scan,
        max_combo_scan,
        premarket,
        afterhours,
        unusual_vol,
        diagnostics,
    ) = render_filters(tier)

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
    render_universe_panel()

    render_scan_controls(
        can_scan_sp500=can_scan_sp500,
        can_scan_nasdaq=can_scan_nasdaq,
        max_nasdaq_scan=int(max_nasdaq_scan),
        max_combo_scan=int(max_combo_scan),
        min_gap=float(min_gap),
        min_price=float(min_price),
        max_price=float(max_price),
        top_n=int(top_n),
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        unusual_vol=bool(unusual_vol),
        diagnostics=bool(diagnostics),
        username=username,
    )

    df = st.session_state.results_df
    render_results(df,can_export_csv, can_ai_notes, render_chart_for_ticker,generate_ai_note)

    # --- Scan History (DB-backed via local SQLite_ ---
    render_history_expander(db_status)

    # --- Admin Users Page ---
    render_admin_users_panel(username, ADMIN_USERS, db_status)

    st.divider()
    st.caption("⚠️ Not financial advice. Educational tool only.")


if __name__ == "__main__":
    main()