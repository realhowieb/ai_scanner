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


# ---------- Page config ----------

st.set_page_config(
    page_title="Breakout Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Tiers / Plans ----------

# Tier configuration & user tier resolution live in the auth.tiering module.
# Fall back to auth.tiering_fallback if the main module is unavailable.
try:
    from auth.tiering import USERS_DB, ADMIN_USERS, get_user_tier, Tier
except Exception:
    from auth.tiering_fallback import USERS_DB, ADMIN_USERS, get_user_tier, Tier

# ---------- Auth ----------
from db.users import seed_neon_users_from_local, load_users, fetch_all_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

from ui.auth import auth_ui
from ui.pricing import pricing_sidebar
from ui.admin_users import render_admin_users_panel
from ui.history import render_history_expander
from ui.results import render_results
from ui.scans import render_scan_controls
from ui.universe_panel import render_universe_panel, init_universe_state
from ui.filters import render_filters
from ui.db_status import render_db_status_badge
from auth.tiering_utils import derive_tier_flags


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

    # Derive capability flags from the Tier object in a single helper,
    # so the interpretation of features is centralized.
    flags = derive_tier_flags(tier)
    can_scan_sp500 = flags["can_scan_sp500"]
    can_scan_nasdaq = flags["can_scan_nasdaq"]
    can_premarket = flags["can_premarket"]
    can_afterhours = flags["can_afterhours"]
    can_unusual_volume = flags["can_unusual_volume"]
    can_export_csv = flags["can_export_csv"]
    can_ai_notes = flags["can_ai_notes"]

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
    init_universe_state()

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